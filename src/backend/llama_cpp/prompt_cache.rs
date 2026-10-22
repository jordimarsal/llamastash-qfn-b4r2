//! Keeping a llama.cpp launch's prompt cache across an unload.
//!
//! `llama-server` can write one slot's KV state to a file (`--slot-save-path`
//! plus `POST /slots/{id_slot}?action=save`) and read it back into a slot of a
//! later launch (`?action=restore`). A slot that was restored answers the next
//! request to the same conversation by processing only what came after the saved
//! prompt.
//!
//! Two hooks on the launch lifecycle, both best-effort and both silent when
//! there is nothing to do:
//!
//! * [`save_before_stop`] runs while the engine is still up, immediately before
//!   an unload takes the launch down — the only moment the KV state exists.
//! * [`restore_before_ready`] runs on a new launch's `Loading → Ready`
//!   transition, which is before that launch can serve anything.
//!
//! What the engine is told: each model file gets a directory under the daemon's
//! cache root, and the launch is handed that directory as its
//! `--slot-save-path`. A dump is only ever read back by the model that wrote it,
//! and it carries the slot's own token sequence: llama.cpp matches an incoming
//! request against those tokens and reuses only the prefix that actually
//! matches, so nothing here has to decide which conversation a file belongs to.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use crate::daemon::supervisor::ManagedModel;
use crate::launch::params::LaunchParams;

/// `launch_config` key holding the directory this launch's slot dumps live in.
/// Seeded by the backend from config plus the resolved binary's caps; its
/// absence is what turns the feature off for a launch — `compose` emits
/// `--slot-save-path` only when it is set, and the save / restore hooks do
/// nothing without it.
pub const LLAMACPP_KNOB_PROMPT_CACHE_DIR: &str = "prompt_cache_dir";
/// `launch_config` keys for the two limits the save hook enforces, projected
/// with the directory so the hooks need no config access of their own.
pub const LLAMACPP_KNOB_PROMPT_CACHE_MAX_BYTES: &str = "prompt_cache_max_bytes";
pub const LLAMACPP_KNOB_PROMPT_CACHE_MAX_AGE_SECS: &str = "prompt_cache_max_age_secs";

/// `Server::caps` key: this `llama-server` build understands
/// `--slot-save-path` and the slot save / restore endpoints.
pub const CAP_SLOT_SAVE: &str = "slot_save";

/// Directory under the daemon's cache root holding every model's dumps.
pub const CACHE_SUBDIR: &str = "prompt-cache";

/// Factory `backend.llamacpp.prompt_cache_max_mib`. A 100k-token dump of a
/// small model is ~3 GiB (32.8 KB per token for Llama-3.2-1B); the cap is what
/// keeps a machine that launches several large-context models from filling its
/// cache disk with state nobody will ask for again.
pub const DEFAULT_PROMPT_CACHE_MAX_MIB: u32 = 8192;
/// Factory `backend.llamacpp.prompt_cache_max_age_secs`. A conversation that
/// returns after an hour of the model being unloaded is rare enough that
/// carrying its state that long costs more than reprocessing it.
pub const DEFAULT_PROMPT_CACHE_MAX_AGE_SECS: u64 = 3600;

/// Upper bound on the slot dumps one unload writes. llama.cpp defaults a launch
/// to four parallel slots; the conversations that matter are the ones a request
/// actually landed on, and a bounded count keeps a launch that has been idle
/// for a long time from writing four full-context dumps before a stop.
const MAX_DUMPS: usize = 4;
/// Wall-clock budget for one unload's saves. The stop is what frees the memory
/// a waiting request needs, so saving must not hold it: past the budget the
/// remaining slots are dropped and the stop proceeds.
const SAVE_BUDGET: Duration = Duration::from_secs(10);
/// One slot action is a file write or read, bounded by the size of the dump and
/// the disk. Generous, because the alternative is a hung unload.
const ACTION_TIMEOUT: Duration = Duration::from_secs(60);

const SLOT_PREFIX: &str = "slot-";
const SLOT_SUFFIX: &str = ".kv";

/// What this launch may keep, read off the launch itself: the directory the
/// backend seeded for it plus the limits seeded with it. `None` when it was
/// seeded nothing — which is how a launch that has no conversation to keep, or
/// one the user switched the feature off for, reaches these hooks.
pub(crate) struct Policy {
  /// The directory handed to `--slot-save-path`.
  dir: PathBuf,
  /// Total bytes every model's dumps may occupy together.
  max_bytes: u64,
  /// Older than this and a dump is dropped rather than read back.
  max_age: Option<Duration>,
}

pub(crate) fn policy_for(params: &LaunchParams) -> Option<Policy> {
  let dir = params
    .launch_config
    .get(LLAMACPP_KNOB_PROMPT_CACHE_DIR)
    .filter(|s| !s.is_empty())?;
  let max_bytes = parse_u64(
    params
      .launch_config
      .get(LLAMACPP_KNOB_PROMPT_CACHE_MAX_BYTES),
  )
  .unwrap_or_else(|| u64::from(DEFAULT_PROMPT_CACHE_MAX_MIB) * 1024 * 1024);
  let max_age = match parse_u64(
    params
      .launch_config
      .get(LLAMACPP_KNOB_PROMPT_CACHE_MAX_AGE_SECS),
  ) {
    // An explicit `0` is the user asking for no age limit. A missing key means
    // the launch was built without the seeded policy, so it gets the factory one.
    Some(0) => None,
    Some(secs) => Some(Duration::from_secs(secs)),
    None => Some(Duration::from_secs(DEFAULT_PROMPT_CACHE_MAX_AGE_SECS)),
  };
  Some(Policy {
    dir: PathBuf::from(dir),
    max_bytes,
    max_age,
  })
}

fn parse_u64(value: Option<&String>) -> Option<u64> {
  value.and_then(|s| s.parse::<u64>().ok())
}

/// The cache root the dumps live under, from the log directory the daemon
/// already handed the backend: `<cache>/logs` → `<cache>/prompt-cache`. Same
/// root as everything else the daemon caches, so `LLAMASTASH_CACHE_DIR` moves
/// them with it. `None` when the launch environment carries no log directory.
pub(crate) fn cache_root(log_dir: &Path) -> Option<PathBuf> {
  log_dir
    .parent()
    .filter(|parent| !parent.as_os_str().is_empty())
    .map(|cache| cache.join(CACHE_SUBDIR))
}

/// One directory per model file. The stem keeps it recognizable in a listing;
/// the digest of the path keeps two different files that share a name — or the
/// same name in two directories — from sharing a cache, which would let one
/// model read another's KV state and call that a hit.
pub(crate) fn model_dir(cache_root: &Path, model_path: &Path) -> PathBuf {
  let stem = model_path
    .file_stem()
    .map(|s| sanitize_component(&s.to_string_lossy()))
    .unwrap_or_else(|| "model".to_string());
  let digest = blake3::hash(model_path.to_string_lossy().as_bytes());
  cache_root.join(format!("{stem}-{}", &digest.to_hex()[..8]))
}

/// A directory component from a model name, kept safe for any filesystem:
/// `.`/`..` cannot be a directory we own, and everything outside
/// `[A-Za-z0-9._-]` is replaced.
fn sanitize_component(raw: &str) -> String {
  let out: String = raw
    .chars()
    .map(|c| {
      if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
        c
      } else {
        '-'
      }
    })
    .take(64)
    .collect();
  if out.is_empty() || out == "." || out == ".." {
    "model".to_string()
  } else {
    out
  }
}

/// The name a slot's dump is written under. llama.cpp validates it
/// (`fs_validate_filename`, no subdirectories), so it stays flat: `slot-0.kv`.
pub(crate) fn slot_file(slot: u32) -> String {
  format!("{SLOT_PREFIX}{slot}{SLOT_SUFFIX}")
}

/// The slot a dump belongs to, read back out of its name.
fn slot_of_file(name: &str) -> Option<u32> {
  name
    .strip_prefix(SLOT_PREFIX)?
    .strip_suffix(SLOT_SUFFIX)?
    .parse()
    .ok()
}

/// Write this launch's idle slots to its model directory, then make room for
/// them. Called with the engine still up; every failure is logged and skipped,
/// because the stop that follows is the part that frees the memory.
pub(crate) async fn save_before_stop(model: &ManagedModel) {
  let Some(policy) = policy_for(model.params()) else {
    return;
  };
  if let Err(e) = std::fs::create_dir_all(&policy.dir) {
    log::warn!("prompt cache: cannot create {}: {e}", policy.dir.display());
    return;
  }
  match enforce_limits(&policy) {
    Ok(freed) if freed > 0 => {
      log::info!("prompt cache: dropped {freed} bytes of older prompt caches to stay under the cap")
    }
    Err(e) => log::warn!("prompt cache: cleanup skipped: {e}"),
    _ => {}
  }
  let Some(saved) = save_slots(model.port()).await else {
    return;
  };
  log::info!(
    "prompt cache: kept {} prompt tokens ({} bytes, {} slots) for {} in {} ms — the next request to this conversation is restored instead of reprocessed",
    saved.tokens,
    saved.bytes,
    saved.slots,
    model_label(model),
    saved.ms,
  );
}

/// Ask the engine which slots it has, then save each one under the name its slot
/// id gives it. Bounded twice — `MAX_DUMPS` slots, `SAVE_BUDGET` of wall clock —
/// because the stop that follows is what a waiting request is waiting on, and a
/// launch that has been idle for a week should not write four full-context dumps
/// before it is told to exit.
async fn save_slots(port: u16) -> Option<Saved> {
  let Some(slots) = slot_ids(port).await else {
    log::debug!("prompt cache: launch on port {port} exposes no slots — nothing saved");
    return None;
  };
  if slots.is_empty() {
    return None;
  }
  let started = Instant::now();
  let (mut slots_written, mut tokens, mut bytes) = (0usize, 0u64, 0u64);
  for slot in slots.into_iter().take(MAX_DUMPS) {
    if started.elapsed() > SAVE_BUDGET {
      log::warn!(
        "prompt cache: save budget of {}s reached — stopping with the remaining slots unwritten",
        SAVE_BUDGET.as_secs()
      );
      break;
    }
    match slot_action(port, slot, "save", &slot_file(slot)).await {
      Ok(body) => {
        slots_written += 1;
        tokens += body["n_saved"].as_u64().unwrap_or_default();
        bytes += body["n_written"].as_u64().unwrap_or_default();
      }
      // A slot mid-generation has nothing stable to write. The stop is not the
      // place to argue about it.
      Err(e) => log::debug!("prompt cache: slot {slot} kept nothing: {e}"),
    }
  }
  if slots_written == 0 {
    return None;
  }
  Some(Saved {
    slots: slots_written,
    tokens,
    bytes,
    ms: started.elapsed().as_millis(),
  })
}

/// What one save pass wrote, for the log line and for the tests that check the
/// engine was asked in the shape the engine expects.
struct Saved {
  slots: usize,
  tokens: u64,
  bytes: u64,
  ms: u128,
}

/// Read this launch's model directory back into its slots. Called on the
/// `Loading → Ready` transition rather than at the first request, because a
/// server that has already answered one has busy slots and a restore into a busy
/// slot is refused — so a returning conversation has to find its state already
/// in the KV cache.
pub(crate) async fn restore_before_ready(model: &ManagedModel) {
  let Some(policy) = policy_for(model.params()) else {
    return;
  };
  if model.inflight() > 0 {
    log::debug!("prompt cache: a request is already being served — left the launch alone");
    return;
  }
  let Some(restored) = restore_dumps(model.port(), &policy).await else {
    return;
  };
  log::info!(
    "prompt cache: restored {} prompt tokens ({} slots) for {} in {} ms",
    restored.tokens,
    restored.slots,
    model_label(model),
    restored.ms,
  );
}

/// Hand every dump this model file has back to the engine, oldest slot first.
/// A dump the engine refuses stays on disk: it is usually a size mismatch (this
/// launch was given a smaller context than the one that wrote it), and the next
/// launch may fit it.
async fn restore_dumps(port: u16, policy: &Policy) -> Option<Restored> {
  let dumps = match list_dumps(policy) {
    Ok(dumps) => dumps,
    Err(e) => {
      log::warn!("prompt cache: cannot read {}: {e}", policy.dir.display());
      return None;
    }
  };
  if dumps.is_empty() {
    return None;
  }
  let started = Instant::now();
  let (mut slots_restored, mut tokens) = (0usize, 0u64);
  for dump in dumps.into_iter().take(MAX_DUMPS) {
    if started.elapsed() > SAVE_BUDGET {
      log::warn!("prompt cache: restore budget reached — the rest was left on disk");
      break;
    }
    let Some(name) = dump
      .path
      .file_name()
      .map(|s| s.to_string_lossy().into_owned())
    else {
      continue;
    };
    match slot_action(port, dump.slot, "restore", &name).await {
      Ok(body) => {
        slots_restored += 1;
        tokens += body["n_restored"].as_u64().unwrap_or_default();
      }
      // The engine read the file and said no: the state does not fit this
      // launch's slot (a smaller context, a different slot count). Kept, since
      // a later launch may fit it; the age limit is what eventually drops it.
      Err(e) => log::info!("prompt cache: slot {} was not restored: {e}", dump.slot),
    }
  }
  if slots_restored == 0 {
    return None;
  }
  Some(Restored {
    slots: slots_restored,
    tokens,
    ms: started.elapsed().as_millis(),
  })
}

struct Restored {
  slots: usize,
  tokens: u64,
  ms: u128,
}

fn model_label(model: &ManagedModel) -> String {
  model
    .params()
    .model_path
    .file_name()
    .map(|s| s.to_string_lossy().into_owned())
    .unwrap_or_else(|| model.params().model_path.display().to_string())
}

/// A dump on disk: which slot it belongs to and how big it is.
struct Dump {
  slot: u32,
  path: PathBuf,
}

fn list_dumps(policy: &Policy) -> std::io::Result<Vec<Dump>> {
  let mut out = Vec::new();
  for entry in std::fs::read_dir(&policy.dir)? {
    let entry = entry?;
    let Some(slot) = entry.file_name().to_str().and_then(slot_of_file) else {
      continue;
    };
    let Ok(meta) = entry.metadata() else { continue };
    if !meta.is_file() {
      continue;
    }
    if let Some(max_age) = policy.max_age {
      if dump_age(&meta).is_some_and(|age| age > max_age) {
        let _ = std::fs::remove_file(entry.path());
        log::debug!(
          "prompt cache: dropped a {}s-old prompt cache",
          max_age.as_secs()
        );
        continue;
      }
    }
    out.push(Dump {
      slot,
      path: entry.path(),
    });
  }
  out.sort_by_key(|d| d.slot);
  Ok(out)
}

fn dump_age(meta: &std::fs::Metadata) -> Option<Duration> {
  meta
    .modified()
    .ok()
    .and_then(|m| SystemTime::now().duration_since(m).ok())
}

/// Drop what no longer serves a purpose: every dump older than the age limit,
/// then whole model directories — the one whose newest dump is oldest first —
/// until the rest fits under the byte cap. Returns the bytes freed.
///
/// Runs before a save, which is the one moment the daemon knows it is about to
/// spend disk.
pub(crate) fn enforce_limits(policy: &Policy) -> std::io::Result<u64> {
  let root = policy
    .dir
    .parent()
    .filter(|p| p.file_name().is_some_and(|n| n == CACHE_SUBDIR))
    .ok_or_else(|| {
      std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        format!("unexpected prompt cache layout in {}", policy.dir.display()),
      )
    })?;

  let mut freed = 0u64;
  // Per-model directory: (newest dump in it, total bytes, its path).
  let mut dirs: Vec<(SystemTime, u64, PathBuf)> = Vec::new();
  let Ok(entries) = std::fs::read_dir(root) else {
    return Ok(0);
  };
  for entry in entries.filter_map(|e| e.ok()) {
    if !entry.path().is_dir() {
      continue;
    }
    let mut newest = SystemTime::UNIX_EPOCH;
    let mut bytes = 0u64;
    let Ok(dumps) = std::fs::read_dir(entry.path()) else {
      continue;
    };
    for dump in dumps.filter_map(|d| d.ok()) {
      let Ok(meta) = dump.metadata() else { continue };
      if !meta.is_file() {
        continue;
      }
      if policy
        .max_age
        .is_some_and(|age| dump_age(&meta).is_some_and(|a| a > age))
      {
        if std::fs::remove_file(dump.path()).is_ok() {
          freed += meta.len();
        }
        continue;
      }
      if let Ok(m) = meta.modified() {
        newest = newest.max(m);
      }
      bytes += meta.len();
    }
    if bytes > 0 {
      dirs.push((newest, bytes, entry.path()));
    }
  }

  let mut total: u64 = dirs.iter().map(|(_, bytes, _)| *bytes).sum();
  // Oldest-model-first: the model that has been asked for least recently is the
  // one whose state is least likely to be wanted.
  while total > policy.max_bytes && !dirs.is_empty() {
    let index = dirs
      .iter()
      .enumerate()
      .min_by_key(|(_, (newest, _, _))| *newest)
      .map(|(i, _)| i);
    let Some(index) = index else { break };
    let (_, bytes, path) = dirs.remove(index);
    match std::fs::remove_dir_all(&path) {
      Ok(()) => {
        log::debug!(
          "prompt cache: dropped {} to stay under the cap",
          path.display()
        );
        total = total.saturating_sub(bytes);
        freed += bytes;
      }
      Err(e) => {
        log::warn!("prompt cache: could not drop {}: {e}", path.display());
        break;
      }
    }
  }
  Ok(freed)
}

// ---- llama-server slot API over raw loopback HTTP --------------------------
//
// The same no-dependency stance as [`super::actuals`]: a hand-rolled request on
// `127.0.0.1:<port>` with `Connection: close`, so a save or a restore costs a
// TCP connection and nothing else.

async fn slot_ids(port: u16) -> Option<Vec<u32>> {
  let request = format!(
    "GET /slots HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\nAccept: application/json\r\n\r\n"
  );
  let body = exchange(port, &request, None).await.ok()?;
  parse_slot_ids(&body)
}

/// The slot ids a running launch exposes, from either shape of `GET /slots`:
/// the bare array current builds answer with, or the older `{"slots": [...]}`.
fn parse_slot_ids(body: &Value) -> Option<Vec<u32>> {
  let rows = body
    .as_array()
    .cloned()
    .or_else(|| body.get("slots").and_then(Value::as_array).cloned())?;
  let mut ids = Vec::new();
  for row in rows {
    if let Some(id) = row.get("id").and_then(Value::as_u64) {
      ids.push(id as u32);
    }
  }
  Some(ids)
}

/// `POST /slots/{slot}?action=<save|restore>` for one file in the launch's
/// `--slot-save-path` directory. Ok carries the engine's own report
/// (`n_saved` / `n_written` / `n_restored` and the timings); Err carries the
/// message it gave.
async fn slot_action(port: u16, slot: u32, action: &str, filename: &str) -> Result<Value, String> {
  let body = json!({ "filename": filename }).to_string();
  let request = format!(
    "POST /slots/{slot}?action={action} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
    body.len()
  );
  exchange(port, &request, Some(&body)).await
}

async fn exchange(port: u16, head: &str, body: Option<&str>) -> Result<Value, String> {
  let fut = async {
    let mut sock = TcpStream::connect(("127.0.0.1", port)).await?;
    sock.write_all(head.as_bytes()).await?;
    if let Some(body) = body {
      sock.write_all(body.as_bytes()).await?;
    }
    read_response(&mut sock).await
  };
  let outcome = tokio::time::timeout(ACTION_TIMEOUT, fut)
    .await
    .map_err(|_| String::from("slot action timed out"))?;
  let (status, resp) = outcome.map_err(|e| format!("slot action failed: {e}"))?;
  let parsed: Value = serde_json::from_slice(&resp)
    .map_err(|e| format!("unreadable response from the slot endpoint: {e}"))?;
  if !(200..300).contains(&status) {
    let message = parsed
      .get("error")
      .and_then(|e| e.get("message").and_then(Value::as_str))
      .or_else(|| parsed.as_str())
      .unwrap_or("unknown error");
    return Err(format!("llama-server refused the slot action: {message}"));
  }
  Ok(parsed)
}

/// Read one HTTP/1.1 response: stop at `Content-Length` when the peer declared
/// it, at EOF otherwise (`Connection: close` guarantees one or the other).
async fn read_response(sock: &mut TcpStream) -> std::io::Result<(u16, Vec<u8>)> {
  const CAP: usize = 1024 * 1024;
  let mut buf = Vec::with_capacity(4096);
  let mut chunk = [0u8; 8192];
  loop {
    let n = sock.read(&mut chunk).await?;
    if n == 0 {
      break;
    }
    buf.extend_from_slice(&chunk[..n]);
    if let Some((_, body_len)) = split_headers(&buf) {
      if body_len.is_some_and(|len| buf.len() >= len) {
        break;
      }
    }
    if buf.len() > CAP {
      break;
    }
  }
  let header_end = buf
    .windows(4)
    .position(|w| w == b"\r\n\r\n")
    .map(|p| p + 4)
    .ok_or_else(|| std::io::Error::other("no HTTP header in response"))?;
  let head = String::from_utf8_lossy(&buf[..header_end]);
  let status = head
    .lines()
    .next()
    .and_then(|line| line.split_whitespace().nth(1))
    .and_then(|s| s.parse::<u16>().ok())
    .ok_or_else(|| std::io::Error::other("no status line in response"))?;
  Ok((status, buf[header_end..].to_vec()))
}

/// `(status code, declared body length)` from the header block seen so far —
/// `None` while the block is not complete.
fn split_headers(buf: &[u8]) -> Option<(u16, Option<usize>)> {
  let header_end = buf
    .windows(4)
    .position(|w| w == b"\r\n\r\n")
    .map(|p| p + 4)?;
  let head = String::from_utf8_lossy(&buf[..header_end]);
  let mut lines = head.lines();
  let status = lines
    .next()
    .and_then(|line| line.split_whitespace().nth(1))
    .and_then(|s| s.parse::<u16>().ok())?;
  let length = lines.find_map(|line| {
    let (name, value) = line.split_once(':')?;
    if name.trim().eq_ignore_ascii_case("content-length") {
      value.trim().parse::<usize>().ok()
    } else {
      None
    }
  });
  Some((status, length))
}

#[cfg(test)]
mod tests {
  use super::*;

  fn params_with(config: &[(&str, &str)]) -> LaunchParams {
    let mut params = LaunchParams::new(
      PathBuf::from("/models/DeepSeek-R1-Q4.gguf"),
      crate::launch::mode::LaunchMode::Chat,
    );
    for (key, value) in config {
      params
        .launch_config
        .insert((*key).to_string(), (*value).to_string());
    }
    params
  }

  // ---- policy: the seeded directory is what turns the feature on ----

  #[test]
  fn a_launch_with_no_seeded_directory_keeps_nothing() {
    assert!(policy_for(&params_with(&[])).is_none());
  }

  #[test]
  fn an_empty_directory_value_is_the_same_as_no_directory() {
    let params = params_with(&[(LLAMACPP_KNOB_PROMPT_CACHE_DIR, "")]);
    assert!(policy_for(&params).is_none());
  }

  #[test]
  fn the_seeded_policy_carries_the_limits_that_came_with_it() {
    let params = params_with(&[
      (LLAMACPP_KNOB_PROMPT_CACHE_DIR, "/cache/prompt-cache/x"),
      (LLAMACPP_KNOB_PROMPT_CACHE_MAX_BYTES, "1048576"),
      (LLAMACPP_KNOB_PROMPT_CACHE_MAX_AGE_SECS, "60"),
    ]);
    let policy = policy_for(&params).expect("policy");
    assert_eq!(policy.dir, PathBuf::from("/cache/prompt-cache/x"));
    assert_eq!(policy.max_bytes, 1048576);
    assert_eq!(policy.max_age, Some(Duration::from_secs(60)));
  }

  #[test]
  fn an_age_of_zero_means_keep_dumps_indefinitely() {
    let params = params_with(&[
      (LLAMACPP_KNOB_PROMPT_CACHE_DIR, "/cache/prompt-cache/x"),
      (LLAMACPP_KNOB_PROMPT_CACHE_MAX_AGE_SECS, "0"),
    ]);
    assert_eq!(policy_for(&params).unwrap().max_age, None);
  }

  #[test]
  fn an_unparseable_limit_falls_back_to_the_factory_one() {
    let params = params_with(&[
      (LLAMACPP_KNOB_PROMPT_CACHE_DIR, "/cache/prompt-cache/x"),
      (LLAMACPP_KNOB_PROMPT_CACHE_MAX_BYTES, "not-a-number"),
    ]);
    let policy = policy_for(&params).unwrap();
    assert_eq!(
      policy.max_bytes,
      u64::from(DEFAULT_PROMPT_CACHE_MAX_MIB) * 1024 * 1024
    );
    assert_eq!(
      policy.max_age,
      Some(Duration::from_secs(DEFAULT_PROMPT_CACHE_MAX_AGE_SECS))
    );
  }

  // ---- where the files live ----

  #[test]
  fn the_cache_root_sits_next_to_the_log_dir_the_daemon_handed_us() {
    assert_eq!(
      cache_root(Path::new("/home/u/.cache/llamastash/logs")),
      Some(PathBuf::from("/home/u/.cache/llamastash/prompt-cache"))
    );
    // No log directory (a launch environment built without one) → no feature.
    assert_eq!(cache_root(Path::new("")), None);
  }

  #[test]
  fn one_directory_per_model_file_and_no_collisions_on_a_shared_name() {
    let root = Path::new("/cache/prompt-cache");
    let a = model_dir(root, Path::new("/models/a/Llama-3.2-1B-Q4_K_M.gguf"));
    let b = model_dir(root, Path::new("/models/b/Llama-3.2-1B-Q4_K_M.gguf"));
    assert!(a.starts_with(root));
    assert!(a
      .file_name()
      .unwrap()
      .to_string_lossy()
      .starts_with("Llama-3.2-1B-Q4_K_M-"));
    assert_ne!(
      a, b,
      "two models sharing a file name must not share a cache"
    );
    assert_eq!(
      a,
      model_dir(root, Path::new("/models/a/Llama-3.2-1B-Q4_K_M.gguf"))
    );
  }

  #[test]
  fn a_model_name_is_reduced_to_one_safe_directory_component() {
    assert_eq!(sanitize_component("deepseek-r1:8b"), "deepseek-r1-8b");
    assert_eq!(sanitize_component("///"), "---");
    // A name that is a path traversal, or nothing at all, is not a directory
    // we can own.
    assert_eq!(sanitize_component("."), "model");
    assert_eq!(sanitize_component(".."), "model");
    assert_eq!(sanitize_component(""), "model");
    let long = sanitize_component(&"x".repeat(400));
    assert_eq!(long.len(), 64);
  }

  #[test]
  fn a_slot_file_names_its_slot_and_reads_back() {
    assert_eq!(slot_file(0), "slot-0.kv");
    assert_eq!(slot_of_file("slot-0.kv"), Some(0));
    assert_eq!(slot_of_file("slot-12.kv"), Some(12));
    // Anything we did not write is not ours to read back.
    assert_eq!(slot_of_file("notes.txt"), None);
    assert_eq!(slot_of_file("slot-x.kv"), None);
  }

  // ---- cap and age ----

  fn write_dump(dir: &Path, slot: u32, bytes: usize, age: Duration) {
    std::fs::create_dir_all(dir).expect("create model dir");
    let path = dir.join(slot_file(slot));
    std::fs::write(&path, vec![7u8; bytes]).expect("write dump");
    let touched = SystemTime::now() - age;
    let file = std::fs::File::options()
      .write(true)
      .open(&path)
      .expect("open");
    file
      .set_times(std::fs::FileTimes::new().set_modified(touched))
      .expect("set mtime");
  }

  fn total_in(root: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(root) else {
      return 0;
    };
    entries
      .filter_map(|e| e.ok())
      .filter_map(|e| e.metadata().ok())
      .filter(|m| m.is_file())
      .map(|m| m.len())
      .sum()
  }

  #[test]
  fn an_expired_dump_is_dropped_instead_of_read_back() {
    let root = crate::util::test_temp::unique_temp_dir("prompt-cache-expired");
    let models = root.join(CACHE_SUBDIR);
    write_dump(&models.join("model-a"), 0, 4096, Duration::from_secs(7200));
    let policy = Policy {
      dir: models.join("model-a"),
      max_bytes: u64::MAX,
      max_age: Some(Duration::from_secs(3600)),
    };
    assert_eq!(enforce_limits(&policy).unwrap(), 4096);
    assert_eq!(total_in(&models.join("model-a")), 0);
    std::fs::remove_dir_all(&root).ok();
  }

  #[test]
  fn over_the_cap_the_least_recently_used_model_goes_first() {
    let root = crate::util::test_temp::unique_temp_dir("prompt-cache-cap");
    let models = root.join(CACHE_SUBDIR);
    // "old" was last written an hour ago, "recent" a minute ago: 8 MiB of
    // dumps against a 6 MiB cap has to cost one of them.
    write_dump(
      &models.join("model-old"),
      0,
      4 * 1024 * 1024,
      Duration::from_secs(3600),
    );
    write_dump(
      &models.join("model-recent"),
      0,
      3 * 1024 * 1024,
      Duration::from_secs(60),
    );
    write_dump(
      &models.join("model-recent"),
      1,
      1024 * 1024,
      Duration::from_secs(60),
    );
    let policy = Policy {
      dir: models.join("model-recent"),
      max_bytes: 6 * 1024 * 1024,
      max_age: None,
    };
    enforce_limits(&policy).expect("limits");
    assert!(
      !models.join("model-old").exists(),
      "stale model cache survives"
    );
    assert!(
      models.join("model-recent").exists(),
      "recent cache was dropped"
    );
    assert!(total_in(&models) <= policy.max_bytes);
    std::fs::remove_dir_all(&root).ok();
  }

  #[test]
  fn under_the_cap_nothing_is_touched() {
    let root = crate::util::test_temp::unique_temp_dir("prompt-cache-under");
    let models = root.join(CACHE_SUBDIR);
    write_dump(&models.join("model-a"), 0, 2048, Duration::from_secs(10));
    let policy = Policy {
      dir: models.join("model-a"),
      max_bytes: 1024 * 1024,
      max_age: Some(Duration::from_secs(3600)),
    };
    assert_eq!(enforce_limits(&policy).unwrap(), 0);
    assert_eq!(total_in(&models.join("model-a")), 2048);
    std::fs::remove_dir_all(&root).ok();
  }

  #[test]
  fn a_directory_that_is_not_under_the_prompt_cache_root_is_refused() {
    let root = crate::util::test_temp::unique_temp_dir("prompt-cache-shape");
    let policy = Policy {
      dir: root.join("somewhere-else"),
      max_bytes: 1024,
      max_age: None,
    };
    assert!(enforce_limits(&policy).is_err());
    std::fs::remove_dir_all(&root).ok();
  }

  #[test]
  fn a_fresh_launch_reads_back_only_its_own_slots_in_order() {
    let root = crate::util::test_temp::unique_temp_dir("prompt-cache-list");
    let models = root.join(CACHE_SUBDIR);
    let dir = models.join("model-a");
    write_dump(&dir, 2, 128, Duration::from_secs(5));
    write_dump(&dir, 0, 128, Duration::from_secs(5));
    std::fs::write(dir.join("unrelated.kv"), b"x").expect("foreign file");
    let policy = Policy {
      dir: dir.clone(),
      max_bytes: u64::MAX,
      max_age: Some(Duration::from_secs(3600)),
    };
    let dumps = list_dumps(&policy).expect("list");
    assert_eq!(dumps.iter().map(|d| d.slot).collect::<Vec<_>>(), vec![0, 2]);
    std::fs::remove_dir_all(&root).ok();
  }

  // ---- the slot API's shapes ----

  #[test]
  fn slot_ids_read_both_shapes_of_the_slots_endpoint() {
    let current: Value = serde_json::from_str(r#"[{"id":0,"n_ctx":8192}]"#).unwrap();
    assert_eq!(parse_slot_ids(&current), Some(vec![0]));
    let older: Value =
      serde_json::from_str(r#"{"slots":[{"id":0},{"id":1}],"total_slots":2}"#).unwrap();
    assert_eq!(parse_slot_ids(&older), Some(vec![0, 1]));
    let no_ids: Value = serde_json::from_str(r#"[{"n_ctx":8192}]"#).unwrap();
    assert_eq!(parse_slot_ids(&no_ids), Some(vec![]));
    // `--no-slots` answers something that is not a slot list at all.
    let not_slots: Value = serde_json::from_str(r#"{"error":{"message":"not found"}}"#).unwrap();
    assert_eq!(parse_slot_ids(&not_slots), None);
  }

  #[test]
  fn a_response_is_only_usable_once_the_header_block_is_complete() {
    let partial = b"HTTP/1.1 200 OK\r\nContent-Length: 12";
    assert!(split_headers(partial).is_none());
    let whole = b"HTTP/1.1 200 OK\r\nContent-Length: 12\r\n\r\n{}";
    assert_eq!(split_headers(whole), Some((200, Some(12))));
    let chunked = b"HTTP/1.1 500 Internal Server Error\r\nConnection: close\r\n\r\n";
    assert_eq!(split_headers(chunked), Some((500, None)));
    let junk = b"garbage";
    assert!(split_headers(junk).is_none());
  }

  // ---- the engine-facing half: what a running launch is asked to do ----

  /// A fake `llama-server` slot endpoint on a loopback port: `GET /slots` is
  /// answered with `slots_body`, every `POST /slots/{id}?action=…` with
  /// `action_status` + `action_body`, and each request line and body is handed
  /// back over the channel so a test can read the exact exchange.
  async fn fake_engine(
    slots_body: &'static str,
    action_status: u16,
    action_body: &'static str,
  ) -> (u16, tokio::sync::mpsc::UnboundedReceiver<(String, String)>) {
    use tokio::io::AsyncWriteExt;
    use tokio::net::TcpListener;
    let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    tokio::spawn(async move {
      loop {
        let Ok((mut sock, _)) = listener.accept().await else {
          break;
        };
        let mut buf = [0u8; 4096];
        let n = sock.read(&mut buf).await.unwrap_or(0);
        let raw = String::from_utf8_lossy(&buf[..n]).into_owned();
        let line = raw.lines().next().unwrap_or_default().to_string();
        let body = raw.split("\r\n\r\n").nth(1).unwrap_or_default().to_string();
        let (status, reply) = if line.starts_with("GET /slots") {
          (200, slots_body)
        } else {
          (action_status, action_body)
        };
        let resp = format!(
          "HTTP/1.1 {status} x\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}",
          reply.len(),
        );
        let _ = sock.write_all(resp.as_bytes()).await;
        let _ = tx.send((line, body));
      }
    });
    (port, rx)
  }

  fn captured(
    rx: &mut tokio::sync::mpsc::UnboundedReceiver<(String, String)>,
  ) -> Vec<(String, String)> {
    let mut out = Vec::new();
    while let Ok(request) = rx.try_recv() {
      out.push(request);
    }
    out
  }

  const SAVE_OK: &str = r#"{"id_slot":0,"filename":"slot-0.kv","n_saved":1200,"n_written":39321600,"timings":{"save_ms":900}}"#;

  #[tokio::test]
  async fn an_unload_asks_the_engine_which_slots_it_has_then_writes_each_one() {
    let (port, mut rx) = fake_engine(r#"[{"id":0},{"id":1}]"#, 200, SAVE_OK).await;
    let saved = save_slots(port).await.expect("a save report");
    assert_eq!(saved.slots, 2);
    assert_eq!(saved.tokens, 2400);
    assert_eq!(saved.bytes, 78_643_200);

    let requests = captured(&mut rx);
    let lines: Vec<&str> = requests.iter().map(|(line, _)| line.as_str()).collect();
    assert_eq!(
      lines,
      [
        "GET /slots HTTP/1.1",
        "POST /slots/0?action=save HTTP/1.1",
        "POST /slots/1?action=save HTTP/1.1",
      ],
      "one dump per slot, each under the name its own slot id gives it"
    );
    assert_eq!(requests[1].1, r#"{"filename":"slot-0.kv"}"#);
    assert_eq!(requests[2].1, r#"{"filename":"slot-1.kv"}"#);
  }

  #[tokio::test]
  async fn a_slot_the_engine_refuses_is_not_a_reason_to_leave_the_others_behind() {
    let (port, mut rx) = fake_engine(
      r#"[{"id":0},{"id":1}]"#,
      500,
      r#"{"error":{"message":"Unable to save slot: incomplete context checkpoints"}}"#,
    )
    .await;
    assert!(save_slots(port).await.is_none());
    let requests = captured(&mut rx);
    assert_eq!(requests.len(), 3, "both slots were still attempted");
  }

  #[tokio::test]
  async fn a_launch_that_is_told_to_keep_nothing_is_not_asked_at_all() {
    let (_port, mut rx) = fake_engine(r#"[{"id":0}]"#, 200, SAVE_OK).await;
    let model = crate::daemon::supervisor::test_support::in_state(
      crate::daemon::supervisor::ManagedState::Ready,
    );
    // `in_state` seeds no prompt-cache directory, which is the shape of every
    // non-chat launch, every `--fit` launch and every build without the
    // endpoints: the hook has to reach it and do nothing.
    save_before_stop(&model).await;
    restore_before_ready(&model).await;
    assert!(captured(&mut rx).is_empty());
  }

  #[tokio::test]
  async fn more_slots_than_the_cap_are_not_all_written() {
    let six = r#"[{"id":0},{"id":1},{"id":2},{"id":3},{"id":4},{"id":5}]"#;
    let (port, mut rx) = fake_engine(six, 200, SAVE_OK).await;
    let saved = save_slots(port).await.expect("a save report");
    assert_eq!(saved.slots, MAX_DUMPS);
    let requests = captured(&mut rx);
    assert_eq!(requests.len(), MAX_DUMPS + 1);
    assert_eq!(requests[MAX_DUMPS].0, "POST /slots/3?action=save HTTP/1.1");
  }

  #[tokio::test]
  async fn a_new_launch_is_handed_its_own_dumps_back_in_slot_order() {
    let root = crate::util::test_temp::unique_temp_dir("prompt-cache-restore");
    let dir = root.join(CACHE_SUBDIR).join("m-01234567");
    write_dump(&dir, 1, 8, Duration::from_secs(1));
    write_dump(&dir, 0, 8, Duration::from_secs(1));
    std::fs::write(dir.join("notes.txt"), b"not ours").expect("stray file");
    let policy = Policy {
      dir: dir.clone(),
      max_bytes: u64::MAX,
      max_age: None,
    };

    let (port, mut rx) = fake_engine(
      "[]",
      200,
      r#"{"id_slot":0,"filename":"slot-0.kv","n_restored":1200,"n_read":39321600}"#,
    )
    .await;
    let restored = restore_dumps(port, &policy)
      .await
      .expect("a restore report");
    assert_eq!(restored.slots, 2);
    assert_eq!(restored.tokens, 2400);

    let requests = captured(&mut rx);
    let lines: Vec<&str> = requests.iter().map(|(line, _)| line.as_str()).collect();
    assert_eq!(
      lines,
      [
        "POST /slots/0?action=restore HTTP/1.1",
        "POST /slots/1?action=restore HTTP/1.1",
      ],
      "slot order, and nothing about a file we did not write"
    );
    assert_eq!(requests[0].1, r#"{"filename":"slot-0.kv"}"#);
    assert!(dir.join("notes.txt").exists());
  }
}
