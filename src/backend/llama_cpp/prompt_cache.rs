//! Keep a llama.cpp model's prompt cache across an unload.
//!
//! `llama-server` can write one slot's KV state to disk (`POST
//! /slots/{id_slot}?action=save`) and read it back into a later launch of the
//! same model (`?action=restore`) — but only when it was started with
//! `--slot-save-path PATH`, and only into a bare file name inside that
//! directory. The daemon uses that pair on the llama.cpp paths that stop a
//! launch on purpose (the proxy idle sweep, make-room, `stop`), and hands what
//! it kept back to the next launch of that model before its first request is
//! forwarded. A continuation of the same conversation is then charged for its
//! new tokens only.
//!
//! Three facts keep this inside the llama.cpp module:
//!
//! - `--slot-save-path` is a llama.cpp flag. A build that does not advertise it
//!   gets no save directory at all ([`super::caps`]), so the launch argv is
//!   unchanged for it.
//! - Whether kept state fits a new launch is the engine's decision, not ours. It
//!   validates the file against its own KV layout and answers an error rather
//!   than half-restoring it; we read that error and move on.
//! - The files are engine state in the engine's format. Nothing else in the
//!   product reads them.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use crate::daemon::context::MethodContext;
use crate::daemon::registry::LaunchId;
use crate::launch::params::LaunchParams;

use super::{LlamaCppConfig, LLAMACPP_KNOB_PROMPT_CACHE_DIR, LLAMACPP_KNOB_SLOT_SAVE_PATH};

/// The flag the save directory is passed with. A user who puts this in a
/// preset's `extras` owns it — see [`plan_save_dir`].
pub(super) const SLOT_SAVE_ARG: &str = "--slot-save-path";

/// Factory `backend.llamacpp.prompt_cache_ttl_secs`: a day. Long enough that
/// stepping away from a conversation, restarting the daemon, and coming back the
/// next morning still hits; short enough that state nobody asked for again is not
/// still on disk a week later.
pub(super) const DEFAULT_TTL_SECS: u64 = 24 * 60 * 60;

/// Factory `backend.llamacpp.prompt_cache_max_bytes`. A 32 k-token window costs
/// roughly 1 GiB of KV state (the spike measured ~33 kB per token), so this holds
/// a handful of live conversations across a few models and nothing much beyond
/// that.
pub(super) const DEFAULT_MAX_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// Directory for kept slot state. `LaunchEnv::log_dir` is
/// [`crate::util::paths::log_dir`] = `<cache dir>/logs`, so its parent is the
/// cache directory the daemon already creates and every test isolates: one
/// sibling of `logs/`, shared by every model, holding nothing but this.
const DIR_NAME: &str = "prompt-cache";

/// Per-model record of what was kept, and the name it is renamed to when a
/// launch takes ownership of restoring it. Both are pruned like the state files.
const PENDING_SUFFIX: &str = "-prompt-cache.json";
const CONSUMED_SUFFIX: &str = "-prompt-cache.consumed.json";

/// Format version, bumped when the shape below changes. An entry written by an
/// older daemon is dropped rather than guessed at.
const VERSION: u8 = 1;

/// `/slots` answers a small array on loopback.
const LOOKUP_TIMEOUT: Duration = Duration::from_secs(5);

/// Saving and restoring are the only calls here that move gigabytes. The spike
/// (`docs/spikes/2026-10-21-prompt-cache-across-unload.md`) measured 0.9 s to
/// write and 0.6-1.1 s to read a 3 GiB state file, so the bound is a guard
/// against a wedged child rather than a budget on real work.
const SLOT_OP_TIMEOUT: Duration = Duration::from_secs(60);

/// A slot list or one small acknowledgement; nothing here is a model output.
const MAX_BODY_BYTES: usize = 256 * 1024;

// ============================================================================
// Where the files go
// ============================================================================

/// Whether a build has the save/restore endpoints at all. They are reachable
/// only behind [`SLOT_SAVE_ARG`], so a build's `--help` advertising the flag is
/// the whole test — and a build that does not have it must not be pointed at a
/// directory it would reject while parsing argv.
pub(super) fn supported_in(help: &str) -> bool {
  help.contains(SLOT_SAVE_ARG)
}

/// The directory a launch is pointed at: `<cache dir>/prompt-cache/`.
///
/// One directory per daemon rather than one per model, because llama.cpp joins
/// the file name onto this path and validates the name as a bare file
/// (`fs_validate_filename`, subdirectories refused). Which model a file belongs
/// to is in its name — see [`model_key`].
pub(super) fn cache_root(log_dir: &Path) -> PathBuf {
  log_dir.parent().unwrap_or(log_dir).join(DIR_NAME)
}

/// File name prefix for one model's kept state, shared by the record and the
/// slot files it names.
///
/// The same shape as the per-launch log names and for the same reason: the stem
/// so a person listing the directory recognises the model, and the header
/// fingerprint so two models sharing a stem cannot collide *and* a model swapped
/// out at the same path cannot be handed the previous model's KV state.
fn model_key(id: &crate::gguf::identity::ModelId) -> String {
  let stem = id
    .path
    .file_stem()
    .and_then(|s| s.to_str())
    .unwrap_or("model");
  // llama.cpp accepts `[A-Za-z0-9._-]` in a slot file name and refuses a leading
  // '-' or a `..` component; anything else in a model's file stem becomes '-'.
  let safe: String = stem
    .chars()
    .map(|c| {
      if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
        c
      } else {
        '-'
      }
    })
    .collect();
  let safe = safe.trim_start_matches('-').replace("..", "-");
  format!("{}-{}", safe, id.short_fingerprint())
}

fn pending_path(dir: &Path, key: &str) -> PathBuf {
  dir.join(format!("{key}{PENDING_SUFFIX}"))
}

fn consumed_path(dir: &Path, key: &str) -> PathBuf {
  dir.join(format!("{key}{CONSUMED_SUFFIX}"))
}

fn is_record_name(name: &str) -> bool {
  name.ends_with(PENDING_SUFFIX) || name.ends_with(CONSUMED_SUFFIX)
}

// ============================================================================
// The record of what was kept
// ============================================================================

/// One model's kept slot state, as written by [`save_before_stop`] and read by
/// [`restore_on_ready`].
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct Entry {
  version: u8,
  /// The model this state came from. Not consulted (the file name carries the
  /// model); recorded so the directory reads legibly.
  model: String,
  saved_at: u64,
  slots: Vec<SavedSlot>,
  /// Present once a launch has taken the entry.
  #[serde(default, skip_serializing_if = "Option::is_none")]
  restored_at: Option<u64>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct SavedSlot {
  /// The llama.cpp slot id the state came from, and the id it is restored into.
  id: u32,
  /// The slot's context window at save time. Kept for the log line that
  /// explains a refusal; the fit test compares the *current* window.
  n_ctx: u64,
  /// File name inside the save directory, as handed to the engine.
  file: String,
  tokens: u64,
  bytes: u64,
}

impl Entry {
  fn age_secs(&self, now: u64) -> u64 {
    now.saturating_sub(self.saved_at)
  }

  fn total_bytes(&self) -> u64 {
    self.slots.iter().map(|s| s.bytes).sum()
  }
}

fn read_entry(path: &Path) -> Option<Entry> {
  let body = match std::fs::read(path) {
    Ok(b) => b,
    Err(e) => {
      log::debug!("prompt cache: cannot read {}: {e}", path.display());
      return None;
    }
  };
  match serde_json::from_slice::<Entry>(&body) {
    Ok(entry) if entry.version == VERSION => Some(entry),
    Ok(entry) => {
      log::debug!(
        "prompt cache: {} is format {} (this daemon writes {VERSION}); dropped",
        path.display(),
        entry.version
      );
      None
    }
    Err(e) => {
      log::debug!(
        "prompt cache: {} is unreadable ({e}); dropped",
        path.display()
      );
      None
    }
  }
}

fn write_entry(dir: &Path, path: &Path, entry: &Entry) {
  let body = match serde_json::to_vec_pretty(entry) {
    Ok(b) => b,
    Err(e) => {
      log::warn!("prompt cache: could not serialise {}: {e}", path.display());
      return;
    }
  };
  let prefix = format!("{}.", DIR_NAME);
  if let Err(e) = crate::util::atomic_write::write_secure(dir, &prefix, path, &body, Some(0o600)) {
    log::warn!("prompt cache: could not write {}: {e}", path.display());
  }
}

fn discard(entry: &Entry, dir: &Path, key: &str) {
  for slot in &entry.slots {
    let _ = std::fs::remove_file(dir.join(&slot.file));
  }
  let _ = std::fs::remove_file(pending_path(dir, key));
  let _ = std::fs::remove_file(consumed_path(dir, key));
}

fn now_unix() -> u64 {
  SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .map(|d| d.as_secs())
    .unwrap_or_default()
}

// ============================================================================
// Launch wiring: point the engine at the directory
// ============================================================================

/// Decide whether this launch should keep a prompt cache, and record the
/// directory it would use.
///
/// Path math only: creating the directory waits for [`enable_for_launch`], which
/// runs once the binary is known and has answered whether it has the endpoints.
/// A llama.cpp build with no `--slot-save-path` must end up with no flag and no
/// directory.
pub(super) fn plan_save_dir(
  cfg: &LlamaCppConfig,
  log_dir: Option<&Path>,
  params: &mut LaunchParams,
) {
  // Always dropped here and re-seeded by `enable_for_launch` below, so a value
  // inherited from `last_params` cannot outlive the config flip that removed it.
  params.launch_config.remove(LLAMACPP_KNOB_SLOT_SAVE_PATH);
  params.launch_config.remove(LLAMACPP_KNOB_PROMPT_CACHE_DIR);
  // `scripts/bench/` compares the composed argv with raw `llama-server`'s, so no
  // default-only flag may be added under that switch — same gate as `jinja`.
  if !cfg.keep_prompt_cache
    || cfg.prompt_cache_max_bytes == 0
    || crate::launch::params::bench_disable_defaults_from_env()
  {
    return;
  }
  // A `--slot-save-path` in `extras` is the caller's own choice, and
  // llama-server honours the last occurrence. The save and restore calls need a
  // directory they can find, which `extras` does not tell us, so the cache is
  // left off rather than pointed at a guess.
  if crate::launch::params::extras_have_flag(&params.extras, SLOT_SAVE_ARG) {
    log::debug!("prompt cache: {SLOT_SAVE_ARG} comes from the launch's own extras; nothing kept");
    return;
  }
  if let Some(log_dir) = log_dir {
    params.launch_config.insert(
      LLAMACPP_KNOB_PROMPT_CACHE_DIR.to_string(),
      cache_root(log_dir).display().to_string(),
    );
  }
}

/// Turn the planned directory on for this launch.
///
/// `supported` is what the resolved build answered to `--help`. llama.cpp
/// validates the path *while parsing argv*, so the directory is created here —
/// the first point that has both the plan and the binary's answer. `compose`
/// emits the flag from the value this writes.
pub(super) fn enable_for_launch(params: &mut LaunchParams, supported: bool) {
  if !supported {
    // No flag on this build, so no directory: the launch argv is unchanged.
    log::debug!("prompt cache: {SLOT_SAVE_ARG} not offered by this build; nothing kept");
    return;
  }
  let Some(dir) = params
    .launch_config
    .get(LLAMACPP_KNOB_PROMPT_CACHE_DIR)
    .map(PathBuf::from)
  else {
    return;
  };
  if let Err(e) = std::fs::create_dir_all(&dir) {
    log::warn!("prompt cache: could not create {}: {e}", dir.display());
    return;
  }
  params.launch_config.insert(
    LLAMACPP_KNOB_SLOT_SAVE_PATH.to_string(),
    dir.display().to_string(),
  );
}

/// The launch's save directory, and the model it belongs to. `None` whenever
/// this launch was never given a save path — the shape every unsupported build,
/// disabled config, and `extras`-owned launch ends up with.
async fn launch_cache(
  ctx: &MethodContext,
  launch_id: &LaunchId,
) -> Option<(crate::daemon::supervisor::ManagedModel, PathBuf)> {
  if !ctx.backend.llamacpp.keep_prompt_cache {
    return None;
  }
  let model = ctx.supervisors.get(launch_id).await?;
  let dir = model
    .params()
    .launch_config
    .get(LLAMACPP_KNOB_SLOT_SAVE_PATH)
    .map(PathBuf::from)?;
  Some((model, dir))
}

// ============================================================================
// Save
// ============================================================================

/// Write this launch's idle slots to disk before it is stopped.
///
/// Called from the llama.cpp `stop` override, so it runs on every path that
/// stops a llama.cpp launch on purpose — the idle sweep, make-room, and `stop`.
/// Best-effort throughout: an engine that will not answer costs a log line, not
/// the stop.
pub(super) async fn save_before_stop(ctx: &MethodContext, launch_id: &LaunchId) {
  let Some((model, dir)) = launch_cache(ctx, launch_id).await else {
    return;
  };
  let key = model_key(model.id());
  let slots = match read_slots(model.port()).await {
    Ok(s) => s,
    Err(e) => {
      log::warn!(
        "prompt cache: /slots on {} unreadable ({e}); nothing kept",
        model.port()
      );
      return;
    }
  };
  // A slot mid-generation cannot be serialised (llama.cpp defers the task until
  // it is idle, and a deferred save would have us waiting for a stop). A slot
  // with no prompt has nothing worth the bytes.
  let warm: Vec<&SlotInfo> = slots
    .iter()
    .filter(|s| !s.is_processing && s.n_prompt_tokens > 0)
    .collect();
  if warm.is_empty() {
    log::debug!(
      "prompt cache: {} has no idle slot with a prompt",
      launch_id.as_str()
    );
    return;
  }

  let cfg = &ctx.backend.llamacpp;
  prune(cfg, &dir);
  // The state a model keeps is one generation deep: a new save replaces what an
  // earlier launch left, which would otherwise accumulate every conversation the
  // daemon has ever had.
  for slot in read_entry(&pending_path(&dir, &key))
    .into_iter()
    .flat_map(|e| e.slots)
    .chain(
      read_entry(&consumed_path(&dir, &key))
        .into_iter()
        .flat_map(|e| e.slots),
    )
  {
    let _ = std::fs::remove_file(dir.join(&slot.file));
  }
  let _ = std::fs::remove_file(pending_path(&dir, &key));
  let _ = std::fs::remove_file(consumed_path(&dir, &key));

  let stamp = now_unix();
  let mut entry = Entry {
    version: VERSION,
    model: model.params().model_path.display().to_string(),
    saved_at: stamp,
    slots: Vec::new(),
    restored_at: None,
  };
  for slot in warm {
    let file = format!("{key}-slot-{}-{stamp}.bin", slot.id);
    let body = serde_json::json!({ "filename": file });
    let path = format!("/slots/{}?action=save", slot.id);
    match request(model.port(), &path, Some(&body), SLOT_OP_TIMEOUT).await {
      Ok(ack) => {
        let written = dir.join(&file);
        let tokens = ack
          .get("n_saved")
          .and_then(serde_json::Value::as_u64)
          .unwrap_or(slot.n_prompt_tokens);
        let bytes = std::fs::metadata(&written)
          .map(|m| m.len())
          .or_else(|_| {
            ack
              .get("n_written")
              .and_then(serde_json::Value::as_u64)
              .ok_or_else(std::io::Error::last_os_error)
          })
          .unwrap_or(0);
        log::info!(
          "prompt cache: kept {} prompt tokens ({}) for {} in slot {}",
          tokens,
          crate::init::detection::fmt_bytes(bytes),
          model.id().path.display(),
          slot.id
        );
        entry.slots.push(SavedSlot {
          id: slot.id,
          n_ctx: slot.n_ctx,
          file,
          tokens,
          bytes,
        });
      }
      Err(e) => log::warn!(
        "prompt cache: slot {} of {} not kept: {e}",
        slot.id,
        model.port()
      ),
    }
  }
  if entry.slots.is_empty() {
    return;
  }
  write_entry(&dir, &pending_path(&dir, &key), &entry);
  // After the write, so the newest state is what survives the cap.
  prune(cfg, &dir);
}

// ============================================================================
// Restore
// ============================================================================

/// Hand a launch back the state the daemon kept when it unloaded this model.
///
/// Runs before the launch's first request is forwarded, so that request is the
/// one that benefits. A slot the engine reports as holding a prompt is left
/// alone — a launch that has already been serving has nothing to take — and a
/// slot whose window is too small for what was kept is skipped rather than
/// truncated.
pub(super) async fn restore_on_ready(ctx: &MethodContext, launch_id: &LaunchId) {
  let Some((model, dir)) = launch_cache(ctx, launch_id).await else {
    return;
  };
  let key = model_key(model.id());
  let pending = pending_path(&dir, &key);
  // The common case is nothing kept, and that must cost nothing: one stat.
  if !pending.exists() {
    return;
  }
  // Take ownership of the entry before asking for it. Two requests racing into
  // one fresh launch both see it; only one rename wins, so only one pays for a
  // restore the other would repeat.
  let claimed = consumed_path(&dir, &key);
  if std::fs::rename(&pending, &claimed).is_err() {
    return;
  }
  let Some(entry) = read_entry(&claimed) else {
    let _ = std::fs::remove_file(&claimed);
    return;
  };
  let cfg = &ctx.backend.llamacpp;
  let now = now_unix();
  if entry.age_secs(now) > cfg.prompt_cache_ttl_secs {
    log::info!(
      "prompt cache: state kept for {} has aged out ({} s) and is dropped",
      model.params().model_path.display(),
      entry.age_secs(now)
    );
    discard(&entry, &dir, &key);
    return;
  }
  let slots = match read_slots(model.port()).await {
    Ok(s) => s,
    Err(e) => {
      // A transport failure says nothing about the state; leave it for the next
      // attempt rather than spending it on a child that never answered.
      log::warn!(
        "prompt cache: /slots on {} unreadable ({e}); left for later",
        model.port()
      );
      let _ = std::fs::rename(&claimed, &pending);
      return;
    }
  };

  let mut attempted = 0usize;
  let mut taken = 0usize;
  for saved in &entry.slots {
    // Only an empty slot can take a restore: llama.cpp refuses while the slot is
    // generating, and a slot holding a prompt has its own state worth keeping.
    let Some(target) = slots
      .iter()
      .find(|s| s.id == saved.id && !s.is_processing && s.n_prompt_tokens == 0)
    else {
      continue;
    };
    // The engine reads the state into its current KV layout. `--fit` may have
    // chosen a different window this time, and a window under what was kept
    // cannot hold it — say so instead of letting the engine fail cryptically.
    if saved.tokens > target.n_ctx {
      log::info!(
        "prompt cache: kept {} prompt tokens for {}, but this launch's slot {} holds {} — nothing restored",
        saved.tokens,
        model.params().model_path.display(),
        target.id,
        target.n_ctx
      );
      continue;
    }
    attempted += 1;
    let body = serde_json::json!({ "filename": saved.file });
    let path = format!("/slots/{}?action=restore", saved.id);
    match request(model.port(), &path, Some(&body), SLOT_OP_TIMEOUT).await {
      Ok(ack) => {
        let restored = ack
          .get("n_restored")
          .and_then(serde_json::Value::as_u64)
          .unwrap_or(saved.tokens);
        log::info!(
          "prompt cache: restored {} prompt tokens for {} into slot {}",
          restored,
          model.params().model_path.display(),
          saved.id
        );
        taken += 1;
      }
      Err(e) => log::warn!(
        "prompt cache: slot {} of {} could not be restored: {e}",
        saved.id,
        model.params().model_path.display()
      ),
    }
  }

  if attempted == 0 {
    // Nothing here fit *this* launch. The state belongs to the model, not to
    // this shape of launch, so it stays for one that matches.
    log::debug!(
      "prompt cache: nothing kept for {} fits this launch; left in place",
      model.params().model_path.display()
    );
    let _ = std::fs::rename(&claimed, &pending);
    return;
  }

  // Consumed: the state files are spent whether the engine took them or refused
  // them, and the record is kept only to say what happened.
  for saved in &entry.slots {
    let _ = std::fs::remove_file(dir.join(&saved.file));
  }
  let done = Entry {
    restored_at: Some(now),
    ..entry
  };
  write_entry(&dir, &claimed, &done);
  log::info!(
    "prompt cache: {taken}/{attempted} slot(s) restored for {} ({})",
    model.params().model_path.display(),
    crate::init::detection::fmt_bytes(done.total_bytes())
  );
}

// ============================================================================
// Engine calls
// ============================================================================

/// One slot as `GET /slots` reports it.
#[derive(Debug)]
struct SlotInfo {
  id: u32,
  n_ctx: u64,
  is_processing: bool,
  n_prompt_tokens: u64,
}

async fn read_slots(port: u16) -> Result<Vec<SlotInfo>, String> {
  let value = request(port, "/slots", None, LOOKUP_TIMEOUT).await?;
  Ok(parse_slots(&value))
}

/// The slot list out of a `/slots` reply: an array, or the same array under a
/// `slots` key. Fields a build does not report read as busy / empty / zero
/// window, which only ever costs a save or a restore, never a wrong one.
fn parse_slots(value: &serde_json::Value) -> Vec<SlotInfo> {
  let items: Option<&Vec<serde_json::Value>> = match value {
    serde_json::Value::Array(items) => Some(items),
    serde_json::Value::Object(fields) => fields.get("slots").and_then(|s| s.as_array()),
    _ => None,
  };
  items
    .into_iter()
    .flatten()
    .map(|slot| SlotInfo {
      id: slot
        .get("id")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0) as u32,
      n_ctx: slot
        .get("n_ctx")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0),
      is_processing: slot
        .get("is_processing")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(true),
      n_prompt_tokens: slot
        .get("n_prompt_tokens")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or_default(),
    })
    .collect()
}

/// One call to the child's HTTP server.
///
/// A hand-rolled request over raw TCP with `Connection: close`, like
/// [`super::actuals`]: these are three calls a daemon makes about its own
/// child, and pulling in `reqwest`/`hyper` for them would be the wrong trade.
async fn request(
  port: u16,
  path: &str,
  body: Option<&serde_json::Value>,
  timeout: Duration,
) -> Result<serde_json::Value, String> {
  let payload = body.map(|b| b.to_string()).unwrap_or_default();
  let head = format!(
    "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\nAccept: application/json\r\nContent-Type: application/json\r\nContent-Length: {len}\r\n\r\n",
    method = if body.is_some() { "POST" } else { "GET" },
    len = payload.len(),
  );
  // `Connection: close` makes the child hang up after the body, so reading to
  // EOF ends the exchange — the same stance as the `/props` read.
  let fut = async {
    let mut sock = TcpStream::connect(("127.0.0.1", port)).await?;
    sock.write_all(head.as_bytes()).await?;
    if !payload.is_empty() {
      sock.write_all(payload.as_bytes()).await?;
    }
    let mut buf = Vec::with_capacity(4096);
    let mut chunk = [0u8; 4096];
    loop {
      let n = sock.read(&mut chunk).await?;
      if n == 0 {
        break;
      }
      buf.extend_from_slice(&chunk[..n]);
      if buf.len() > MAX_BODY_BYTES {
        break;
      }
    }
    Ok::<_, std::io::Error>(buf)
  };
  let raw = tokio::time::timeout(timeout, fut)
    .await
    .map_err(|_| "no reply in time".to_string())?
    .map_err(|e| format!("connection failed: {e}"))?;
  let (status, resp) = split_response(&raw)?;
  if !(200..300).contains(&status) {
    let detail = String::from_utf8_lossy(resp);
    let detail = detail.trim();
    let detail = if detail.len() > 240 {
      &detail[..240]
    } else {
      detail
    };
    return Err(format!("HTTP {status}: {detail}"));
  }
  serde_json::from_slice(resp).map_err(|e| format!("unreadable reply: {e}"))
}

/// Status line and body from a raw HTTP response.
fn split_response(raw: &[u8]) -> Result<(u16, &[u8]), String> {
  let head_end = raw
    .windows(4)
    .position(|w| w == b"\r\n\r\n")
    .ok_or_else(|| "no HTTP headers in reply".to_string())?;
  let head = String::from_utf8_lossy(&raw[..head_end]);
  let status = head
    .lines()
    .next()
    .and_then(|line| line.split_whitespace().nth(1))
    .and_then(|code| code.parse::<u16>().ok())
    .ok_or_else(|| "no status line in reply".to_string())?;
  Ok((status, &raw[head_end + 4..]))
}

// ============================================================================
// Keeping the disk bounded
// ============================================================================

/// Drop kept state the daemon is no longer willing to hold.
///
/// Runs at save time, which is the only moment the daemon is adding to this
/// directory. A record older than `prompt_cache_ttl_secs` goes first; then, if
/// the total is over `prompt_cache_max_bytes`, the oldest records go until it
/// fits. A state file no record names is a save interrupted by a crash, and it
/// goes once it is older than the TTL too.
fn prune(cfg: &LlamaCppConfig, dir: &Path) {
  let Ok(reads) = std::fs::read_dir(dir) else {
    return;
  };
  let now = now_unix();
  let mut records: Vec<(PathBuf, Entry)> = Vec::new();
  let mut named: Vec<String> = Vec::new();
  let mut stale_files: Vec<PathBuf> = Vec::new();

  for read in reads.flatten() {
    let path = read.path();
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
      continue;
    };
    if !is_record_name(name) {
      continue;
    }
    match read_entry(&path) {
      Some(entry) => records.push((path, entry)),
      None => {
        let _ = std::fs::remove_file(&path);
      }
    }
  }

  records.sort_by_key(|(_, entry)| entry.saved_at);
  let mut total: u64 = 0;
  for (path, entry) in &records {
    if entry.age_secs(now) > cfg.prompt_cache_ttl_secs {
      let key = record_key(path);
      discard(entry, dir, &key);
      continue;
    }
    total += entry.total_bytes();
    for slot in &entry.slots {
      named.push(slot.file.clone());
    }
  }
  // Oldest first: the point of keeping state is the conversation that is still
  // live, and that is the newest one on disk.
  for (path, entry) in &records {
    if total <= cfg.prompt_cache_max_bytes {
      break;
    }
    let key = record_key(path);
    if pending_path(dir, &key) == *path || consumed_path(dir, &key) == *path {
      log::info!(
        "prompt cache: {} over the {} cap — dropped {}",
        crate::init::detection::fmt_bytes(total),
        crate::init::detection::fmt_bytes(cfg.prompt_cache_max_bytes),
        entry.model
      );
      total = total.saturating_sub(entry.total_bytes());
      discard(entry, dir, &key);
    }
  }

  for read in std::fs::read_dir(dir).into_iter().flatten().flatten() {
    let path = read.path();
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
      continue;
    };
    if is_record_name(name) || named.iter().any(|n| n == name) {
      continue;
    }
    let older_than_ttl = read
      .metadata()
      .and_then(|m| m.modified())
      .map(|t| {
        t.duration_since(UNIX_EPOCH)
          .map(|d| now.saturating_sub(d.as_secs()))
          .unwrap_or(0)
      })
      .unwrap_or(0)
      > cfg.prompt_cache_ttl_secs;
    if older_than_ttl {
      stale_files.push(path);
    }
  }
  for path in stale_files {
    let _ = std::fs::remove_file(&path);
  }
}

/// The model key a record file name carries.
fn record_key(path: &Path) -> String {
  let name = path
    .file_name()
    .and_then(|n| n.to_str())
    .unwrap_or_default()
    .to_string();
  name
    .trim_end_matches(PENDING_SUFFIX)
    .trim_end_matches(CONSUMED_SUFFIX)
    .to_string()
}

#[cfg(test)]
mod tests {
  use super::*;

  /// A directory that cleans itself up: unit tests in `src/` never share one.
  fn temp_dir(label: &str) -> tempfile::TempDir {
    tempfile::Builder::new()
      .prefix(&format!("llamastash-prompt-cache-{label}-"))
      .tempdir()
      .unwrap()
  }

  fn entry(key: &str, tokens: u64, bytes: u64, saved_at: u64) -> Entry {
    Entry {
      version: VERSION,
      model: "/models/x.gguf".to_string(),
      saved_at,
      slots: vec![SavedSlot {
        id: 0,
        n_ctx: 32768,
        file: format!("{key}-slot-0.bin"),
        tokens,
        bytes,
      }],
      restored_at: None,
    }
  }

  fn model_id(path: &str, fingerprint: [u8; 4]) -> crate::gguf::identity::ModelId {
    let mut header = [0u8; 32];
    header[..4].copy_from_slice(&fingerprint);
    crate::gguf::identity::ModelId {
      path: PathBuf::from(path),
      header_blake3: header,
    }
  }

  #[test]
  fn a_build_without_the_flag_gets_no_directory() {
    assert!(!supported_in(
      "       --mlock    force system to keep model in RAM"
    ));
    assert!(supported_in(
      "     --slot-save-path PATH    directory for slot state serialization"
    ));
  }

  fn params() -> LaunchParams {
    LaunchParams::new(
      PathBuf::from("/models/x.gguf"),
      crate::launch::mode::LaunchMode::Chat,
    )
  }

  fn cfg() -> LlamaCppConfig {
    LlamaCppConfig::default()
  }

  /// The plan is pure path math, so it is testable without a daemon: it says
  /// which directory a launch would use, and `enable_for_launch` is what decides
  /// whether the build gets it.
  #[test]
  fn a_planned_directory_is_recorded_on_the_launch() {
    let mut p = params();
    plan_save_dir(&cfg(), Some(Path::new("/cache/llamastash/logs")), &mut p);
    let expected = Path::new("/cache/llamastash/logs")
      .parent()
      .expect("logs dir has a parent")
      .join("prompt-cache")
      .to_string_lossy()
      .into_owned();
    assert_eq!(
      p.launch_config
        .get(LLAMACPP_KNOB_PROMPT_CACHE_DIR)
        .map(String::as_str),
      Some(expected.as_str())
    );
    // Planning alone never hands the engine a flag: only a confirmed build does.
    assert!(!p.launch_config.contains_key(LLAMACPP_KNOB_SLOT_SAVE_PATH));
  }

  #[test]
  fn turning_it_off_leaves_the_launch_exactly_as_it_was() {
    let mut p = params();
    let mut off = cfg();
    off.keep_prompt_cache = false;
    plan_save_dir(&off, Some(Path::new("/cache/llamastash/logs")), &mut p);
    assert!(p.launch_config.is_empty());

    // `0` bytes is "keep nothing", and it is worth saying so without spawning a
    // save the budget would delete in the same breath.
    let mut capped = params();
    let mut no_room = cfg();
    no_room.prompt_cache_max_bytes = 0;
    plan_save_dir(
      &no_room,
      Some(Path::new("/cache/llamastash/logs")),
      &mut capped,
    );
    assert!(capped.launch_config.is_empty());
  }

  /// A caller who passes `--slot-save-path` themselves knows what they want; the
  /// daemon cannot read their directory back out of `extras`, so it stays out.
  #[test]
  fn a_directory_the_caller_owns_is_left_to_them() {
    let mut p = params();
    p.extras.push("--slot-save-path".into());
    p.extras.push("/home/u/my-slot-dir".into());
    plan_save_dir(&cfg(), Some(Path::new("/cache/llamastash/logs")), &mut p);
    assert!(!p.launch_config.contains_key(LLAMACPP_KNOB_PROMPT_CACHE_DIR));
  }

  /// An inherited `last_params` copy carries the previous launch's keys. The plan
  /// drops both before deciding, so `keep_prompt_cache: false` takes effect on
  /// the next launch rather than living on through inheritance.
  #[test]
  fn an_inherited_directory_is_replanned_not_kept() {
    let mut p = params();
    p.launch_config.insert(
      LLAMACPP_KNOB_SLOT_SAVE_PATH.to_string(),
      "/cache/llamastash/prompt-cache".to_string(),
    );
    let mut off = cfg();
    off.keep_prompt_cache = false;
    plan_save_dir(&off, Some(Path::new("/cache/llamastash/logs")), &mut p);
    assert!(!p.launch_config.contains_key(LLAMACPP_KNOB_SLOT_SAVE_PATH));
  }

  #[test]
  fn an_unsupported_build_is_never_given_the_flag_or_the_directory() {
    let tmp = temp_dir("enable");
    let log_dir = tmp.path().join("logs");
    let mut p = params();
    plan_save_dir(&cfg(), Some(&log_dir), &mut p);
    let planned = p
      .launch_config
      .get(LLAMACPP_KNOB_PROMPT_CACHE_DIR)
      .cloned()
      .unwrap();
    enable_for_launch(&mut p, false);
    assert!(!p.launch_config.contains_key(LLAMACPP_KNOB_SLOT_SAVE_PATH));
    assert!(
      std::fs::metadata(&planned).is_err(),
      "no directory for a build that cannot use one"
    );

    enable_for_launch(&mut p, true);
    assert_eq!(
      p.launch_config
        .get(LLAMACPP_KNOB_SLOT_SAVE_PATH)
        .map(String::as_str),
      Some(planned.as_str())
    );
    assert!(
      std::fs::metadata(&planned).is_ok_and(|m| m.is_dir()),
      "llama.cpp checks the path while parsing argv, so it has to exist first"
    );
  }

  #[test]
  fn cache_root_sits_beside_the_log_directory() {
    // `<cache dir>/logs` is the daemon's log dir; kept state is its sibling, so
    // a state-dir / cache-dir override moves both together.
    assert_eq!(
      cache_root(Path::new("/home/u/.cache/llamastash/logs")),
      PathBuf::from("/home/u/.cache/llamastash/prompt-cache")
    );
    // A log dir with no meaningful parent (a test handed a bare temp dir) still
    // keeps its files inside it.
    assert_eq!(cache_root(Path::new("/")), PathBuf::from("/prompt-cache"));
  }

  #[test]
  fn model_key_is_a_name_llama_cpp_will_accept() {
    // `fs_validate_filename(name, allow_subdirs = false)`: no leading '-', no
    // `..`, and only `[A-Za-z0-9._-]`.
    let id = model_id(
      "/models/-Llama 3.2-1B (Q4_K_M).gguf",
      [0xab, 0xcd, 0xef, 0x01],
    );
    let key = model_key(&id);
    assert!(!key.starts_with('-'));
    assert!(!key.contains(".."));
    assert!(!key.contains('/'));
    assert!(key
      .chars()
      .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-')));
    assert!(key.starts_with("Llama-3.2-1B--Q4_K_M-"), "{key}");
    assert!(key.ends_with("abcdef01"), "{key}");
    assert_eq!(model_key(&id), key, "the same model gets the same key");

    // A file name with an inner `..` is a name the engine would refuse, so it
    // is rewritten rather than passed through.
    let dotted = model_key(&model_id("/models/a..b.gguf", [1, 2, 3, 4]));
    assert!(!dotted.contains(".."), "{dotted}");
  }

  #[test]
  fn a_different_model_at_the_same_path_is_not_the_same_key() {
    // A GGUF swapped in at a path people already used must not be handed the
    // previous model's KV state.
    let before = model_key(&model_id("/models/x.gguf", [1, 2, 3, 4]));
    let after = model_key(&model_id("/models/x.gguf", [5, 6, 7, 8]));
    assert_ne!(before, after);
    assert_eq!(
      before.rsplit_once('-').unwrap().0,
      after.rsplit_once('-').unwrap().0,
      "the readable part is still the same model name"
    );
  }

  #[test]
  fn record_paths_round_trip_the_key() {
    let dir = Path::new("/d");
    let pending = pending_path(dir, "a-1234");
    let consumed = consumed_path(dir, "a-1234");
    assert!(is_record_name(
      pending.file_name().unwrap().to_str().unwrap()
    ));
    assert!(is_record_name(
      consumed.file_name().unwrap().to_str().unwrap()
    ));
    assert!(!is_record_name("a-1234-slot-0-1700000000.bin"));
    assert_eq!(record_key(&pending), "a-1234");
    assert_eq!(record_key(&consumed), "a-1234");
  }

  #[test]
  fn an_unreadable_or_foreign_record_is_dropped_not_guessed() {
    let tmp = temp_dir("record");
    let dir = tmp.path();
    let path = dir.join("x-prompt-cache.json");
    std::fs::write(&path, b"{ not json").unwrap();
    assert!(read_entry(&path).is_none());
    let foreign = dir.join("y-prompt-cache.json");
    std::fs::write(
      &foreign,
      format!(
        "{{\"version\":{},\"model\":\"m\",\"saved_at\":1,\"slots\":[]}}",
        VERSION + 1
      ),
    )
    .unwrap();
    assert!(read_entry(&foreign).is_none());
  }

  #[test]
  fn a_written_record_reads_back() {
    let tmp = temp_dir("roundtrip");
    let dir = tmp.path();
    let path = pending_path(dir, "x");
    write_entry(dir, &path, &entry("x", 1000, 32_000, 1234));
    let read = read_entry(&path).expect("reads back");
    assert_eq!(read.saved_at, 1234);
    assert_eq!(read.slots[0].tokens, 1000);
    assert_eq!(read.restored_at, None);
  }

  #[test]
  fn pruning_drops_aged_and_over_cap_records() {
    let tmp = temp_dir("prune");
    let dir = tmp.path();
    let now = now_unix();
    let cfg = LlamaCppConfig {
      prompt_cache_ttl_secs: 10,
      prompt_cache_max_bytes: 1000,
      ..LlamaCppConfig::default()
    };

    // One fresh, small record.
    let fresh = entry("fresh", 10, 400, now);
    std::fs::write(dir.join("fresh-slot-0.bin"), vec![7u8; 400]).unwrap();
    write_entry(dir, &pending_path(dir, "fresh"), &fresh);

    // One past the age limit, with its state file.
    let old = entry("old", 10, 400, now - 60);
    std::fs::write(dir.join("old-slot-0.bin"), vec![7u8; 400]).unwrap();
    write_entry(dir, &pending_path(dir, "old"), &old);

    prune(&cfg, dir);
    assert!(pending_path(dir, "fresh").exists());
    assert!(dir.join("fresh-slot-0.bin").exists());
    assert!(!pending_path(dir, "old").exists());
    assert!(
      !dir.join("old-slot-0.bin").exists(),
      "aged state goes with its record"
    );

    // A newer record pushes the total over the cap; the older one goes.
    let newer = entry("newer", 10, 700, now + 1);
    std::fs::write(dir.join("newer-slot-0.bin"), vec![7u8; 700]).unwrap();
    write_entry(dir, &pending_path(dir, "newer"), &newer);
    prune(&cfg, dir);
    assert!(
      dir.join("newer-slot-0.bin").exists(),
      "the cap keeps the newest state"
    );
    assert!(!pending_path(dir, "fresh").exists());
    assert!(!dir.join("fresh-slot-0.bin").exists());
  }

  #[test]
  fn pruning_clears_state_no_record_names_once_it_is_stale() {
    let tmp = temp_dir("orphan");
    let dir = tmp.path();
    let orphan = dir.join("gone-slot-0-1.bin");
    std::fs::write(&orphan, vec![7u8; 8]).unwrap();
    // Backdated past the TTL: what a save interrupted by a crash leaves behind.
    let aged = SystemTime::now() - Duration::from_secs(120);
    // A writable handle: on Windows setting the mtime on a read-only handle
    // is refused with Access is denied.
    std::fs::OpenOptions::new()
      .write(true)
      .open(&orphan)
      .unwrap()
      .set_modified(aged)
      .unwrap();
    let recent = dir.join("live-slot-0-1.bin");
    std::fs::write(&recent, vec![7u8; 8]).unwrap();
    prune(
      &LlamaCppConfig {
        prompt_cache_ttl_secs: 10,
        ..LlamaCppConfig::default()
      },
      dir,
    );
    assert!(!orphan.exists());
    assert!(recent.exists(), "a recent orphan is left alone");
  }

  #[test]
  fn slots_are_read_from_either_reply_shape() {
    let flat = serde_json::json!([
      { "id": 0, "n_ctx": 32768, "is_processing": false, "n_prompt_tokens": 512 },
      { "id": 1, "n_ctx": 32768, "is_processing": true, "n_prompt_tokens": 0 }
    ]);
    let slots = parse_slots(&flat);
    assert_eq!(slots.len(), 2);
    assert_eq!(
      (slots[0].id, slots[0].n_ctx, slots[0].n_prompt_tokens),
      (0, 32768, 512)
    );
    assert!(!slots[0].is_processing);
    assert!(slots[1].is_processing);

    let wrapped = serde_json::json!({ "slots": [{ "id": 2, "n_ctx": 1024 }] });
    let wrapped_slots = parse_slots(&wrapped);
    assert_eq!(wrapped_slots.len(), 1);
    // A slot that does not say it is idle is treated as busy, and one that does
    // not say its window is treated as too small to restore into: both cost a
    // save or a restore, never a wrong one.
    assert!(wrapped_slots[0].is_processing);
    assert_eq!(wrapped_slots[0].n_ctx, 1024);

    assert!(parse_slots(&serde_json::json!("nonsense")).is_empty());
  }

  #[test]
  fn a_response_splits_at_the_header_boundary() {
    let raw = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{\"n_saved\":3}";
    let (status, body) = split_response(raw).unwrap();
    assert_eq!(status, 200);
    assert_eq!(body, b"{\"n_saved\":3}");
    assert!(split_response(b"no headers").is_err());
    assert!(split_response(b"HTTP/1.1 oops\r\n\r\n").is_err());
  }

  #[test]
  fn discard_removes_a_record_and_the_state_it_names() {
    let tmp = temp_dir("discard");
    let dir = tmp.path();
    let kept = entry("x", 1, 1, 1);
    std::fs::write(dir.join(&kept.slots[0].file), b"x").unwrap();
    std::fs::write(pending_path(dir, "x"), b"{}").unwrap();
    discard(&kept, dir, "x");
    assert!(!dir.join(&kept.slots[0].file).exists());
    assert!(!pending_path(dir, "x").exists());
  }
}
