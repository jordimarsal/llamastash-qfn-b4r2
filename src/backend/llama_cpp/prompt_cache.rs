//! Keeping a prompt cache across an unload.
//!
//! A launch stopped mid-conversation throws away everything the engine had
//! computed for it: llama.cpp keeps its KV state in the child process, so the
//! next request to the same model reprocesses the prompt from token zero.
//! llama.cpp can write a slot's KV state to a file (`POST
//! /slots/{id}?action=save`) and read it back into a slot of a later launch
//! (`?action=restore`) — that pair is the only way to carry the state across a
//! process boundary.
//!
//! The engine does the matching, not us. A restored slot keeps the token
//! sequence it was saved with, and llama.cpp routes an incoming request to the
//! slot whose prompt shares the longest common prefix with it
//! (`--slot-prompt-similarity`, on by default) and processes only the new tail.
//! What this module has to get right is *which file belongs to which model and
//! which slot*: a file is named after both, and goes back into that same slot.
//!
//! Reached only through [`crate::backend::Backend`]'s `before_stop` /
//! `after_ready` hooks, so neither the eviction path nor the launch path carries
//! a llama-server endpoint.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use serde_json::Value;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use crate::daemon::context::MethodContext;
use crate::daemon::supervisor::ManagedModel;
use crate::gguf::identity::ModelId;

use super::{LLAMACPP_KNOB_SLOT_SAVE, LLAMACPP_KNOB_SLOT_SAVE_DIR};

/// One save or restore moves a whole slot's KV cache to or from disk. The bound
/// is a floor for a healthy engine (a GiB-scale copy takes seconds) and a
/// ceiling for a wedged one: losing the cache is better than holding a model's
/// unload open.
const SLOT_CALL_TIMEOUT: Duration = Duration::from_secs(30);

/// The budget for one save or one restore round on one launch. A save round runs
/// inside a stop, and `make_room` stops candidates inside the request that
/// triggered it — so a model whose slots will not answer must not stall the
/// launch it was standing in the way of. A restore runs before a launch is called
/// Ready, so this is also how much longer a relaunch may take to become routable:
/// a cache that will not load is worth less than a model that answers.
const ROUND_TIMEOUT: Duration = Duration::from_secs(20);

/// `GET /slots` is a few hundred bytes and answers immediately.
const SLOTS_LIST_TIMEOUT: Duration = Duration::from_secs(3);

/// A launch stopped with several conversations resident is saved slot by slot.
/// Past this many, the marginal slot holds a small conversation nobody is about
/// to resume and the stop is already slow.
const MAX_SLOTS_SAVED: usize = 8;

/// `/slots` on a many-slot launch is large, but never megabyte-scale.
const MAX_RESPONSE_BYTES: usize = 1 << 20;

/// Where a daemon's checkpoints live: beside its `state.json` and `runtime.json`.
/// Not a model file — a dump of what a live model had computed, rebuilt by the
/// next conversation if lost — and not global either: two daemons on one host
/// must not restore, or prune away, each other's work, and only a daemon's own
/// rooted state dir keeps them apart.
pub(crate) fn slot_dir_for_state(state_dir: &Path) -> PathBuf {
  state_dir.join("prompt-cache")
}

/// Make a checkpoint directory usable, and say whether it is. llama.cpp validates
/// `--slot-save-path` while it is parsing its arguments and refuses to start on a
/// path that is not a directory, so the directory has to exist **before** the
/// flag is emitted — a launch that dies on a missing directory on a fresh state
/// dir loses the model, not just the cache.
///
/// `None` means the path cannot be one: the launch proceeds without the feature.
pub(crate) fn prepare_slot_dir(dir: &Path) -> Option<PathBuf> {
  match std::fs::create_dir_all(dir) {
    Ok(()) if dir.is_dir() => Some(dir.to_path_buf()),
    Ok(()) => None,
    Err(e) => {
      log::warn!(
        "prompt cache: {} is not a usable directory: {e}",
        dir.display()
      );
      None
    }
  }
}

/// The stable per-model prefix of a slot file. Keyed on the model's identity —
/// path **and** header digest — rather than on a launch id, because the whole
/// point is to be found again by a launch that does not exist yet; the header
/// digest is what stops a re-downloaded model from reading back another
/// quantization's KV as if it were its own.
pub fn model_cache_key(model: &ModelId) -> String {
  let seed = format!("{}\u{0}{}", model.path.display(), model.header_hex());
  let digest = blake3::hash(seed.as_bytes());
  digest.as_bytes()[..8]
    .iter()
    .map(|b| format!("{b:02x}"))
    .collect()
}

/// A slot file's name. llama.cpp validates it (`fs_validate_filename`: no path
/// separators, ≤255 bytes), and it has to read back as a slot id, so it is a
/// flat name built from characters that pass both checks.
pub fn slot_file_name(model: &ModelId, slot: i64) -> String {
  format!("{}-slot{slot}.bin", model_cache_key(model))
}

/// The slot a file belongs to, or `None` for a file this feature did not write.
/// Foreign files are left alone rather than deleted: the directory is the
/// engine's own save path, and a caller who pointed it somewhere they use for
/// other things must not have those things pruned.
pub fn slot_id_for_file(file_name: &str, model: &ModelId) -> Option<i64> {
  let prefix = format!("{}-slot", model_cache_key(model));
  let rest = file_name.strip_prefix(&prefix)?;
  let id = rest.strip_suffix(".bin")?;
  if id.is_empty()
    || id.len() > 6
    || !id.bytes().all(|b| b.is_ascii_digit())
    || !id.parse::<i64>().is_ok_and(|n| n >= 0)
  {
    return None;
  }
  id.parse().ok()
}

/// One slot as `GET /slots` reports it. llama.cpp adds the prompt fields only
/// once a slot has held a prompt, so a missing count is not "empty": -1 means
/// "unknown", and a slot that might hold something is worth a save call.
pub struct SlotState {
  pub slot: i64,
  pub prompt_tokens: i64,
  pub is_processing: bool,
}

pub fn parse_slot_states(body: &Value) -> Vec<SlotState> {
  body
    .as_array()
    .into_iter()
    .flatten()
    .filter_map(|row| {
      let slot = row.get("id")?.as_i64()?;
      let prompt_tokens = match row.get("n_prompt_tokens") {
        Some(v) => v.as_i64().unwrap_or(-1),
        None => -1,
      };
      Some(SlotState {
        slot,
        prompt_tokens,
        is_processing: row
          .get("is_processing")
          .and_then(Value::as_bool)
          .unwrap_or(false),
      })
    })
    .collect()
}

/// Which slots are worth a save call: one that holds a prompt and is not in the
/// middle of generating. llama.cpp defers a save request against a busy slot
/// rather than interrupting it, and a stop is the last moment that slot will
/// ever answer.
pub fn slots_to_save(states: &[SlotState]) -> Vec<i64> {
  let mut slots: Vec<i64> = states
    .iter()
    .filter(|s| !s.is_processing && s.prompt_tokens != 0)
    .map(|s| s.slot)
    .collect();
  slots.sort_unstable();
  slots.truncate(MAX_SLOTS_SAVED);
  slots
}

pub struct CacheFile {
  pub path: PathBuf,
  pub file_name: String,
  pub bytes: u64,
  pub modified: SystemTime,
}

/// The directory as it is on disk. Unreadable entries are skipped rather than
/// failing the whole listing: a checkpoint is a bonus, never a requirement.
pub fn list_cache_files(dir: &Path) -> std::io::Result<Vec<CacheFile>> {
  let mut out = Vec::new();
  for item in std::fs::read_dir(dir)? {
    let entry = match item {
      Ok(entry) => entry,
      Err(_) => continue,
    };
    let Ok(file_type) = entry.file_type() else {
      continue;
    };
    if !file_type.is_file() {
      continue;
    }
    let Ok(metadata) = entry.metadata() else {
      continue;
    };
    let Some(file_name) = entry.file_name().to_str().map(str::to_string) else {
      continue;
    };
    let modified = metadata
      .modified()
      .unwrap_or_else(|_| SystemTime::UNIX_EPOCH);
    out.push(CacheFile {
      path: entry.path(),
      file_name,
      bytes: metadata.len(),
      modified,
    });
  }
  Ok(out)
}

/// The shape of a file this feature wrote, whatever model it belongs to. The
/// save path is the engine's own directory and a caller may point it somewhere
/// they keep other things; those are neither counted against the cap nor chosen
/// to pay it.
pub fn is_slot_file(file_name: &str) -> bool {
  let Some(rest) = file_name.strip_suffix(".bin") else {
    return false;
  };
  let Some((key, slot)) = rest.split_once("-slot") else {
    return false;
  };
  key.len() == 16
    && key.bytes().all(|b| b.is_ascii_hexdigit())
    && !slot.is_empty()
    && slot.len() <= 6
    && slot.bytes().all(|b| b.is_ascii_digit())
}

/// Oldest first until the directory fits `cap_bytes`. The newest checkpoints are
/// the ones a conversation is most likely to resume from, so they are kept and
/// the tail of history is what goes. A single file larger than the whole budget
/// is doomed on its own — a cap that only applies when there are two files is
/// not a cap.
pub fn prune_plan(files: &[CacheFile], cap_bytes: u64) -> Vec<PathBuf> {
  let mut ordered: Vec<&CacheFile> = files
    .iter()
    .filter(|file| is_slot_file(&file.file_name))
    .collect();
  ordered.sort_by(|a, b| b.modified.cmp(&a.modified));
  let mut doomed = Vec::new();
  let mut used = 0u64;
  for file in ordered {
    if used.saturating_add(file.bytes) > cap_bytes {
      doomed.push(file.path.clone());
    } else {
      used = used.saturating_add(file.bytes);
    }
  }
  doomed
}

/// This model's slot files that the round just now did not write. A model's
/// cache is a snapshot of its current slots, not a history: keeping the older
/// generations would spend the shared cap on prompts that no longer exist, and
/// could hand a future launch a slot file that no longer matches its slots.
pub fn stale_files(files: &[CacheFile], model: &ModelId, just_written: &[String]) -> Vec<PathBuf> {
  files
    .iter()
    .filter(|file| {
      slot_id_for_file(&file.file_name, model).is_some() && !just_written.contains(&file.file_name)
    })
    .map(|file| file.path.clone())
    .collect()
}

/// A save that wrote no token wrote nothing worth restoring. llama.cpp reports
/// `n_saved: 0` for an idle slot, and leaves a header-only file behind.
pub fn save_produced_state(result: &Value) -> bool {
  result.get("n_saved").and_then(Value::as_i64).unwrap_or(0) > 0
}

/// Which files this model may take back, oldest slot id first. llama.cpp wraps
/// an out-of-range slot id (`id % n_slots`), so a saved slot that no longer
/// exists lands on a real one rather than failing; asking in slot order is what
/// makes the mapping between launch N and launch N+1 predictable.
pub fn files_to_restore(files: &[CacheFile], model: &ModelId) -> Vec<(i64, PathBuf)> {
  let mut targets: Vec<(i64, PathBuf)> = files
    .iter()
    .filter_map(|file| {
      slot_id_for_file(&file.file_name, model).map(|slot| (slot, file.path.clone()))
    })
    .collect();
  targets.sort_by_key(|(slot, _)| *slot);
  targets.truncate(MAX_SLOTS_SAVED);
  targets
}

/// The directory this launch was told to use, or `None` when it has no prompt
/// cache to keep — a launch started before this feature existed, one with the
/// config switch off, one whose caller manages the path themselves, one whose
/// build has no slot endpoints. Those are exactly the facts `compose` acted on,
/// so reading them back is what makes both hooks a no-op on a launch that never
/// got the flag: nothing to infer, nothing to guess.
fn slot_dir_for_launch(model: &ManagedModel) -> Option<PathBuf> {
  slot_dir_of(&model.params().launch_config)
}

/// The same read straight off a launch config. That pair of keys is the whole
/// contract between `seed_launch_knobs`, `compose`, and these two hooks.
fn slot_dir_of(config: &BTreeMap<String, String>) -> Option<PathBuf> {
  if config.get(LLAMACPP_KNOB_SLOT_SAVE).map(String::as_str) != Some("true") {
    return None;
  }
  config.get(LLAMACPP_KNOB_SLOT_SAVE_DIR).map(PathBuf::from)
}

/// A launch is about to be stopped. Ask the engine, slot by slot, to write down
/// what it has. Every failure here is a log line: the stop itself is already
/// doing something the user asked for, and a prompt cache that could not be
/// saved is paid for later by reprocessing a prompt.
pub async fn save_before_stop(ctx: &MethodContext, model: &ManagedModel) {
  let Some(dir) = slot_dir_for_launch(model) else {
    return;
  };
  let states = match list_slots(model.port()).await {
    Ok(states) => states,
    Err(e) => {
      log::debug!("prompt cache: no slot list from {}: {e}", model.port());
      return;
    }
  };
  let slots = slots_to_save(&states);
  if slots.is_empty() {
    log::debug!(
      "prompt cache: nothing resident in {} to save",
      model.id().path.display()
    );
    return;
  }
  let dir = match prepare_slot_dir(&dir) {
    Some(dir) => dir,
    None => return,
  };

  let model_id = model.id();
  let port = model.port();
  let round = Instant::now();
  let mut written = Vec::new();
  let mut bytes = 0u64;
  let mut tokens = 0i64;
  for slot in slots {
    let Some(remaining) = ROUND_TIMEOUT.checked_sub(round.elapsed()) else {
      log::warn!(
        "prompt cache: saving {} ran past {} s and was cut off",
        port,
        ROUND_TIMEOUT.as_secs()
      );
      break;
    };
    let file_name = slot_file_name(model_id, slot);
    match slot_action(
      port,
      "save",
      slot,
      &file_name,
      remaining.min(SLOT_CALL_TIMEOUT),
    )
    .await
    {
      Ok(result) => {
        if !save_produced_state(&result) {
          // Nothing behind it: keeping the file would cost its header to the
          // cap and a wasted restore call next launch.
          remove_file_quietly(&dir.join(&file_name));
          continue;
        }
        tokens += result.get("n_saved").and_then(Value::as_i64).unwrap_or(0);
        bytes += result.get("n_written").and_then(Value::as_u64).unwrap_or(0);
        written.push(file_name);
      }
      Err(e) => {
        log::debug!("prompt cache: slot {slot} of {port} refused to save: {e}");
      }
    }
  }

  // Cleanup is part of the save: this model's superseded generations first, then
  // the cap over whatever the directory now holds. Both run after the writes, so
  // what is measured is what was just written.
  let Ok(files) = list_cache_files(&dir) else {
    return;
  };
  let cap = llamacpp_prompt_cache_cap(ctx);
  let doomed = stale_files(&files, model_id, &written)
    .into_iter()
    .chain(prune_plan(&files, cap))
    .collect::<Vec<_>>();
  for path in &doomed {
    remove_file_quietly(path);
  }
  if written.is_empty() {
    return;
  }
  // What the stop paid, in the round it was already spending on this launch.
  let took = round.elapsed();
  log::info!(
    "prompt cache: saved {} slot(s) of {} in {} ms — {tokens} tokens, {} MiB{} dropped",
    written.len(),
    model_id.path.display(),
    took.as_millis(),
    bytes / 1_048_576,
    if doomed.is_empty() {
      String::new()
    } else {
      format!(", {} older file(s)", doomed.len())
    }
  );
}

/// A launch is Ready and there are files for its model. Hand them back before
/// the first request arrives, into the slots they came from.
pub async fn restore_after_ready(model: &ManagedModel) {
  let Some(dir) = slot_dir_for_launch(model) else {
    return;
  };
  let files = match list_cache_files(&dir) {
    Ok(files) => files,
    Err(e) => {
      log::debug!("prompt cache: cannot read {}: {e}", dir.display());
      return;
    }
  };
  let targets = files_to_restore(&files, model.id());
  if targets.is_empty() {
    return;
  }

  let port = model.port();
  let round = Instant::now();
  let mut slots = 0usize;
  let mut tokens = 0i64;
  let mut bytes = 0u64;
  for (slot, path) in targets {
    let Some(remaining) = ROUND_TIMEOUT.checked_sub(round.elapsed()) else {
      log::warn!(
        "prompt cache: the restore into {port} ran past {} s and was cut off",
        ROUND_TIMEOUT.as_secs()
      );
      break;
    };
    let Some(file_name) = path.file_name().map(|s| s.to_string_lossy().into_owned()) else {
      continue;
    };
    match slot_action(
      port,
      "restore",
      slot,
      &file_name,
      remaining.min(SLOT_CALL_TIMEOUT),
    )
    .await
    {
      Ok(result) => {
        slots += 1;
        tokens += result
          .get("n_restored")
          .and_then(Value::as_i64)
          .unwrap_or(0);
        bytes += result.get("n_read").and_then(Value::as_u64).unwrap_or(0);
        // Consumed either way: the engine has it, or the file does not describe
        // this launch and re-reading it every start would achieve nothing.
        remove_file_quietly(&path);
      }
      Err(e) => {
        log::debug!("prompt cache: slot {slot} of {port} refused to restore: {e}");
        remove_file_quietly(&path);
      }
    }
  }
  if slots > 0 {
    // What the relaunch paid to become Ready with the conversation back in it.
    let took = round.elapsed();
    log::info!(
      "prompt cache: restored {slots} slot(s) into {} in {} ms — {tokens} tokens, {} MiB",
      model.id().path.display(),
      took.as_millis(),
      bytes / 1_048_576
    );
  }
}

/// `backend.llamacpp.prompt_cache_max_mb`, read where it is honoured. The
/// generic daemon path must not know what a llama.cpp knob is, so this reaches
/// through the same typed config the rest of the backend reads.
fn llamacpp_prompt_cache_cap(ctx: &MethodContext) -> u64 {
  ctx.backend.llamacpp.prompt_cache_max_bytes()
}

async fn list_slots(port: u16) -> Result<Vec<SlotState>, String> {
  let body = http_json("GET", "/slots", port, None, SLOTS_LIST_TIMEOUT).await?;
  Ok(parse_slot_states(&body))
}

/// `POST /slots/{id}?action=save|restore` with `{"filename": …}`. The engine
/// joins that name to its own `--slot-save-path`, which is the directory
/// `slot_save_dir` records — the two are the same decision made once.
async fn slot_action(
  port: u16,
  action: &str,
  slot: i64,
  file_name: &str,
  timeout: Duration,
) -> Result<Value, String> {
  let path = format!("/slots/{slot}?action={action}");
  let body = serde_json::json!({ "filename": file_name });
  http_json("POST", &path, port, Some(body), timeout).await
}

/// The server is loopback-only and plain HTTP, like every other endpoint the
/// daemon already reads (`/props`, `/metrics`) — those go through
/// [`crate::net::http`], which has no read-timeout knob. A save against a slot
/// the engine has stopped answering must not sit open until the process-wide
/// default does, so this one speaks HTTP directly and bounds every call.
async fn http_json(
  method: &str,
  path: &str,
  port: u16,
  body: Option<Value>,
  timeout: Duration,
) -> Result<Value, String> {
  let body_bytes = body.map(|v| v.to_string().into_bytes());
  let request = async {
    let mut sock = TcpStream::connect(("127.0.0.1", port))
      .await
      .map_err(|e| format!("connect 127.0.0.1:{port}: {e}"))?;
    let mut head = format!("{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n");
    let headers = match &body_bytes {
      Some(bytes) => format!(
        "Content-Type: application/json\r\nContent-Length: {}\r\n",
        bytes.len()
      ),
      None => "Content-Length: 0\r\n".to_string(),
    };
    head.push_str(&headers);
    head.push_str("Connection: close\r\n\r\n");
    sock
      .write_all(head.as_bytes())
      .await
      .map_err(|e| format!("request: {e}"))?;
    if let Some(bytes) = &body_bytes {
      sock
        .write_all(bytes)
        .await
        .map_err(|e| format!("body: {e}"))?;
    }
    read_response(&mut sock).await
  };
  match tokio::time::timeout(timeout, request).await {
    Ok(result) => result,
    Err(_) => Err(format!("timed out after {} s", timeout.as_secs())),
  }
}

async fn read_response<S: AsyncRead + Unpin>(sock: &mut S) -> Result<Value, String> {
  let mut head = Vec::new();
  let mut byte = [0u8; 1];
  while head.len() < MAX_RESPONSE_BYTES {
    let read = sock
      .read(&mut byte)
      .await
      .map_err(|e| format!("read: {e}"))?;
    if read == 0 {
      break;
    }
    head.push(byte[0]);
    if head.ends_with(b"\r\n\r\n") || head.ends_with(b"\n\n") {
      break;
    }
  }
  let head_text = String::from_utf8_lossy(&head);
  let mut status = 0u16;
  let mut content_length: Option<usize> = None;
  let mut chunked = false;
  for line in head_text.lines() {
    if let Some(rest) = line.strip_prefix("HTTP/1.") {
      status = rest
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.parse().ok())
        .unwrap_or(0);
    } else if let Some(value) = line
      .strip_prefix("content-length:")
      .or_else(|| line.strip_prefix("Content-Length:"))
    {
      content_length = value.trim().parse().ok();
    } else if line
      .to_ascii_lowercase()
      .replace(' ', "")
      .starts_with("transfer-encoding:chunked")
    {
      chunked = true;
    }
  }
  if content_length.is_some_and(|len| len > MAX_RESPONSE_BYTES) {
    return Err(format!("response body over {} bytes", MAX_RESPONSE_BYTES));
  }
  // Read the body with the same ceiling rather than trusting a declared length:
  // `Connection: close` is what ends it, and a server that never closes would
  // otherwise hold the request open until the timeout while the body grew.
  let mut body = Vec::with_capacity(content_length.unwrap_or(8 * 1024).min(MAX_RESPONSE_BYTES));
  let mut buffer = vec![0u8; 16 * 1024];
  while body.len() <= MAX_RESPONSE_BYTES {
    let read = sock
      .read(&mut buffer)
      .await
      .map_err(|e| format!("read: {e}"))?;
    if read == 0 {
      break;
    }
    body.extend_from_slice(&buffer[..read]);
  }
  if body.len() > MAX_RESPONSE_BYTES {
    return Err(format!("response body over {} bytes", MAX_RESPONSE_BYTES));
  }
  if status >= 400 {
    return Err(engine_error(status, &body));
  }
  let payload;
  if chunked {
    payload = decode_chunked(&body)?;
  } else {
    payload = body;
  }
  serde_json::from_slice(&payload).map_err(|e| format!("unreadable response: {e}"))
}

/// llama.cpp answers a refusal with `{"error": "…", "details": "…"}`, and the
/// details are what tells a wrong slot id apart from a model that no longer fits
/// its own context.
fn engine_error(status: u16, body: &[u8]) -> String {
  let text = String::from_utf8_lossy(body);
  match serde_json::from_str::<Value>(&text) {
    Ok(value) => {
      let message = value
        .get("error")
        .and_then(Value::as_str)
        .unwrap_or(text.trim());
      match value.get("details").and_then(Value::as_str) {
        Some(details) => format!("{status} {message} ({details})"),
        None => format!("{status} {message}"),
      }
    }
    Err(_) => format!("{status} {}", text.trim()),
  }
}

/// Body without `Content-Length`, so only the close marks the end — which is why
/// the request asks for `Connection: close`. Chunked framing is decoded rather
/// than assumed away: llama-server's own endpoints use `Content-Length`, but a
/// proxy or a fork in between may not.
fn decode_chunked(tail: &[u8]) -> Result<Vec<u8>, String> {
  let mut out = Vec::new();
  let mut rest = tail;
  loop {
    let Some(size) = read_chunk_size(&mut rest) else {
      break;
    };
    if size == 0 {
      break;
    }
    if rest.len() < size {
      break;
    }
    out.extend_from_slice(&rest[..size]);
    rest = &rest[size..];
    if rest.starts_with(b"\r\n") {
      rest = &rest[2..];
    } else if rest.starts_with(b"\n") {
      rest = &rest[1..];
    }
  }
  Ok(out)
}

/// `"5\r\ndata…"` → 5, with the line consumed. A frame header that is not a
/// number means the framing is not what was assumed, and the caller reads what it
/// has.
fn read_chunk_size(rest: &mut &[u8]) -> Option<usize> {
  let newline = rest.iter().position(|b| *b == b'\n')?;
  let size = String::from_utf8_lossy(&rest[..newline])
    .split(';')
    .next()?
    .trim()
    .parse()
    .ok()?;
  *rest = &rest[newline + 1..];
  Some(size)
}

fn remove_file_quietly(path: &Path) {
  if let Err(e) = std::fs::remove_file(path) {
    if e.kind() != std::io::ErrorKind::NotFound {
      log::debug!("prompt cache: could not remove {}: {e}", path.display());
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::ffi::OsString;
  use std::time::UNIX_EPOCH;

  fn model(name: &str) -> ModelId {
    ModelId {
      path: Path::new("/models/").join(name),
      header_blake3: [7u8; 32],
    }
  }

  fn file(name: &str, bytes: u64, age_secs: u64) -> CacheFile {
    CacheFile {
      path: Path::new("/state/prompt-cache").join(name),
      file_name: name.to_string(),
      bytes,
      modified: UNIX_EPOCH + Duration::from_secs(1_700_000_000 - age_secs),
    }
  }

  fn doomed(names: &[&str]) -> Vec<PathBuf> {
    names
      .iter()
      .map(|n| Path::new("/state/prompt-cache").join(n))
      .collect()
  }

  fn config(dir: Option<&str>, capability: &str) -> BTreeMap<String, String> {
    let mut config = BTreeMap::new();
    config.insert(LLAMACPP_KNOB_SLOT_SAVE.to_string(), capability.to_string());
    if let Some(dir) = dir {
      config.insert(LLAMACPP_KNOB_SLOT_SAVE_DIR.to_string(), dir.to_string());
    }
    config
  }

  #[test]
  fn a_slot_file_is_named_after_its_model_and_slot() {
    let name = slot_file_name(&model("a.gguf"), 2);
    assert_eq!(name.len(), 16 + "-slot2.bin".len());
    assert!(name.ends_with("-slot2.bin"), "{name}");
  }

  /// The identity the whole feature keys on: two files sharing a name never
  /// share a cache entry, and the same file always does.
  #[test]
  fn the_key_follows_model_identity_not_the_path_alone() {
    let a = model("same.gguf");
    let mut other_header = model("same.gguf");
    other_header.header_blake3 = [9u8; 32];
    assert_eq!(model_cache_key(&a), model_cache_key(&a));
    assert_ne!(model_cache_key(&a), model_cache_key(&other_header));

    let mut renamed = model("renamed.gguf");
    renamed.header_blake3 = a.header_blake3;
    assert_ne!(model_cache_key(&a), model_cache_key(&renamed));
  }

  #[test]
  fn a_slot_file_reads_back_its_slot() {
    let m = model("a.gguf");
    for slot in [0i64, 1, 7, 127] {
      let name = slot_file_name(&m, slot);
      assert_eq!(slot_id_for_file(&name, &m), Some(slot), "{name}");
    }
  }

  /// Files the feature did not write are left alone. The save path is the
  /// engine's own directory and a caller may point it at one they use for other
  /// things; the cap spends itself on what this feature wrote.
  #[test]
  fn foreign_files_are_neither_read_nor_named_for_deletion() {
    let m = model("a.gguf");
    for name in [
      "foreign.bin",
      "notes.txt",
      "othermodel-slot1.bin",
      &format!("{}-slotX.bin", model_cache_key(&m)),
      &format!("{}-slot-1.bin", model_cache_key(&m)),
      &format!("{}-slot9999999999.bin", model_cache_key(&m)),
      &format!("{}-slot1.dat", model_cache_key(&m)),
    ] {
      assert_eq!(slot_id_for_file(name, &m), None, "{name}");
    }
    assert!(
      prune_plan(&[file("foreign.bin", 9_000_000_000, 0)], 1_000).is_empty(),
      "a file this feature did not write is not the feature's to delete"
    );
    // …and it does not spend their budget either.
    let m2 = model("b.gguf");
    assert_eq!(
      prune_plan(
        &[
          file("foreign.bin", 9_000_000_000, 0),
          file(&slot_file_name(&m2, 0), 1_000, 0)
        ],
        1_000
      ),
      Vec::<PathBuf>::new()
    );
    assert!(is_slot_file(&slot_file_name(&m2, 3)));
    assert_eq!(
      stale_files(&[file("foreign.bin", 10, 0)], &m, &[]),
      Vec::<PathBuf>::new()
    );
  }

  #[test]
  fn slots_holding_state_are_saved_and_busy_ones_are_not() {
    // The shape `GET /slots` returns after traffic: prompt fields appear only on
    // slots that have held a prompt.
    let body = serde_json::json!([
      { "id": 1, "n_ctx": 4096, "is_processing": false, "id_task": 3, "n_prompt_tokens": 812 },
      { "id": 0, "n_ctx": 4096, "is_processing": false, "n_prompt_tokens": 0 },
      { "id": 2, "n_ctx": 4096, "is_processing": true, "id_task": 5, "n_prompt_tokens": 12 },
    ]);
    let states = parse_slot_states(&body);
    assert_eq!(states.len(), 3);
    assert_eq!(
      slots_to_save(&states),
      vec![1],
      "a mid-generation slot is not saved, and an empty one has nothing to save"
    );
  }

  /// Older and trimmed builds omit the counters. "Unknown" is not "empty" — the
  /// engine answers a save against a truly idle slot with `n_saved: 0`, and that
  /// is what costs nothing.
  #[test]
  fn a_build_without_token_counts_is_saved_anyway() {
    let body = serde_json::json!([
      { "id": 1, "n_ctx": 4096, "is_processing": false },
      { "id": 0, "n_ctx": 4096, "is_processing": false },
    ]);
    assert_eq!(slots_to_save(&parse_slot_states(&body)), vec![0, 1]);
  }

  #[test]
  fn a_response_that_is_not_a_slot_list_yields_nothing_to_save() {
    assert!(parse_slot_states(&serde_json::json!({ "error": "no slots" })).is_empty());
    assert!(slots_to_save(&[]).is_empty());
    assert!(!save_produced_state(&serde_json::json!({ "n_saved": 0 })));
    assert!(save_produced_state(&serde_json::json!({ "n_saved": 3 })));
  }

  #[test]
  fn saves_past_the_first_few_slots_are_left_alone() {
    let states: Vec<SlotState> = (0..12)
      .map(|slot| SlotState {
        slot,
        prompt_tokens: 10,
        is_processing: false,
      })
      .collect();
    assert_eq!(slots_to_save(&states).len(), MAX_SLOTS_SAVED);
  }

  /// The cap is a byte budget, and the newest checkpoints are the ones a
  /// conversation is likeliest to resume from.
  #[test]
  fn the_cap_drops_oldest_first_and_keeps_the_newest() {
    let m = model("a.gguf");
    let files = [
      file(&slot_file_name(&m, 0), 1_000, 3_000),
      file(&slot_file_name(&m, 1), 1_000, 2_000),
      file(&slot_file_name(&m, 2), 1_000, 1_000),
    ];
    assert_eq!(
      prune_plan(&files, 2_500),
      doomed(&[slot_file_name(&m, 0).as_str()])
    );
    assert_eq!(prune_plan(&files, 3_000), Vec::<PathBuf>::new());
    assert_eq!(
      prune_plan(&[file(&slot_file_name(&m, 0), 5_000, 0)], 1_000),
      doomed(&[slot_file_name(&m, 0).as_str()]),
      "one file bigger than the whole budget is still over it"
    );
  }

  #[test]
  fn stale_slot_files_of_this_model_only_go() {
    let m = model("a.gguf");
    let mine = slot_file_name(&m, 0);
    let other = model("b.gguf");
    let files = [
      file(&mine, 10, 100),
      file(&slot_file_name(&other, 0), 10, 100),
      file("foreign.bin", 10, 100),
    ];
    assert_eq!(
      stale_files(&files, &m, &[slot_file_name(&m, 1)]),
      doomed(&[mine.as_str()]),
      "this model's superseded generation goes; nobody else's files do"
    );
  }

  #[test]
  fn empty_saves_keep_nothing() {
    assert!(!save_produced_state(&serde_json::json!({})));
    assert!(!save_produced_state(&serde_json::json!({ "n_saved": 0 })));
    assert!(save_produced_state(&serde_json::json!({ "n_saved": 1 })));
  }

  #[test]
  fn restore_targets_this_model_in_slot_order() {
    let m = model("a.gguf");
    let files = [
      file(&slot_file_name(&m, 2), 10, 500),
      file(&slot_file_name(&m, 0), 10, 500),
      file(&slot_file_name(&m, 11), 10, 500),
      file(&slot_file_name(&model("b.gguf"), 1), 10, 500),
      file("foreign.bin", 10, 500),
    ];
    let targets = files_to_restore(&files, &m);
    assert_eq!(
      targets.iter().map(|(slot, _)| *slot).collect::<Vec<_>>(),
      vec![0, 2, 11]
    );
  }

  /// A launch keeps a prompt cache only when the seeding and the capability both
  /// said so: either key missing means the hooks have nothing to do.
  #[test]
  fn a_launch_without_the_recorded_directory_and_flag_keeps_nothing() {
    assert_eq!(
      slot_dir_of(&config(Some("/state/prompt-cache"), "true")),
      Some(PathBuf::from("/state/prompt-cache"))
    );
    assert_eq!(
      slot_dir_of(&config(Some("/state/prompt-cache"), "false")),
      None
    );
    assert_eq!(slot_dir_of(&config(None, "true")), None);
    assert_eq!(slot_dir_of(&config(None, "absent")), None);
  }

  /// Two daemons on one host get one directory each.
  #[test]
  fn checkpoints_are_per_daemon() {
    assert_eq!(
      slot_dir_for_state(Path::new("/state/a")),
      PathBuf::from("/state/a/prompt-cache")
    );
    assert_ne!(
      slot_dir_for_state(Path::new("/state/a")),
      slot_dir_for_state(Path::new("/state/b"))
    );
  }

  /// The engine validates `--slot-save-path` at argument-parse time, so a
  /// directory the launch names has to be there before the flag is emitted.
  #[test]
  fn a_missing_checkpoint_directory_is_made_before_the_engine_is_asked_for_it() {
    let root = crate::util::test_temp::unique_temp_dir("prompt-cache-prepare");
    let dir = slot_dir_for_state(&root);
    assert!(!dir.exists());
    assert_eq!(prepare_slot_dir(&dir), Some(dir.clone()));
    assert!(dir.is_dir());
    std::fs::remove_dir_all(&root).ok();
  }

  #[test]
  fn a_path_that_is_not_a_directory_is_no_checkpoint_at_all() {
    let root = crate::util::test_temp::unique_temp_dir("prompt-cache-not-a-dir");
    let path = root.join("prompt-cache");
    std::fs::write(&path, "not a directory").unwrap();
    assert_eq!(prepare_slot_dir(&path), None);
    std::fs::remove_dir_all(&root).ok();
  }

  #[test]
  fn the_engine_refusal_is_read_out_of_the_error_body() {
    let body = br#"{"error":"Unable to restore slot","details":"No available space in KV cache or invalid slot save file"}"#;
    let message = engine_error(400, body);
    assert!(message.contains("Unable to restore slot"), "{message}");
    assert!(message.contains("invalid slot save file"), "{message}");
    assert!(message.starts_with("400"), "{message}");
  }

  #[tokio::test]
  async fn a_response_without_a_content_length_is_read_to_the_close() {
    let raw = concat!(
      "HTTP/1.1 200 OK\r\n",
      "Connection: close\r\n\r\n",
      "{\"id_slot\":0,\"n_saved\":1}"
    );
    let (mut stream, mut writer) = tokio::io::duplex(4 * 1024 * 1024);
    tokio::spawn(async move {
      tokio::io::AsyncWriteExt::write_all(&mut writer, raw.as_bytes())
        .await
        .unwrap();
      tokio::io::AsyncWriteExt::shutdown(&mut writer)
        .await
        .unwrap();
    });
    let parsed = read_response(&mut stream).await.unwrap();
    assert_eq!(parsed["n_saved"].as_i64(), Some(1));
  }

  #[test]
  fn chunk_frames_are_read_and_the_framing_dropped() {
    let raw = "5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n";
    let out = decode_chunked(raw.as_bytes()).unwrap();
    assert_eq!(String::from_utf8(out).unwrap(), "hello world");
  }

  #[tokio::test]
  async fn a_chunked_response_is_decoded_to_its_frames() {
    let raw = concat!(
      "HTTP/1.1 200 OK\r\n",
      "Transfer-Encoding: chunked\r\n\r\n",
      "11\r\n{\"n_saved\":\r\n2\r\n9}\r\n0\r\n\r\n"
    );
    let (mut stream, mut writer) = tokio::io::duplex(4 * 1024 * 1024);
    tokio::spawn(async move {
      tokio::io::AsyncWriteExt::write_all(&mut writer, raw.as_bytes())
        .await
        .unwrap();
      tokio::io::AsyncWriteExt::shutdown(&mut writer)
        .await
        .unwrap();
    });
    let parsed = read_response(&mut stream).await.unwrap();
    assert_eq!(
      parsed["n_saved"].as_i64(),
      Some(9),
      "the frames, not the framing: {parsed}"
    );
  }

  #[tokio::test]
  async fn a_rejected_request_keeps_what_the_engine_said() {
    let raw = concat!(
      "HTTP/1.1 400 Bad Request\r\n",
      "Content-Length: 24\r\n\r\n",
      "{\"error\":\"bad slot save\"}"
    );
    let (mut stream, mut writer) = tokio::io::duplex(4 * 1024 * 1024);
    tokio::spawn(async move {
      tokio::io::AsyncWriteExt::write_all(&mut writer, raw.as_bytes())
        .await
        .unwrap();
      tokio::io::AsyncWriteExt::shutdown(&mut writer)
        .await
        .unwrap();
    });
    let err = read_response(&mut stream).await.unwrap_err();
    assert!(err.contains("bad slot save"), "{err}");
    assert!(err.starts_with("400"), "{err}");
  }

  #[tokio::test]
  async fn an_overlong_body_is_refused_rather_than_read_for_ever() {
    let big = "x".repeat(MAX_RESPONSE_BYTES + 4096);
    let raw = format!("HTTP/1.1 200 OK\r\n\r\n{big}");
    let (mut stream, mut writer) = tokio::io::duplex(4 * 1024 * 1024);
    tokio::spawn(async move {
      tokio::io::AsyncWriteExt::write_all(&mut writer, raw.as_bytes())
        .await
        .unwrap();
      tokio::io::AsyncWriteExt::shutdown(&mut writer)
        .await
        .unwrap();
    });
    let err = read_response(&mut stream).await.unwrap_err();
    assert!(err.contains("over"), "{err}");

    // A declared length over the ceiling is refused before a single body byte is
    // read, so the cap costs nothing on a response that is far too big.
    let (mut stream, mut writer) = tokio::io::duplex(4 * 1024 * 1024);
    tokio::spawn(async move {
      tokio::io::AsyncWriteExt::write_all(
        &mut writer,
        b"HTTP/1.1 200 OK\r\nContent-Length: 999999999999\r\n\r\n",
      )
      .await
      .unwrap();
      tokio::io::AsyncWriteExt::shutdown(&mut writer)
        .await
        .unwrap();
    });
    let err = read_response(&mut stream).await.unwrap_err();
    assert!(err.contains("over"), "{err}");
  }

  #[test]
  fn an_extra_slot_save_path_is_recognised_in_either_form() {
    // `--slot-save-path /dir` and `--slot-save-path=/dir` are the same flag to
    // the engine, so both have to be recognised here too: a launch that carries
    // one of them has already answered the question of where the files go.
    let spaced = vec![
      OsString::from("--port"),
      OsString::from("8080"),
      OsString::from("--slot-save-path"),
      OsString::from("/home/me/slots"),
    ];
    assert!(super::super::extras_carry(&spaced, "--slot-save-path"));
    assert_eq!(
      super::super::extras_value(&spaced, "--slot-save-path").as_deref(),
      Some("/home/me/slots")
    );
    let inline = vec![OsString::from("--slot-save-path=/home/me/slots")];
    assert_eq!(
      super::super::extras_value(&inline, "--slot-save-path").as_deref(),
      Some("/home/me/slots")
    );
    let absent = vec![
      OsString::from("--slots"),
      OsString::from("--port"),
      OsString::from("8080"),
    ];
    assert!(!super::super::extras_carry(&absent, "--slot-save-path"));
    assert!(super::super::extras_value(&absent, "--slot-save-path").is_none());
  }
}
