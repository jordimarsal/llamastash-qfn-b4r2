//! Prompt cache across an unload: integration coverage for keeping a
//! llama.cpp model's prompt state when the daemon unloads it.
//!
//! A real `run_foreground` daemon, a real proxy request, the real
//! `stop_model` → `Backend::stop` path, and a real second proxy request that
//! auto-starts the model again — with the `fake_llama_server` fixture standing
//! in for `llama-server`'s slot save/restore endpoints (`GET /slots`,
//! `POST /slots/<id>?action=save|restore`).
//!
//! The fixture advertises `--slot-save-path` in its `--help` only under
//! `FAKE_LLAMA_SLOT_SAVE=1`, which is how these tests answer the daemon's
//! capability probe "yes, this build keeps prompt state". The two tests in this
//! file differ in what the *config* says, not in what the build supports: one
//! leaves `backend.llamacpp.keep_prompt_cache` at its factory `true`, the other
//! turns it off and asserts the launch is exactly what it used to be.

#![cfg(feature = "test-fixtures")]

use std::{
  net::{Ipv4Addr, SocketAddr, TcpListener as StdTcpListener},
  path::{Path, PathBuf},
  time::Duration,
};

use llamastash::backend::llama_cpp::LlamaCppConfig;
use llamastash::config::loader::{PortRange, ProxyConfig};
use llamastash::config::BackendConfig;
use llamastash::daemon::discovery_task::DiscoveryOptions;
use llamastash::daemon::{run_foreground, DaemonOptions};
use llamastash::discovery::scanner::{ScanOptions, ScanRoot};
use llamastash::discovery::watcher::WatcherOptions;
use llamastash::discovery::ModelSource;
use llamastash::gguf::test_fixtures::build_minimal_gguf;
use llamastash::ipc::Client;
use llamastash::test_support::{http_get, http_post};
use serde_json::Value;
use tokio::time::{sleep, timeout};

const PROMPT_RECORD_MARK: &str = "-prompt-cache";

fn unique_temp(label: &str) -> PathBuf {
  llamastash::test_support::unique_temp_dir("ls-pc", label)
}

fn fake_binary() -> PathBuf {
  PathBuf::from(env!("CARGO_BIN_EXE_fake_llama_server"))
}

fn pick_free_port() -> u16 {
  let l = StdTcpListener::bind((Ipv4Addr::LOCALHOST, 0)).expect("bind ephemeral");
  l.local_addr().expect("local_addr").port()
}

fn allocate_port_range() -> PortRange {
  llamastash::test_support::allocate_port_range(8)
}

async fn wait_for_socket(path: &Path) {
  let deadline = std::time::Instant::now() + Duration::from_secs(30);
  loop {
    if std::time::Instant::now() > deadline {
      panic!("daemon socket never appeared: {}", path.display());
    }
    if Client::connect(path).await.is_ok() {
      return;
    }
    sleep(Duration::from_millis(20)).await;
  }
}

async fn wait_for_proxy(addr: SocketAddr) {
  let deadline = std::time::Instant::now() + Duration::from_secs(30);
  loop {
    if std::time::Instant::now() > deadline {
      panic!("proxy never answered on {addr}");
    }
    let (status, _, _) = http_get(addr, "/health", &[]).await;
    if status == 200 {
      return;
    }
    sleep(Duration::from_millis(25)).await;
  }
}

/// A daemon rooted in `state_dir`, discovering one GGUF, with the proxy on a
/// port of the caller's choosing. `keep_prompt_cache` is what the config says.
async fn start_daemon(
  state_dir: &Path,
  scan_root: &Path,
  proxy_port: u16,
  keep_prompt_cache: bool,
) -> tokio::task::JoinHandle<()> {
  let opts = DaemonOptions {
    binary: Some(fake_binary()),
    port_range: allocate_port_range(),
    discovery: DiscoveryOptions {
      scan_roots: vec![ScanRoot {
        path: scan_root.to_path_buf(),
        source: ModelSource::UserPath,
      }],
      scan: ScanOptions::default(),
      watcher: WatcherOptions {
        debounce: Duration::from_millis(75),
        periodic_rescan: Duration::from_secs(30),
        channel_capacity: 16,
      },
      backend: Default::default(),
      backend_force: Default::default(),
    },
    proxy: ProxyConfig {
      enabled: true,
      port: Some(proxy_port),
      // Long enough that nothing here is evicted by the sweeper: these tests
      // unload the model by hand.
      idle_ttl_secs: 0,
      ..ProxyConfig::default()
    },
    backend: BackendConfig {
      llamacpp: LlamaCppConfig {
        keep_prompt_cache,
        ..LlamaCppConfig::default()
      },
      ..BackendConfig::default()
    },
    ..DaemonOptions::rooted_at(state_dir.to_path_buf())
  };
  let handle = tokio::spawn(async move {
    let _ = run_foreground(opts).await;
  });
  wait_for_socket(state_dir).await;
  handle
}

async fn shutdown(handle: tokio::task::JoinHandle<()>, socket: &Path) {
  if let Ok(mut client) = Client::connect(socket).await {
    let _ = client.call("shutdown", None).await;
  }
  let _ = timeout(Duration::from_secs(5), handle).await;
}

/// The one running model the daemon reports: `(launch_id, port)`. Polls until
/// a row reaches `ready`, which is also the point where a save or a restore
/// could be asked of it.
async fn ready_launch(client: &mut Client) -> (String, u16) {
  let deadline = std::time::Instant::now() + Duration::from_secs(60);
  loop {
    let body = client.call("status", None).await.expect("status");
    for row in body["models"].as_array().cloned().unwrap_or_default() {
      if row["state"]["state"] == "ready" {
        let launch_id = row["launch_id"].as_str().expect("launch_id").to_string();
        let port = row["port"].as_u64().expect("port") as u16;
        return (launch_id, port);
      }
    }
    if std::time::Instant::now() > deadline {
      panic!("no model reached Ready via status");
    }
    sleep(Duration::from_millis(50)).await;
  }
}

/// Initial discovery races the accept loop: wait until the seeded GGUF is in
/// the proxy's own model list.
async fn wait_for_model(proxy: SocketAddr, name: &str) {
  let deadline = std::time::Instant::now() + Duration::from_secs(30);
  loop {
    if std::time::Instant::now() > deadline {
      panic!("proxy never listed {name}");
    }
    let (status, _, body) = http_get(proxy, "/v1/models", &[]).await;
    if status == 200 {
      let value = serde_json::from_slice::<Value>(&body).expect("models json");
      if value["data"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .iter()
        .any(|row| row["id"] == name)
      {
        return;
      }
    }
    sleep(Duration::from_millis(50)).await;
  }
}

/// Send a conversation through the proxy and wait for the answer. The proxy
/// streams whatever the engine answers, so the body is checked as text: the
/// point here is that the request reached a model, not what it said back.
async fn chat(proxy: SocketAddr, model: &str, content: &str) -> String {
  let body =
    format!(r#"{{"model":"{model}","messages":[{{"role":"user","content":"{content}"}}]}}"#,);
  let (status, _, body) = http_post(proxy, "/v1/chat/completions", &body, &[]).await;
  let text = String::from_utf8_lossy(&body).to_string();
  assert_eq!(status, 200, "chat request through the proxy: {text}");
  assert!(text.contains("data:"), "no answer came back: {text}");
  text
}

/// What the engine's own slot currently holds, read straight off its
/// `GET /slots`.
async fn slot_prompt(port: u16) -> u64 {
  let addr = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port);
  let (status, _, body) = http_get(addr, "/slots", &[]).await;
  assert_eq!(status, 200, "GET /slots on {addr}");
  let value = serde_json::from_slice::<Value>(&body).expect("/slots json");
  let slots = value
    .as_array()
    .cloned()
    .or_else(|| value.get("slots").and_then(|s| s.as_array().cloned()))
    .expect("/slots array");
  slots
    .iter()
    .map(|s| s["n_prompt_tokens"].as_u64().unwrap_or(0))
    .sum()
}

/// Ask the engine to save slot 0 by hand. A `400` is the answer a server that
/// was never given `--slot-save-path` gives.
async fn try_slot_save(port: u16) -> u16 {
  let addr = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port);
  let (status, _, _) = http_post(
    addr,
    "/slots/0?action=save",
    r#"{"filename":"probe.bin"}"#,
    &[],
  )
  .await;
  status
}

/// A conversation worth keeping: `words` of filler, which the fixture counts
/// into its slot exactly as a real server would charge for a real prompt.
fn conversation(words: usize) -> String {
  (1..=words)
    .map(|i| format!("word{i}"))
    .collect::<Vec<_>>()
    .join(" ")
}

/// The kept records in the daemon's prompt-cache directory, pending and
/// consumed alike.
fn kept_records(cache: &Path) -> Vec<(PathBuf, Value)> {
  let Ok(reads) = std::fs::read_dir(cache) else {
    return Vec::new();
  };
  reads
    .flatten()
    .filter(|e| {
      e.path()
        .file_name()
        .and_then(|n| n.to_str())
        .map(|n| n.contains(PROMPT_RECORD_MARK))
        .unwrap_or(false)
    })
    .map(|e| {
      let path = e.path();
      let text = std::fs::read_to_string(&path).expect("record readable");
      (
        path,
        serde_json::from_str::<Value>(&text).expect("record json"),
      )
    })
    .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_model_unloaded_by_the_daemon_gets_its_prompt_back() {
  // The fixture's capability answer: this build has the slot endpoints.
  std::env::set_var("FAKE_LLAMA_SLOT_SAVE", "1");

  let state_dir = unique_temp("keep");
  let scan_root = unique_temp("keep-models");
  std::fs::write(scan_root.join("keepme.gguf"), build_minimal_gguf("llama")).unwrap();
  // `rooted_at` roots the logs under the state dir, and the prompt cache is the
  // logs directory's sibling.
  let cache = state_dir.join("prompt-cache");

  let proxy_port = pick_free_port();
  let proxy = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), proxy_port);
  let handle = start_daemon(&state_dir, &scan_root, proxy_port, true).await;
  wait_for_proxy(proxy).await;
  wait_for_model(proxy, "keepme").await;
  let mut client = Client::connect(&state_dir).await.expect("connect");

  // 1) A conversation through the proxy. Nothing is running, so the proxy
  //    auto-starts the model — and finds nothing kept for it, which is the
  //    case that must cost nothing.
  let first = conversation(1200);
  chat(proxy, "keepme", &first).await;
  let (launch_id, port) = ready_launch(&mut client).await;
  let held_before = slot_prompt(port).await;
  assert!(
    held_before > 500,
    "the fixture's slot should hold the conversation that was just asked for"
  );

  // 2) An unload on purpose, through the same `Backend::stop` the idle sweep and
  //    make-room use.
  let stopped = client
    .call(
      "stop_model",
      Some(serde_json::json!({ "launch_id": launch_id, "grace_secs": 1 })),
    )
    .await
    .expect("stop_model");
  assert_eq!(stopped["launch_id"], launch_id.as_str(), "{stopped:?}");
  assert_eq!(stopped["state"]["state"], "stopped", "{stopped:?}");

  // 3) What the unload kept. That the engine wrote a file at all says it was
  //    launched with `--slot-save-path`: without it, the save it was asked to
  //    do answers 400 and nothing is recorded.
  let records = kept_records(&cache);
  assert_eq!(records.len(), 1, "one kept record in {}", cache.display());
  let (record_path, record) = &records[0];
  assert!(
    record_path
      .file_name()
      .and_then(|n| n.to_str())
      .is_some_and(|n| n.ends_with("-prompt-cache.json")),
    "{record_path:?} should be the kept record, not the spent one"
  );
  let saved = record["slots"][0].as_object().expect("one saved slot");
  let tokens = saved["tokens"].as_u64().expect("tokens");
  let file = saved["file"].as_str().expect("file");
  assert!(
    tokens >= held_before,
    "kept {tokens} prompt tokens, the slot held {held_before}"
  );
  assert!(file.starts_with("keepme-"), "{file} should name the model");
  assert!(
    std::fs::metadata(cache.join(file)).is_ok_and(|m| m.len() > 0),
    "{file} should be the state itself, not just a note about it"
  );

  // 4) The conversation comes back. The proxy starts the model again, and hands
  //    it the state before the next request is forwarded.
  chat(proxy, "keepme", "and then what?").await;
  let (second_id, second_port) = ready_launch(&mut client).await;
  assert_ne!(second_id, launch_id, "this is a new process");
  let held_after = slot_prompt(second_port).await;
  assert!(
    held_after >= tokens,
    "the new launch's slot holds {held_after} prompt tokens; the unloaded one had {tokens}"
  );

  // 5) The state is spent, and what remains says so.
  let records = kept_records(&cache);
  assert_eq!(
    records.len(),
    1,
    "the record stays as a record of what happened"
  );
  let (consumed_path, consumed) = &records[0];
  assert!(
    consumed_path
      .file_name()
      .and_then(|n| n.to_str())
      .is_some_and(|n| n.ends_with("-prompt-cache.consumed.json")),
    "{consumed_path:?} should be the consumed record"
  );
  assert!(consumed["restored_at"].is_u64(), "{consumed:?}");
  assert!(
    std::fs::metadata(cache.join(file)).is_err(),
    "the state file is read once and then removed"
  );

  shutdown(handle, &state_dir).await;
  std::fs::remove_dir_all(&state_dir).ok();
  std::fs::remove_dir_all(&scan_root).ok();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn turning_it_off_leaves_the_launch_and_the_disk_as_they_were() {
  std::env::set_var("FAKE_LLAMA_SLOT_SAVE", "1");

  let state_dir = unique_temp("off");
  let scan_root = unique_temp("off-models");
  std::fs::write(scan_root.join("keepme.gguf"), build_minimal_gguf("llama")).unwrap();
  let cache = state_dir.join("prompt-cache");

  let proxy_port = pick_free_port();
  let proxy = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), proxy_port);
  let handle = start_daemon(&state_dir, &scan_root, proxy_port, false).await;
  wait_for_proxy(proxy).await;
  wait_for_model(proxy, "keepme").await;
  let mut client = Client::connect(&state_dir).await.expect("connect");

  chat(proxy, "keepme", &conversation(600)).await;
  let (launch_id, port) = ready_launch(&mut client).await;

  // The engine was never given `--slot-save-path`, which is what its own 400
  // says — the launch argv is the one this feature shipped without.
  assert_eq!(
    try_slot_save(port).await,
    400,
    "a launch with the feature off must not be pointed at a save path"
  );

  let stopped = client
    .call(
      "stop_model",
      Some(serde_json::json!({ "launch_id": launch_id, "grace_secs": 1 })),
    )
    .await
    .expect("stop_model");
  assert_eq!(stopped["launch_id"], launch_id.as_str(), "{stopped:?}");

  assert!(
    !cache.exists() || kept_records(&cache).is_empty(),
    "nothing is kept, and no directory is made for it"
  );

  // The model still comes back and still answers: the feature is an addition,
  // not a dependency.
  chat(proxy, "keepme", "hello again").await;
  let (_, second_port) = ready_launch(&mut client).await;
  let held = slot_prompt(second_port).await;
  assert!(held > 0, "the new launch evaluated the prompt it was given");
  assert!(
    held < 100,
    "a fresh launch holds only its own request ({held}), not a restored prompt"
  );

  shutdown(handle, &state_dir).await;
  std::fs::remove_dir_all(&state_dir).ok();
  std::fs::remove_dir_all(&scan_root).ok();
}
