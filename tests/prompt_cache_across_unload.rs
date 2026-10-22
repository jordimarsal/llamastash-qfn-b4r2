//! A stopped launch's prompt cache survives it (R14).
//!
//! Drives the whole loop against the `fake_llama_server` fixture, which serves
//! one slot and moves "the prompt it holds" through the `--slot-save-path`
//! directory exactly as llama.cpp moves a KV cache: `POST /slots/{id}
//! ?action=save` in one process, `?action=restore` in the next. Nothing here
//! inspects the mechanism — it asks the daemon to stop a model that had a
//! conversation and then to start it again, and checks what the new launch
//! reports.
//!
//! Built only when `--features test-fixtures`.

#![cfg(feature = "test-fixtures")]

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::time::Duration;

use llamastash::config::loader::PortRange;
use llamastash::config::ProxyConfig;
use llamastash::daemon::{run_foreground, DaemonOptions};
use llamastash::gguf::test_fixtures::build_minimal_gguf;
use llamastash::ipc::Client;
use serde_json::{json, Value};
use tokio::time::timeout;

fn fake_binary() -> PathBuf {
  PathBuf::from(env!("CARGO_BIN_EXE_fake_llama_server"))
}

fn unique_temp(label: &str) -> PathBuf {
  llamastash::test_support::unique_temp_dir("ls-pc", label)
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
    tokio::time::sleep(Duration::from_millis(20)).await;
  }
}

/// A launch is started and waited for through `status`, the same way the CLI
/// starts one. Returns `(launch_id, port)`.
async fn start_ready(client: &mut Client, model_path: &Path) -> (String, u16) {
  let body = client
    .call(
      "start_model",
      Some(json!({ "model_path": model_path.to_string_lossy(), "mode": "chat" })),
    )
    .await
    .expect("start_model");
  let launch_id = body["launch_id"].as_str().expect("launch_id").to_string();
  let port = body["port"].as_u64().expect("port") as u16;

  let deadline = std::time::Instant::now() + Duration::from_secs(30);
  loop {
    let status = client.call("status", None).await.expect("status");
    if status["models"]
      .as_array()
      .expect("models")
      .iter()
      .any(|m| m["launch_id"] == json!(&launch_id) && m["state"]["state"] == json!("ready"))
    {
      return (launch_id, port);
    }
    if std::time::Instant::now() > deadline {
      panic!("launch {launch_id} never reached ready");
    }
    tokio::time::sleep(Duration::from_millis(40)).await;
  }
}

/// The fixture's own slot list: what the engine says it is holding. Read from the
/// launched child rather than from anything llamastash reports, so the assertion
/// is about engine state and not about our own bookkeeping.
fn resident_tokens(port: u16) -> Vec<i64> {
  let body = rpc_get(port, "/slots");
  body
    .as_array()
    .unwrap_or_else(|| panic!("GET /slots on {port} is not a list: {body}"))
    .iter()
    .map(|slot| {
      slot
        .get("n_prompt_tokens")
        .and_then(Value::as_i64)
        .unwrap_or(-1)
    })
    .collect()
}

fn serve_prompt(port: u16) {
  let body = json!({
    "model": "fixture",
    "messages": [{ "role": "user", "content": "a prompt long enough to be resident" }],
  })
  .to_string();
  let response = rpc_request("POST", "/v1/chat/completions", port, Some(body.as_str()));
  assert!(
    response.starts_with("HTTP/1.1 200"),
    "the fixture should answer a chat request: {response}"
  );
}

/// Plain HTTP/1.1 against the launched child. Deliberately blocking and
/// deliberately not shared with llamastash: this is the observer's instrument.
fn rpc_get(port: u16, path: &str) -> Value {
  let response = rpc_request("GET", path, port, None);
  let (_, body) = response
    .split_once("\r\n\r\n")
    .unwrap_or(("", response.as_str()));
  serde_json::from_str(body).unwrap_or_else(|e| panic!("GET {path} on {port}: {e}\n{response}"))
}

fn rpc_request(method: &str, path: &str, port: u16, body: Option<&str>) -> String {
  let mut sock = TcpStream::connect(("127.0.0.1", port)).expect("connect child");
  sock
    .set_read_timeout(Some(Duration::from_secs(10)))
    .expect("read timeout");
  let head = match body {
    Some(body) => format!(
      "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
      body.len()
    ),
    None => format!(
      "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n"
    ),
  };
  sock.write_all(head.as_bytes()).expect("write request");
  if let Some(body) = body {
    sock.write_all(body.as_bytes()).expect("write body");
  }
  sock.flush().expect("flush");
  let mut out = String::new();
  sock.read_to_string(&mut out).expect("read response");
  out
}

/// The files the daemon wrote, as `llamastash stop` left them.
fn checkpoint_dir(state: &Path) -> PathBuf {
  state.join("prompt-cache")
}

fn checkpoint_names(state: &Path) -> Vec<String> {
  let dir = checkpoint_dir(state);
  let mut names: Vec<String> = std::fs::read_dir(&dir)
    .map(|entries| {
      entries
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| entry.file_name().to_str().map(str::to_string))
        .collect()
    })
    .unwrap_or_default();
  names.sort();
  names
}

struct Scenario {
  state: PathBuf,
  model_path: PathBuf,
  socket: PathBuf,
  daemon: tokio::task::JoinHandle<()>,
}

async fn scenario(label: &str) -> Scenario {
  let state = unique_temp(label);
  let model_dir = unique_temp(&format!("{label}-model"));
  let model_path = model_dir.join("m.gguf");
  std::fs::write(&model_path, build_minimal_gguf("llama")).unwrap();

  let opts = DaemonOptions {
    binary: Some(fake_binary()),
    port_range: allocate_port_range(),
    proxy: ProxyConfig {
      enabled: false,
      ..ProxyConfig::default()
    },
    ..DaemonOptions::rooted_at(state.clone())
  };
  let socket = opts.state_dir.clone();
  let daemon = tokio::spawn(async move {
    let _ = run_foreground(opts).await;
  });
  wait_for_socket(&socket).await;
  Scenario {
    state,
    model_path,
    socket,
    daemon,
  }
}

async fn teardown(client: &mut Client, scenario: &mut Scenario) {
  let _ = client.call("shutdown", None).await;
  let _ = timeout(Duration::from_secs(10), &mut scenario.daemon).await;
  let _ = std::fs::remove_dir_all(&scenario.state);
  let _ = std::fs::remove_dir_all(scenario.model_path.parent().expect("model dir"));
}

/// converse → stop → a file; start again → the new launch holds the prompt.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_stopped_conversation_is_resumed_by_the_next_launch() {
  let mut scenario = scenario("resume").await;
  let mut client = Client::connect(&scenario.socket).await.expect("connect");

  let (launch_id, port) = start_ready(&mut client, &scenario.model_path).await;
  serve_prompt(port);
  assert_eq!(
    resident_tokens(port),
    vec![512],
    "the fixture should report the prompt it served"
  );

  let stopped = client
    .call(
      "stop_model",
      Some(json!({ "launch_id": &launch_id, "grace_secs": 5 })),
    )
    .await
    .expect("stop_model");
  assert_eq!(stopped["state"]["state"], json!("stopped"));

  // The child is really gone before anything is read from the directory: the
  // daemon hands the same port to the next launch, so "this port answers" is not
  // what distinguishes the two.
  let gone = std::time::Instant::now() + Duration::from_secs(10);
  loop {
    if std::net::TcpStream::connect(("127.0.0.1", port)).is_err() {
      break;
    }
    if std::time::Instant::now() > gone {
      panic!("the stopped launch still answers on {port}");
    }
    tokio::time::sleep(Duration::from_millis(50)).await;
  }

  let names = checkpoint_names(&scenario.state);
  assert_eq!(
    names.len(),
    1,
    "one checkpoint for one resident slot: {names:?}"
  );
  let file = names[0].as_str();
  assert!(
    file.len() == 16 + "-slot0.bin".len() && file.ends_with("-slot0.bin"),
    "the checkpoint is named for the model and the slot it came from: {file}"
  );
  let written = std::fs::read_to_string(checkpoint_dir(&scenario.state).join(file)).expect("read");
  assert!(
    written.contains("tokens 512"),
    "the engine wrote the state it was holding: {written}"
  );

  // The whole point: a launch that does not exist yet picks that state up. It
  // arrives in the same slot the checkpoint was taken from, which is what makes
  // a file meaningful across a process boundary at all.
  let (second_id, second_port) = start_ready(&mut client, &scenario.model_path).await;
  assert_ne!(launch_id, second_id, "the second launch is a new one");
  assert_eq!(
    resident_tokens(second_port),
    vec![512],
    "the second launch should be holding the first one's prompt"
  );

  // Consumed, not kept: the engine has it now, and a second copy of the same
  // prompt in the directory would only spend the cap.
  let after = checkpoint_names(&scenario.state);
  assert!(
    !after.contains(&file.to_string()),
    "the checkpoint that was restored should be gone: {after:?}"
  );

  let stopped = client
    .call(
      "stop_model",
      Some(json!({ "launch_id": &second_id, "grace_secs": 5 })),
    )
    .await
    .expect("stop_model");
  assert_eq!(stopped["state"]["state"], json!("stopped"));
  teardown(&mut client, &mut scenario).await;
}

/// A launch that never held a prompt has nothing to write, and the feature stays
/// quiet rather than leaving an empty checkpoint behind.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_launch_that_never_served_a_prompt_writes_nothing() {
  let mut scenario = scenario("silent").await;
  let mut client = Client::connect(&scenario.socket).await.expect("connect");

  let (launch_id, port) = start_ready(&mut client, &scenario.model_path).await;
  assert_eq!(
    resident_tokens(port),
    vec![0],
    "an untouched slot holds nothing"
  );

  let stopped = client
    .call(
      "stop_model",
      Some(json!({ "launch_id": &launch_id, "grace_secs": 5 })),
    )
    .await
    .expect("stop_model");
  assert_eq!(stopped["state"]["state"], json!("stopped"));

  assert!(
    checkpoint_names(&scenario.state).is_empty(),
    "nothing resident means nothing checkpointed: {:?}",
    checkpoint_dir(&scenario.state)
  );
  teardown(&mut client, &mut scenario).await;
}
