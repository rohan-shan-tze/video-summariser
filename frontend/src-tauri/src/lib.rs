// Include tonic-generated gRPC stubs from the proto build step.
// tonic_build compiles video.proto into video.rs and places it in OUT_DIR.
// The include_proto! macro splices that generated code in here at compile time.
pub mod video_proto {
    tonic::include_proto!("video");
}

use video_proto::video_service_client::VideoServiceClient;
use video_proto::ChatRequest;

use serde::{Deserialize, Serialize};
use tauri_plugin_dialog::DialogExt;

// The Python gRPC server always binds to this address.
const GRPC_ADDR: &str = "http://127.0.0.1:50051";

// ChatResponse mirrors the proto ChatResponse fields.
// Serde makes it serializable so Tauri can send it to the React webview as JSON.
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatResponse {
    pub reply: String,
    pub needs_clarification: bool,
    pub options: Vec<String>,
    pub artifact_path: String,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HistoryMessage {
    pub role: String,
    pub text: String,
    pub artifact_path: String,
    pub timestamp: String,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionInfo {
    pub session_id: String,
    pub created_at: String,
    pub video_path: String,
}

// send_message is a Tauri command: React calls invoke("send_message", {...})
// and gets back a ChatResponse (or an error string).
//
// Why handle gRPC in Rust rather than in the webview directly?
//   - Browsers/webviews cannot speak raw HTTP/2 gRPC (they only speak HTTP/1.1
//     or gRPC-Web which needs a proxy). The Rust layer has no such restriction.
//   - Keeps the gRPC dependency out of the npm bundle.
//   - The Rust process already runs with full OS permissions.
#[tauri::command]
async fn send_message(
    session_id: String,
    text: String,
    video_path: String,
) -> Result<ChatResponse, String> {
    // Connect to the Python gRPC server. connect() is cheap - tonic uses
    // a lazy connection pool so this doesn't block if the server is busy.
    let mut client = VideoServiceClient::connect(GRPC_ADDR)
        .await
        .map_err(|e| format!("Could not connect to backend: {e}. Is the Python gRPC server running?"))?;

    let request = tonic::Request::new(ChatRequest {
        session_id,
        text,
        video_path,
    });

    let response = client
        .send_message(request)
        .await
        .map_err(|e| format!("gRPC error: {e}"))?
        .into_inner();

    Ok(ChatResponse {
        reply: response.reply,
        needs_clarification: response.needs_clarification,
        options: response.options,
        artifact_path: response.artifact_path,
    })
}

// get_history fetches all persisted messages for a session from the Python backend.
// Called on app startup so prior conversations can be re-rendered in the UI.
#[tauri::command]
async fn get_history(session_id: String) -> Result<Vec<HistoryMessage>, String> {
    use video_proto::HistoryRequest;

    let mut client = VideoServiceClient::connect(GRPC_ADDR)
        .await
        .map_err(|e| format!("Could not connect to backend: {e}"))?;

    let request = tonic::Request::new(HistoryRequest { session_id });
    let response = client
        .get_history(request)
        .await
        .map_err(|e| format!("gRPC error: {e}"))?
        .into_inner();

    Ok(response
        .messages
        .into_iter()
        .map(|m| HistoryMessage {
            role: m.role,
            text: m.text,
            artifact_path: m.artifact_path,
            timestamp: m.timestamp,
        })
        .collect())
}

// list_sessions returns all past sessions ordered by creation time (newest first).
// Used to populate a session picker so the user can resume a prior conversation.
#[tauri::command]
async fn list_sessions() -> Result<Vec<SessionInfo>, String> {
    use video_proto::ListSessionsRequest;

    let mut client = VideoServiceClient::connect(GRPC_ADDR)
        .await
        .map_err(|e| format!("Could not connect to backend: {e}"))?;

    let request = tonic::Request::new(ListSessionsRequest {});
    let response = client
        .list_sessions(request)
        .await
        .map_err(|e| format!("gRPC error: {e}"))?
        .into_inner();

    Ok(response
        .sessions
        .into_iter()
        .map(|s| SessionInfo {
            session_id: s.session_id,
            created_at: s.created_at,
            video_path: s.video_path,
        })
        .collect())
}

// open_file_dialog opens a native OS file picker filtered to .mp4 files.
// Returns the selected path as a string, or an empty string if cancelled.
#[tauri::command]
async fn open_file_dialog(app: tauri::AppHandle) -> String {
    let file = app
        .dialog()
        .file()
        .add_filter("Video", &["mp4"])
        .blocking_pick_file();

    match file {
        Some(path) => path.to_string(),
        None => String::new(),
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![send_message, open_file_dialog, get_history, list_sessions])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
