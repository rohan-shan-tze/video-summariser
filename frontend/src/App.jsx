import { useState, useRef, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { openPath } from "@tauri-apps/plugin-opener";
import "./App.css";

function newSessionId() {
  return `session-${Date.now()}`;
}

function App() {
  const [sessionId, setSessionId]   = useState(newSessionId());
  const [sessions, setSessions]     = useState([]);   // [{sessionId, createdAt}]
  const [messages, setMessages]     = useState([]);
  const [input, setInput]           = useState("");
  const [videoPath, setVideoPath]   = useState("");
  const [loading, setLoading]       = useState(false);
  const [error, setError]           = useState("");
  const [showSessions, setShowSessions] = useState(false);
  const [showGuide, setShowGuide]       = useState(false);
  const bottomRef                   = useRef(null);

  // On mount: fetch the session list so the user can resume a prior conversation.
  useEffect(() => {
    invoke("list_sessions")
      .then((rows) => setSessions(rows))
      .catch(() => {}); // backend may not be running yet; silently ignore
  }, []);

  // Escape key closes the guide modal.
  useEffect(() => {
    if (!showGuide) return;
    function onKey(e) { if (e.key === "Escape") setShowGuide(false); }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [showGuide]);

  // Scroll to bottom whenever messages change.
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Load history for the given session and replace the current message list.
  // Also restores the video path if the file still exists on disk (checked via
  // the presence of a non-empty path — Tauri has no cheap fs.exists, so we
  // optimistically set it and let the backend reject a missing file if queried).
  async function loadSession(sid, savedVideoPath) {
    setSessionId(sid);
    setMessages([]);
    setError("");
    setShowSessions(false);

    // Restore video path if one was saved; clear it otherwise.
    if (savedVideoPath) {
      setVideoPath(savedVideoPath);
    } else {
      setVideoPath("");
    }

    try {
      const rows = await invoke("get_history", { sessionId: sid });
      const loaded = rows.map((row, i) => ({
        id: i,
        role: row.role,
        text: row.text,
        artifactPath: row.artifactPath || "",
        options: [],
      }));
      setMessages(loaded);
    } catch (e) {
      setError(`Could not load history: ${e}`);
    }
  }

  // Start a brand-new session (no history).
  function startNewSession() {
    const sid = newSessionId();
    setSessionId(sid);
    setMessages([]);
    setVideoPath("");
    setError("");
    setShowSessions(false);
  }

  // Open native file picker filtered to .mp4
  async function pickVideo() {
    const path = await invoke("open_file_dialog");
    if (path) {
      setVideoPath(path);
      setError("");
      appendMessage("system", `Video selected: ${path.split(/[\\/]/).pop()}`);
    }
  }

  // Send a text message (or resolve a clarification option)
  async function sendMessage(text) {
    if (!text.trim()) return;
    setInput("");
    setError("");
    appendMessage("user", text);
    setLoading(true);

    try {
      const resp = await invoke("send_message", {
        sessionId,
        text,
        videoPath: videoPath || "",
      });

      appendMessage("assistant", resp.reply, resp.artifactPath || "");

      // If the backend wants clarification, render the options as buttons.
      if (resp.needsClarification && resp.options.length > 0) {
        appendMessage("options", "", "", resp.options);
      }

      // Refresh session list so the new session appears if this is the first turn.
      invoke("list_sessions")
        .then((rows) => setSessions(rows))
        .catch(() => {});
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  function appendMessage(role, text, artifactPath = "", options = []) {
    setMessages((prev) => [
      ...prev,
      { id: Date.now() + Math.random(), role, text, artifactPath, options },
    ]);
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  }

  // Format ISO timestamp to a short human-readable string.
  function fmtDate(iso) {
    try {
      return new Date(iso).toLocaleString(undefined, {
        month: "short", day: "numeric",
        hour: "2-digit", minute: "2-digit",
      });
    } catch {
      return iso;
    }
  }

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <span className="header-title">[ VIDEO SUMMARISER ]</span>
        <div className="header-right">
          <button className="pick-btn" onClick={() => setShowGuide(true)}>
            [ GUIDE ]
          </button>
          <button className="pick-btn" onClick={() => setShowSessions((v) => !v)}>
            [ SESSIONS ]
          </button>
          <button className="pick-btn" onClick={pickVideo}>
            {videoPath ? `[ ${videoPath.split(/[\\/]/).pop()} ]` : "[ PICK VIDEO ]"}
          </button>
        </div>
      </header>

      {/* Session picker dropdown */}
      {showSessions && (
        <div className="session-backdrop" onClick={() => setShowSessions(false)} />
      )}
      {showSessions && (
        <div className="session-panel">
          <button className="session-new-btn" onClick={startNewSession}>
            + New session
          </button>
          {sessions.length === 0 && (
            <div className="session-empty">No saved sessions yet.</div>
          )}
          {sessions.map((s) => (
            <button
              key={s.sessionId}
              className={`session-item ${s.sessionId === sessionId ? "active" : ""}`}
              onClick={() => loadSession(s.sessionId, s.videoPath)}
            >
              <span className="session-date">{fmtDate(s.createdAt)}</span>
              {s.videoPath && (
                <span className="session-id">{s.videoPath.split(/[\\/]/).pop()}</span>
              )}
            </button>
          ))}
        </div>
      )}

      {/* Message list */}
      <main className="chat-area">
        {messages.length === 0 && (
          <div className="empty-state">
            <p className="hint-primary">Pick a video, then ask a question.</p>
            <p className="hint">
              Try: "transcribe the video" / "what objects appear?" /
              "are there graphs?" / "summarize" / "extractive summary" /
              "make a PDF" / "make a PowerPoint"
            </p>
            <p className="hint">Press [ GUIDE ] for a full list of commands.</p>
          </div>
        )}

        {messages.map((msg) => (
          <MessageBubble key={msg.id} msg={msg} onOption={sendMessage} />
        ))}

        {loading && (
          <div className="bubble assistant loading">
            <span className="dot-flash">...</span>
          </div>
        )}

        {error && (
          <div className="bubble error">
            ERROR: {error}
          </div>
        )}

        <div ref={bottomRef} />
      </main>

      {/* Guide modal */}
      {showGuide && (
        <div className="guide-overlay" onClick={() => setShowGuide(false)}>
          <div className="guide-modal" onClick={(e) => e.stopPropagation()}>
            <div className="guide-header">
              <span>[ USER GUIDE ]</span>
              <button className="guide-close" onClick={() => setShowGuide(false)}>[ X ]</button>
            </div>
            <div className="guide-body">

              <section className="guide-section">
                <h3>TRANSCRIPTION</h3>
                <p>Converts all speech in the video to text.</p>
                <div className="guide-examples">
                  <code>Transcribe the video</code>
                  <code>What was said?</code>
                  <code>Give me the transcript</code>
                </div>
              </section>

              <section className="guide-section">
                <h3>OBJECT DETECTION</h3>
                <p>Samples frames and identifies objects using YOLOv8n.</p>
                <div className="guide-examples">
                  <code>What objects appear in the video?</code>
                  <code>What can you see?</code>
                  <code>Detect objects</code>
                </div>
              </section>

              <section className="guide-section">
                <h3>TEXT / GRAPH DETECTION</h3>
                <p>Reads on-screen text via OCR and detects charts or graphs.</p>
                <div className="guide-examples">
                  <code>Are there any graphs or charts?</code>
                  <code>What text appears on screen?</code>
                  <code>Extract text from the video</code>
                </div>
              </section>

              <section className="guide-section">
                <h3>SUMMARIZATION</h3>
                <p>Two backends are available:</p>
                <table className="guide-table">
                  <thead>
                    <tr><th>Backend</th><th>Trigger keywords</th><th>Notes</th></tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>LLM (Llama 3.2 1B)</td>
                      <td><code>summarize</code>, <code>summary</code>, <code>brief</code>, <code>detailed summary</code></td>
                      <td>Default. Natural synthesised output. First call ~20s to load model.</td>
                    </tr>
                    <tr>
                      <td>Extractive (LexRank)</td>
                      <td><code>extractive summary</code>, <code>extractive</code></td>
                      <td>Instant. Selects real sentences verbatim - no generation.</td>
                    </tr>
                  </tbody>
                </table>
                <div className="guide-examples">
                  <code>Summarize the video</code>
                  <code>Give me a detailed summary</code>
                  <code>Extractive summary</code>
                  <code>Brief extractive summary</code>
                </div>
              </section>

              <section className="guide-section">
                <h3>REPORT GENERATION</h3>
                <p>Generates a PDF or PowerPoint. Transcribes and summarizes automatically if not done yet.</p>
                <div className="guide-examples">
                  <code>Make a PDF report</code>
                  <code>Generate a PowerPoint</code>
                  <code>Create a PPTX with the key points</code>
                  <code>Summarize the video and generate a PDF</code>
                </div>
                <p className="guide-note">Generated files appear as FILE READY links in the chat. Click to open.</p>
              </section>

              <section className="guide-section">
                <h3>TIPS</h3>
                <ul className="guide-tips">
                  <li>Results are cached per session - asking for a summary after a transcript reuses the transcript without re-processing.</li>
                  <li>If the system asks for clarification, pick one of the options shown or rephrase your query.</li>
                  <li>Switch sessions from [ SESSIONS ] to resume a past conversation with its original video.</li>
                  <li>Pick a video first before querying - most intents require one.</li>
                </ul>
              </section>

            </div>
          </div>
        </div>
      )}

      {/* Input bar */}
      <footer className="input-bar">
        <textarea
          className="chat-input"
          rows={2}
          value={input}
          onChange={(e) => { setInput(e.target.value); if (error) setError(""); }}
          onKeyDown={handleKeyDown}
          placeholder="Type a message... (Enter to send, Shift+Enter for newline)"
          disabled={loading}
        />
        <button
          className="send-btn"
          onClick={() => sendMessage(input)}
          disabled={loading || !input.trim()}
        >
          SEND
        </button>
      </footer>
    </div>
  );
}

function MessageBubble({ msg, onOption }) {
  if (msg.role === "options") {
    return (
      <div className="options-row">
        {msg.options.map((opt) => (
          <button key={opt} className="option-btn" onClick={() => onOption(opt)}>
            {opt}
          </button>
        ))}
      </div>
    );
  }

  return (
    <div className={`bubble ${msg.role}`}>
      {msg.role === "user" && <span className="role-label">&gt; YOU</span>}
      {msg.role === "assistant" && <span className="role-label">* SYS</span>}
      {msg.role === "system" && <span className="role-label">-- INFO</span>}
      <pre className="bubble-text">{msg.text}</pre>
      {msg.artifactPath && (
        <div
          className="artifact-notice clickable"
          onClick={() => openPath(msg.artifactPath)}
          title={msg.artifactPath}
        >
          FILE READY: {msg.artifactPath.split(/[\\/]/).pop()} [click to open]
        </div>
      )}
    </div>
  );
}

export default App;
