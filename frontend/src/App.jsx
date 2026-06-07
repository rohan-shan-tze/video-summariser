import { useState, useRef, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import "./App.css";

// Stable session ID for this app run. In Phase 8 this will be selectable
// from persisted sessions; for now one session per launch is sufficient.
const SESSION_ID = `session-${Date.now()}`;

function App() {
  const [messages, setMessages]     = useState([]);
  const [input, setInput]           = useState("");
  const [videoPath, setVideoPath]   = useState("");
  const [loading, setLoading]       = useState(false);
  const [error, setError]           = useState("");
  const bottomRef                   = useRef(null);

  // Scroll to bottom whenever messages change.
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

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
        sessionId: SESSION_ID,
        text,
        videoPath: videoPath || "",
      });

      appendMessage("assistant", resp.reply, resp.artifactPath || "");

      // If the backend wants clarification, render the options as buttons
      // in a special "options" message.
      if (resp.needsClarification && resp.options.length > 0) {
        appendMessage("options", "", "", resp.options);
      }
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

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <span className="header-title">[ VIDEO SUMMARISER ]</span>
        <button className="pick-btn" onClick={pickVideo}>
          {videoPath ? `[ ${videoPath.split(/[\\/]/).pop()} ]` : "[ PICK VIDEO ]"}
        </button>
      </header>

      {/* Message list */}
      <main className="chat-area">
        {messages.length === 0 && (
          <div className="empty-state">
            <p>Pick a video, then ask me anything.</p>
            <p className="hint">
              Try: "transcribe the video" / "what objects appear?" /
              "are there graphs?" / "summarize" / "make a PDF" / "make a PowerPoint"
            </p>
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

      {/* Input bar */}
      <footer className="input-bar">
        <textarea
          className="chat-input"
          rows={2}
          value={input}
          onChange={(e) => setInput(e.target.value)}
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
        <div className="artifact-notice">
          FILE READY: {msg.artifactPath.split(/[\\/]/).pop()}
        </div>
      )}
    </div>
  );
}

export default App;
