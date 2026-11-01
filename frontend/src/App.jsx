import React, { useEffect, useRef, useState } from "react";
import {
  ArrowUp, Bot, BrainCircuit, Check, ChevronDown, Command, Copy,
  Github, Menu, Moon, Plus, Sparkles, Sun, Terminal, User, Zap
} from "lucide-react";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const suggestions = [
  "What are Amit's career goals?",
  "Which backend technologies does Amit use?",
  "Tell me about the AI interview platform.",
  "What is Amit currently learning?"
];

const welcome = {
  role: "assistant",
  content: "Hey! I'm Amit's profile assistant. Ask me about his backend experience, projects, technical skills or career goals. I'll use the available profile knowledge to answer.",
  sources: []
};

function Orb({ thinking }) {
  return (
    <div className={`orb-scene ${thinking ? "is-thinking" : ""}`} aria-label="Animated AI orb">
      <div className="orb-grid" />
      <div className="orb-halo halo-one" />
      <div className="orb-halo halo-two" />
      <div className="orb-core">
        <div className="orb-gloss" />
        <div className="orb-inner-ring" />
        <BrainCircuit size={31} strokeWidth={1.35} />
      </div>
      <span className="orb-spark spark-a" />
      <span className="orb-spark spark-b" />
      <span className="orb-spark spark-c" />
    </div>
  );
}

function Message({ message, index }) {
  const [copied, setCopied] = useState(false);
  const isUser = message.role === "user";

  async function copyAnswer() {
    await navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 1400);
  }

  return (
    <article
      className={`message-row ${isUser ? "user-row" : "assistant-row"}`}
    >
      <div className={`message-avatar ${isUser ? "user-avatar" : "ai-avatar"}`}>
        {isUser ? <User size={16} /> : <Sparkles size={16} />}
      </div>
      <div className="message-content">
        <div className="message-meta">
          <span>{isUser ? "You" : "AMIT AI"}</span>
          {!isUser && <span className="model-tag"><span /> QWEN · LOCAL</span>}
        </div>
        <div className="message-bubble">{message.content}</div>
        {!isUser && message.sources?.length > 0 && (
          <details className="source-details">
            <summary><ChevronDown size={13} /> Retrieved context ({message.sources.length})</summary>
            <div className="source-list">
              {message.sources.map((source, i) => (
                <div className="source-item" key={i}>
                  <span className="source-number">0{i + 1}</span>
                  <p>{typeof source === "string" ? source : JSON.stringify(source)}</p>
                </div>
              ))}
            </div>
          </details>
        )}
        {!isUser && (
          <button className="copy-button" onClick={copyAnswer}>
            {copied ? <Check size={13} /> : <Copy size={13} />}
            {copied ? "Copied" : "Copy answer"}
          </button>
        )}
      </div>
    </article>
  );
}

export default function App() {
  const [messages, setMessages] = useState([welcome]);
  const [input, setInput] = useState("");
  const [thinking, setThinking] = useState(false);
  const [dark, setDark] = useState(false);
  const [mobileMenu, setMobileMenu] = useState(false);
  const bottomRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, thinking]);

  async function sendMessage(text = input) {
    const question = text.trim();
    if (!question || thinking) return;
    setInput("");
    setMessages(prev => [...prev, { role: "user", content: question }]);
    setThinking(true);

    try {
      const response = await fetch(`${API_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: question })
      });
      if (!response.ok) {
        const detail = await response.text();
        throw new Error(detail || `Request failed (${response.status})`);
      }
      const data = await response.json();
      setMessages(prev => [...prev, {
        role: "assistant",
        content: data.answer || data.llm_response || "The backend returned an empty answer.",
        sources: data.sources || data.retrieved_documents || []
      }]);
    } catch (error) {
      setMessages(prev => [...prev, {
        role: "assistant",
        content: `I couldn't reach the chatbot API. ${error.message}. Check that FastAPI is running and VITE_API_URL points to it.`,
        sources: []
      }]);
    } finally {
      setThinking(false);
      inputRef.current?.focus();
    }
  }

  function newChat() {
    setMessages([welcome]);
    setInput("");
    setMobileMenu(false);
  }

  return (
    <div className={`app-shell ${dark ? "theme-dark" : "theme-light"}`}>
      <div className="ambient ambient-a" />
      <div className="ambient ambient-b" />
      <header className="topbar">
        <a className="brand" href="#">
          <span className="brand-mark"><Terminal size={17} /></span>
          <span>amit<span className="brand-dot">.</span>ai</span>
          <span className="brand-divider" />
          <span className="brand-caption">PERSONAL KNOWLEDGE ENGINE</span>
        </a>
        <div className="topbar-actions">
          <div className="system-status"><span className="status-pulse" /> SYSTEM ONLINE</div>
          <button className="icon-button theme-toggle" onClick={() => setDark(v => !v)} aria-label="Toggle theme">
            {dark ? <Sun size={17} /> : <Moon size={17} />}
          </button>
          <button className="icon-button mobile-menu-button" onClick={() => setMobileMenu(v => !v)} aria-label="Open menu" aria-expanded={mobileMenu}><Menu size={18} /></button>
          <a className="portfolio-link" href="https://amit-portfolio-virid-eight.vercel.app/" target="_blank" rel="noreferrer">PORTFOLIO <span>↗</span></a>
        </div>
      </header>

      <main className="main-layout">
        <button
          className={`sidebar-scrim ${mobileMenu ? "scrim-visible" : ""}`}
          onClick={() => setMobileMenu(false)}
          aria-label="Close menu"
          tabIndex={mobileMenu ? 0 : -1}
        />
        <aside className={`sidebar ${mobileMenu ? "sidebar-open" : ""}`}>
          <button className="new-chat-button" onClick={newChat}><Plus size={16} /> New conversation <span>⌘ K</span></button>
          <div className="side-label">QUICK PROMPTS</div>
          <div className="prompt-list">
            {suggestions.map((item, i) => (
              <button key={item} className="prompt-link" onClick={() => { sendMessage(item); setMobileMenu(false); }}>
                <span className="prompt-index">0{i + 1}</span>{item}
              </button>
            ))}
          </div>
          <div className="sidebar-bottom">
            <div className="knowledge-card">
              <div className="knowledge-icon"><BrainCircuit size={17} /></div>
              <div><strong>Knowledge base</strong><span>Personal profile · 19 vectors</span></div>
              <span className="knowledge-live" />
            </div>
            <div className="sidebar-footer"><span>POWERED BY</span><strong>OLLAMA <i>×</i> LANGCHAIN</strong></div>
          </div>
        </aside>

        <section className="workspace">
          <div className="workspace-heading">
            <div>
              <div className="eyebrow"><span className="eyebrow-line" /> PRIVATE AI ASSISTANT</div>
              <h1>Ask <span>anything.</span></h1>
              <p className="heading-subtitle">A smarter interface to Amit's professional world.</p>
            </div>
            <div className="session-chip"><span className="session-dot" /> LOCAL SESSION <span className="session-separator">/</span> QWEN 0.6B</div>
          </div>

          <div className="hero-orb-wrap">
            <div className="orb-stage">
              <span className="stage-glow glow-lilac" />
              <span className="stage-glow glow-peach" />
              <div className="orbit-card orbit-card-python"><span>⌘</span> PYTHON</div>
              <div className="orbit-card orbit-card-rag"><span>✦</span> RAG READY</div>
              <div className="orbit-card orbit-card-local"><i /> LOCAL AI</div>
              <Orb thinking={thinking} />
            </div>
            <div className="orb-caption">
              <span>{thinking ? "PROCESSING YOUR QUERY" : "READY WHEN YOU ARE"}</span>
              <div className="orb-caption-line" />
              <small>{thinking ? "Retrieving context · generating response" : "Ask a question to start exploring"}</small>
            </div>
          </div>

          <div className="chat-panel">
            <div className="chat-panel-top">
              <div className="chat-title"><span className="chat-live-dot" /> CONVERSATION <span className="chat-count">{String(Math.max(0, messages.length - 1)).padStart(2, "0")}</span></div>
              <button className="clear-button" onClick={newChat}>CLEAR CHAT <span>↗</span></button>
            </div>
            <div className="messages-area">
              {messages.map((message, index) => <Message key={`${index}-${message.role}`} message={message} index={index} />)}
              {thinking && (
                <div className="thinking-row">
                  <div className="message-avatar ai-avatar"><Sparkles size={16} /></div>
                  <div><div className="message-meta">AMIT AI <span className="model-tag"><span /> THINKING</span></div><div className="thinking-bubble"><i /><i /><i /></div></div>
                </div>
              )}
              <div ref={bottomRef} />
            </div>
            <form className="composer" onSubmit={e => { e.preventDefault(); sendMessage(); }}>
              <div className="composer-topline"><span><Command size={12} /> ASK YOUR QUESTION</span><span>ENTER TO SEND · SHIFT + ENTER FOR NEW LINE</span></div>
              <div className="composer-input-row">
                <input
                  ref={inputRef}
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); } }}
                  placeholder="Ask about skills, experience, projects..."
                  disabled={thinking}
                />
                <button className="send-button" type="submit" disabled={!input.trim() || thinking}><ArrowUp size={19} /></button>
              </div>
            </form>
          </div>
          <footer className="workspace-footer"><span><Zap size={12} /> RUNNING LOCALLY</span><span>YOUR DATA · YOUR CONTROL</span><span>BUILT WITH CURIOSITY <b>✳</b></span></footer>
        </section>
      </main>
    </div>
  );
}
