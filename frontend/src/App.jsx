import React, { useState, useRef, useEffect } from 'react';

const API_BASE_URL = 'http://127.0.0.1:8000';

const EXAMPLE_QUERIES = [
  {
    icon: '📄',
    label: 'Ask about the document',
    query: 'What is the Week 2 development plan?'
  },
  {
    icon: '🗄️',
    label: 'Ask a database question',
    query: 'What was the stock price in 2024?'
  },
  {
    icon: '👁️',
    label: 'Ask about an image',
    query: 'What does this bar chart show?'
  },
  {
    icon: '🔄',
    label: 'Test Self-RAG',
    query: 'What does the document say about revenue?'
  }
];

// Subcomponent: Collapsible Source Citation Item
function SourceItem({ source }) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className="source-item">
      <div
        className="source-item-header"
        onClick={() => setIsExpanded(!isExpanded)}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            setIsExpanded(!isExpanded);
          }
        }}
      >
        <div className="source-meta">
          <span className="source-doc-icon">📄</span>
          <span className="source-doc-name">sample.pdf</span>
          <span className="source-divider">•</span>
          <span className="source-page-badge">Page {source.page}</span>
          <span className="source-divider">•</span>
          <span className="source-chunk-badge">Chunk {source.chunk_id}</span>
        </div>
        <button
          className="source-toggle-btn"
          type="button"
          aria-label="Toggle excerpt view"
        >
          {isExpanded ? 'Hide excerpt ▲' : 'View source ▼'}
        </button>
      </div>

      {isExpanded && (
        <div className="source-text-preview">
          <div className="preview-label">Extracted Context Excerpt:</div>
          <p className="preview-content">{source.text}</p>
        </div>
      )}
    </div>
  );
}

// Subcomponent: Self-RAG Execution Pipeline Card
function SelfRagCard({ selfRag }) {
  if (!selfRag || !selfRag.enabled) return null;

  const { attempts, relevance_found, query_rewritten } = selfRag;

  return (
    <div className="self-rag-card">
      <div className="self-rag-header">
        <div className="self-rag-header-left">
          <span className="self-rag-icon">🔄</span>
          <span className="self-rag-title">Self-RAG Process Status</span>
        </div>
        <span className="self-rag-status-badge">Completed</span>
      </div>

      <div className="self-rag-steps">
        {/* Step 1: Initial Search */}
        <div className="step-item step-completed">
          <div className="step-marker">✓</div>
          <div className="step-content">
            <div className="step-label">Initial Search (Attempt 1)</div>
            <div className="step-desc">Retrieved candidate document chunks from index</div>
          </div>
        </div>

        {/* Step 2: Relevance Check 1 */}
        <div className="step-item step-completed">
          <div className="step-marker">✓</div>
          <div className="step-content">
            <div className="step-label">Relevance Assessment</div>
            <div className="step-desc">
              {query_rewritten
                ? 'Retrieved chunks evaluated as irrelevant or insufficient'
                : 'Relevant chunks identified with high confidence'}
            </div>
          </div>
        </div>

        {/* If Query Rewriting occurred */}
        {query_rewritten && (
          <>
            <div className="step-item step-warning">
              <div className="step-marker">⚠</div>
              <div className="step-content">
                <div className="step-label">Query Rewritten</div>
                <div className="step-desc">Autonomous reformulator expanded query for precision</div>
              </div>
            </div>

            <div className="step-item step-completed">
              <div className="step-marker">✓</div>
              <div className="step-content">
                <div className="step-label">Search (Attempt 2)</div>
                <div className="step-desc">Executed secondary retrieval with reformulated query</div>
              </div>
            </div>

            <div className="step-item step-completed">
              <div className="step-marker">✓</div>
              <div className="step-content">
                <div className="step-label">Second Relevance Check</div>
                <div className="step-desc">Evaluated results against reformulated query</div>
              </div>
            </div>
          </>
        )}

        {/* Final Step: Grounded Result */}
        <div className={`step-item ${relevance_found ? 'step-completed' : 'step-notice'}`}>
          <div className="step-marker">{relevance_found ? '✓' : '✕'}</div>
          <div className="step-content">
            <div className="step-label">
              {relevance_found
                ? 'Grounded Context Confirmed'
                : 'Relevant Information Not Found in Document'}
            </div>
            <div className="step-desc">
              {relevance_found
                ? 'Generated concise answer strictly bounded to retrieved context'
                : 'Correctly prevented hallucination by acknowledging missing data'}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// Subcomponent: Extracted Vision Image Display Card
function VisionImageCard({ imageFile, imageUrl }) {
  const [isLoaded, setIsLoaded] = useState(false);
  const [hasError, setHasError] = useState(false);

  if (!imageFile && !imageUrl) return null;

  return (
    <div className="vision-image-card">
      <div className="vision-card-header">
        <div className="vision-header-left">
          <span className="vision-header-icon">🖼️</span>
          <span className="vision-header-title">Retrieved Document Image</span>
        </div>
        <span className="vision-badge">VISION MODALITY</span>
      </div>

      <div className="vision-image-wrapper">
        {!isLoaded && !hasError && (
          <div className="image-loading-skeleton">
            <span className="skeleton-pulse">Loading extracted image...</span>
          </div>
        )}

        {hasError ? (
          <div className="image-error-box">
            <span>⚠️ Could not load image preview from server</span>
          </div>
        ) : (
          <img
            src={imageUrl}
            alt={imageFile || 'Extracted document graphic'}
            className={`vision-img ${isLoaded ? 'loaded' : ''}`}
            onLoad={() => setIsLoaded(true)}
            onError={() => setHasError(true)}
          />
        )}
      </div>

      <div className="vision-card-footer">
        <div className="vision-file-info">
          <span className="vision-file-icon">📄</span>
          <span className="vision-file-name">{imageFile}</span>
        </div>
        <a
          href={imageUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="vision-open-btn"
        >
          Open Full Image ↗
        </a>
      </div>
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [activeAgent, setActiveAgent] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  // Auto-scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  // Handle textarea auto-height adjustment
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 160)}px`;
    }
  }, [input]);

  const handleSend = async (queryToSend) => {
    const text = (queryToSend || input).trim();
    if (!text || isLoading) return;

    const userMessage = {
      id: Date.now(),
      sender: 'user',
      text: text,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const res = await fetch(`${API_BASE_URL}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ query: text })
      });

      if (!res.ok) {
        throw new Error(`Server returned status ${res.status}: ${res.statusText}`);
      }

      const data = await res.json();
      const nextAgent = data.next_agent || 'text';
      setActiveAgent(nextAgent);

      const aiMessage = {
        id: Date.now() + 1,
        sender: 'ai',
        agent: nextAgent,
        text: data.response || 'No response generated.',
        self_rag: data.self_rag || null,
        sources: data.sources || [],
        image_file: data.image_file || null,
        image_url: data.image_url
          ? `${API_BASE_URL}${data.image_url}`
          : data.image_file
          ? `${API_BASE_URL}/images/${data.image_file}`
          : null,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };

      setMessages((prev) => [...prev, aiMessage]);
    } catch (err) {
      console.error('API Error:', err);
      const errorMessage = {
        id: Date.now() + 1,
        sender: 'error',
        text: 'Unable to connect to OmniBrain backend.\n\nPlease make sure FastAPI is running on port 8000.\n(http://127.0.0.1:8000)',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleNewConversation = () => {
    setMessages([]);
    setActiveAgent(null);
    setInput('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.focus();
    }
  };

  const handleExampleClick = (queryText) => {
    setInput(queryText);
    if (textareaRef.current) {
      textareaRef.current.focus();
    }
  };

  const formatAgentName = (agentKey) => {
    if (!agentKey) return 'AGENT';
    switch (agentKey.toLowerCase()) {
      case 'text':
        return 'TEXT AGENT';
      case 'sql':
        return 'SQL AGENT';
      case 'vision':
        return 'VISION AGENT';
      default:
        return `${agentKey.toUpperCase()} AGENT`;
    }
  };

  return (
    <div className="app-layout">
      {/* Mobile Sidebar Overlay */}
      {sidebarOpen && (
        <div
          className="sidebar-overlay"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Left Sidebar */}
      <aside className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <div className="brand">
            <div className="brand-logo">🧠</div>
            <div className="brand-info">
              <h1 className="brand-title">OmniBrain</h1>
              <span className="brand-subtitle">Agentic AI Platform</span>
            </div>
          </div>
          <button
            className="mobile-close-btn"
            onClick={() => setSidebarOpen(false)}
            aria-label="Close sidebar"
          >
            ✕
          </button>
        </div>

        <div className="status-indicator-bar">
          <span className="status-dot"></span>
          <span className="status-text">System Online</span>
        </div>

        <div className="sidebar-content">
          {/* Agents List Section */}
          <div className="sidebar-section">
            <h2 className="section-title">AGENTS</h2>
            <div className="agent-cards-list">
              <div
                className={`agent-nav-card ${
                  activeAgent?.toLowerCase() === 'text' ? 'active' : ''
                }`}
              >
                <div className="agent-icon text-icon">📄</div>
                <div className="agent-details">
                  <div className="agent-name">Text Agent</div>
                  <div className="agent-sub">Document RAG</div>
                </div>
                {activeAgent?.toLowerCase() === 'text' && (
                  <span className="active-pill">ACTIVE</span>
                )}
              </div>

              <div
                className={`agent-nav-card ${
                  activeAgent?.toLowerCase() === 'sql' ? 'active' : ''
                }`}
              >
                <div className="agent-icon sql-icon">🗄️</div>
                <div className="agent-details">
                  <div className="agent-name">SQL Agent</div>
                  <div className="agent-sub">Structured Data</div>
                </div>
                {activeAgent?.toLowerCase() === 'sql' && (
                  <span className="active-pill">ACTIVE</span>
                )}
              </div>

              <div
                className={`agent-nav-card ${
                  activeAgent?.toLowerCase() === 'vision' ? 'active' : ''
                }`}
              >
                <div className="agent-icon vision-icon">👁️</div>
                <div className="agent-details">
                  <div className="agent-name">Vision Agent</div>
                  <div className="agent-sub">Images & Charts</div>
                </div>
                {activeAgent?.toLowerCase() === 'vision' && (
                  <span className="active-pill">ACTIVE</span>
                )}
              </div>
            </div>
          </div>

          {/* Capabilities Section */}
          <div className="sidebar-section">
            <h2 className="section-title">CAPABILITIES</h2>
            <ul className="capabilities-list">
              <li>
                <span className="check-icon">✓</span> Multi-Modal RAG
              </li>
              <li>
                <span className="check-icon">✓</span> Self-RAG
              </li>
              <li>
                <span className="check-icon">✓</span> Query Rewriting
              </li>
              <li>
                <span className="check-icon">✓</span> Agentic Routing
              </li>
            </ul>
          </div>
        </div>

        {/* Sidebar Footer: New Conversation Button */}
        <div className="sidebar-footer">
          <button
            className="new-conv-btn"
            onClick={handleNewConversation}
          >
            <span className="plus-icon">+</span>
            <span>New Conversation</span>
          </button>
        </div>
      </aside>

      {/* Main Panel */}
      <main className="main-panel">
        {/* Main Header */}
        <header className="main-header">
          <div className="header-left">
            <button
              className="mobile-menu-btn"
              onClick={() => setSidebarOpen(true)}
              aria-label="Open sidebar"
            >
              ☰
            </button>
            <div className="header-title-group">
              <h2 className="header-title">OmniBrain Assistant</h2>
              <p className="header-sub">Ask questions about your documents and data</p>
            </div>
          </div>

          {/* Active Agent Badge (shown only when available) */}
          {activeAgent && (
            <div className="header-agent-badge">
              <span className="badge-meta">ACTIVE AGENT</span>
              <span className={`badge-tag tag-${activeAgent.toLowerCase()}`}>
                {activeAgent.toUpperCase()}
              </span>
            </div>
          )}
        </header>

        {/* Chat / Content Canvas */}
        <div className="chat-canvas">
          {messages.length === 0 ? (
            /* Welcome Screen */
            <div className="welcome-screen">
              <div className="welcome-icon-box">
                <span className="welcome-brain-emoji">🧠</span>
              </div>
              <h3 className="welcome-heading">How can I help you?</h3>
              <p className="welcome-text">
                Ask OmniBrain about your documents, structured data, images, charts,
                or other supported content.
              </p>

              <div className="example-queries-grid">
                {EXAMPLE_QUERIES.map((item, idx) => (
                  <button
                    key={idx}
                    className="example-query-card"
                    onClick={() => handleExampleClick(item.query)}
                  >
                    <span className="example-icon">{item.icon}</span>
                    <span className="example-label">{item.label}</span>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            /* Chat Messages List */
            <div className="messages-list">
              {messages.map((msg) => {
                if (msg.sender === 'user') {
                  return (
                    <div key={msg.id} className="message-row user-row">
                      <div className="user-bubble">
                        <div className="message-content">{msg.text}</div>
                        <div className="message-time">{msg.timestamp}</div>
                      </div>
                    </div>
                  );
                }

                if (msg.sender === 'error') {
                  return (
                    <div key={msg.id} className="message-row error-row">
                      <div className="error-card">
                        <div className="error-header">
                          <span className="error-icon">⚠️</span>
                          <strong>Connection Error</strong>
                        </div>
                        <div className="error-body">{msg.text}</div>
                        <div className="message-time">{msg.timestamp}</div>
                      </div>
                    </div>
                  );
                }

                // AI Message
                const isTextAgent = msg.agent?.toLowerCase() === 'text';
                const isVisionAgent = msg.agent?.toLowerCase() === 'vision';
                const hasSources = msg.sources && msg.sources.length > 0;

                return (
                  <div key={msg.id} className="message-row ai-row">
                    <div className="ai-avatar">🧠</div>
                    <div className="ai-bubble">
                      <div className="ai-meta-bar">
                        <span className={`agent-badge badge-${msg.agent?.toLowerCase()}`}>
                          {formatAgentName(msg.agent)}
                        </span>
                        <span className="message-time">{msg.timestamp}</span>
                      </div>

                      {/* Main Agent Answer */}
                      <div className="message-content ai-text">{msg.text}</div>

                      {/* FEATURE 1: Retrieved Vision Image */}
                      {isVisionAgent && (msg.image_file || msg.image_url) && (
                        <VisionImageCard
                          imageFile={msg.image_file}
                          imageUrl={msg.image_url}
                        />
                      )}

                      {/* FEATURE 2: Self-RAG Process Status (for Text Agent) */}
                      {isTextAgent && msg.self_rag && (
                        <SelfRagCard selfRag={msg.self_rag} />
                      )}

                      {/* FEATURE 3: Source & Page Citations (for Text Agent) */}
                      {isTextAgent && hasSources && (
                        <div className="sources-container">
                          <div className="sources-header">
                            <span className="sources-title-icon">📚</span>
                            <span className="sources-title">Retrieved Sources & Citations</span>
                            <span className="sources-count-badge">
                              {msg.sources.length} {msg.sources.length === 1 ? 'chunk' : 'chunks'}
                            </span>
                          </div>

                          <div className="sources-list">
                            {msg.sources.map((source, sIdx) => (
                              <SourceItem key={sIdx} source={source} />
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Fallback capability indicator if no detailed self-rag */}
                      {isTextAgent && !msg.self_rag && (
                        <div className="self-rag-tag">
                          <span className="rag-check">✓</span> Self-RAG enabled
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}

              {/* Loading State */}
              {isLoading && (
                <div className="message-row ai-row loading-row">
                  <div className="ai-avatar pulse">🧠</div>
                  <div className="ai-bubble loading-bubble">
                    <div className="loading-header">
                      <span className="loading-brand">OMNIBRAIN</span>
                    </div>
                    <div className="loading-body">
                      <span>OmniBrain is thinking...</span>
                      <span className="loading-dots">
                        <span>.</span>
                        <span>.</span>
                        <span>.</span>
                      </span>
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Bottom Input Area */}
        <div className="chat-input-section">
          <div className="input-container">
            <textarea
              ref={textareaRef}
              className="chat-textarea"
              rows={1}
              placeholder="Ask OmniBrain anything..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isLoading}
            />
            <button
              className="send-button"
              onClick={() => handleSend()}
              disabled={!input.trim() || isLoading}
              title="Send query"
              aria-label="Send query"
            >
              <svg
                className="send-icon"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <line x1="22" y1="2" x2="11" y2="13"></line>
                <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
              </svg>
            </button>
          </div>

          <div className="input-footer-bar">
            <span className="brand-tag">OmniBrain Agentic Multi-Modal RAG</span>
            <span className="keyboard-hint">Press Enter to send</span>
          </div>
        </div>
      </main>
    </div>
  );
}
