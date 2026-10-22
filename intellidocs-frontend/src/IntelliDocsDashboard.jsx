// src/IntelliDocsDashboard.jsx
import React, { useState, useEffect } from 'react';

const API = 'http://localhost:8000/api/v1';

export default function IntelliDocsDashboard() {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [query, setQuery] = useState('');
  const [loadingAnswer, setLoadingAnswer] = useState(false);
  const [chatHistory, setChatHistory] = useState([]);
  const [activeSources, setActiveSources] = useState([]);
  const [sessionId, setSessionId] = useState(null);
  const [allSessions, setAllSessions] = useState([]); // New: Sidebar history
  const [isSidebarOpen, setIsSidebarOpen] = useState(true); // New: Sidebar toggle

  // Load sessions and current chat history
  useEffect(() => {
    const init = async () => {
      await loadSessions();
      let sid = localStorage.getItem('intellidocs_session');
      if (!sid) {
        sid = await createNewSession();
      } else {
        setSessionId(sid);
        await loadHistory(sid);
      }
    };
    init();
  }, []);

  const loadSessions = async () => {
    try {
      const res = await fetch(`${API}/sessions`);
      const data = await res.json();
      setAllSessions(data.sessions || []);
    } catch (e) { console.error(e); }
  };

  const createNewSession = async () => {
    try {
      const res = await fetch(`${API}/sessions`, { method: 'POST' });
      const data = await res.json();
      localStorage.setItem('intellidocs_session', data.session_id);
      setSessionId(data.session_id);
      setChatHistory([]);
      setActiveSources([]);
      await loadSessions();
      return data.session_id;
    } catch (e) { console.error(e); }
  };

  const loadHistory = async (sid) => {
    try {
      const res = await fetch(`${API}/sessions/${sid}/messages`);
      const data = await res.json();
      const msgs = (data.messages || []).map((m) => ({
        role: m.role, content: m.content, citations: m.citations || []
      }));
      setChatHistory(msgs);
      setActiveSources([]);
    } catch (e) { console.error(e); }
  };

  const switchSession = async (sid) => {
    localStorage.setItem('intellidocs_session', sid);
    setSessionId(sid);
    await loadHistory(sid);
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    const fd = new FormData();
    fd.append('file', file);
    try {
      const res = await fetch(`${API}/upload`, { method: 'POST', body: fd });
      const data = await res.json();
      if (res.ok) {
        alert(`Ingestion Successful! ID: ${data.document_id}`);
      } else {
        alert(`Upload failed: ${JSON.stringify(data.detail || data)}`);
      }
    } catch (err) { alert('Backend unreachable.'); }
    finally { setUploading(false); }
  };

  const handleAskQuestion = async (e) => {
    e.preventDefault();
    if (!query.trim() || !sessionId) return;

    setLoadingAnswer(true);
    const fd = new FormData();
    fd.append('query', query);
    fd.append('session_id', sessionId);
    // We now hardcode 'auto' — backend handles detection
    fd.append('lang', 'auto'); 

    try {
      const res = await fetch(`${API}/chat`, { method: 'POST', body: fd });
      const data = await res.json();

      setChatHistory((prev) => [
        ...prev,
        { role: 'user', content: query },
        { role: 'assistant', content: data.answer, citations: data.citations },
      ]);
      setActiveSources(data.retrieved_sources || []);
      setQuery('');
      await loadSessions(); // refresh sidebar to show updated time
    } catch (err) {
      console.error(err);
    } finally {
      setLoadingAnswer(false);
    }
  };

  // --- Styles ---
  const layoutStyle = { display: 'flex', height: '100vh', background: '#0f172a', color: '#f8fafc', fontFamily: 'sans-serif' };
  const sidebarStyle = { width: isSidebarOpen ? '260px' : '0', background: '#1e293b', transition: 'width 0.3s', overflow: 'hidden', display: 'flex', flexDirection: 'column', borderRight: '1px solid #334155' };
  const mainStyle = { flex: 1, display: 'flex', flexDirection: 'column', padding: '20px', overflowY: 'auto' };
  const boxStyle = { background: '#1e293b', border: '1px solid #334155', padding: '20px', borderRadius: '8px', marginBottom: '20px' };

  return (
    <div style={layoutStyle}>
      {/* Sidebar: History */}
      <div style={sidebarStyle}>
        <div style={{ padding: '20px', borderBottom: '1px solid #334155' }}>
          <button onClick={createNewSession} style={{ width: '100%', padding: '10px', background: '#059669', color: '#fff', border: 'none', borderRadius: '6px', cursor: 'pointer', fontWeight: 'bold' }}>
            + New Chat
          </button>
        </div>
        <div style={{ flex: 1, overflowY: 'auto', padding: '10px' }}>
          <p style={{ fontSize: '11px', color: '#64748b', marginLeft: '10px', marginBottom: '10px' }}>RECENT CHATS</p>
          {allSessions.map((s) => (
            <div 
              key={s.session_id} 
              onClick={() => switchSession(s.session_id)}
              style={{ padding: '10px', marginBottom: '5px', borderRadius: '6px', cursor: 'pointer', background: s.session_id === sessionId ? '#334155' : 'transparent', color: s.session_id === sessionId ? '#34d399' : '#94a3b8', fontSize: '13px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}
            >
              Chat • {new Date(s.created_at).toLocaleTimeString()}
            </div>
          ))}
        </div>
      </div>

      {/* Main App */}
      <div style={mainStyle}>
        <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', borderBottom: '1px solid #334155', paddingBottom: '10px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
            <button onClick={() => setIsSidebarOpen(!isSidebarOpen)} style={{ background: 'transparent', color: '#94a3b8', border: 'none', fontSize: '20px', cursor: 'pointer' }}>☰</button>
            <h1 style={{ color: '#34d399', margin: 0, fontSize: '20px' }}>BhashaRAG</h1>
          </div>
          <span style={{ fontSize: '12px', color: '#64748b' }}>Auto-detect Language Mode</span>
        </header>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '20px', maxWidth: '1400px', width: '100%', margin: '0 auto' }}>
          {/* Ingestion Panel */}
          <div style={boxStyle}>
            <h2 style={{ fontSize: '14px', marginTop: 0, color: '#e2e8f0' }}>Document Ingestion</h2>
            <div style={{ border: '2px dashed #475569', padding: '20px', textAlign: 'center', borderRadius: '6px', background: '#0f172a' }}>
              <input type="file" accept=".pdf" onChange={(e) => setFile(e.target.files[0])} id="file-upload" style={{ display: 'none' }} />
              <label htmlFor="file-upload" style={{ cursor: 'pointer', color: '#94a3b8', fontSize: '13px' }}>
                {file ? <strong style={{ color: '#34d399' }}>{file.name}</strong> : 'Click to upload a PDF'}
              </label>
            </div>
            <button onClick={handleUpload} disabled={!file || uploading} style={{ width: '100%', marginTop: '15px', background: uploading ? '#475569' : '#059669', color: '#fff', border: 'none', padding: '10px', borderRadius: '6px', cursor: uploading ? 'not-allowed' : 'pointer', fontWeight: 'bold' }}>
              {uploading ? 'Processing...' : 'Ingest to Vector Engine'}
            </button>
          </div>

          {/* Chat Panel */}
          <div style={{ ...boxStyle, minHeight: '500px', display: 'flex', flexDirection: 'column' }}>
            <div style={{ flex: 1, overflowY: 'auto', background: '#0f172a', padding: '15px', borderRadius: '6px', marginBottom: '15px' }}>
              {chatHistory.length === 0 && <p style={{ color: '#475569', textAlign: 'center', fontStyle: 'italic' }}>Ask a question in any language...</p>}
              {chatHistory.map((msg, i) => (
                <div key={i} style={{ marginBottom: '15px', padding: '12px', borderRadius: '6px', background: msg.role === 'user' ? '#334155' : '#1e293b', border: msg.role === 'user' ? 'none' : '1px solid #475569', maxWidth: '85%', marginLeft: msg.role === 'user' ? 'auto' : '0' }}>
                  <p style={{ margin: 0, fontSize: '14px', lineHeight: '1.5', whiteSpace: 'pre-wrap' }}>{msg.content}</p>
                  {msg.citations?.length > 0 && (
                    <div style={{ marginTop: '8px', display: 'flex', gap: '5px', flexWrap: 'wrap' }}>
                      <span style={{ fontSize: '10px', color: '#64748b', fontWeight: 'bold' }}>SOURCES:</span>
                      {msg.citations.map((c, j) => (
                        <span key={j} style={{ fontSize: '10px', background: '#064e3b', color: '#34d399', padding: '2px 6px', borderRadius: '4px', fontFamily: 'monospace' }}>Page {c}</span>
                      ))}
                    </div>
                  )}
                </div>
              ))}
              {loadingAnswer && <p style={{ fontSize: '12px', color: '#94a3b8', fontStyle: 'italic' }}>Thinking...</p>}
            </div>

            <form onSubmit={handleAskQuestion} style={{ display: 'flex', gap: '10px' }}>
              <input 
                type="text" 
                value={query} 
                onChange={(e) => setQuery(e.target.value)} 
                placeholder="Ask in any language (e.g., 'What is this project about?')" 
                style={{ flex: 1, background: '#0f172a', border: '1px solid #475569', borderRadius: '6px', padding: '12px', color: '#fff', fontSize: '14px' }} 
              />
              <button type="submit" disabled={loadingAnswer || !query.trim()} style={{ background: '#2563eb', color: '#fff', border: 'none', padding: '0 25px', borderRadius: '6px', cursor: 'pointer', fontWeight: 'bold' }}>
                Send
              </button>
            </form>
          </div>
        </div>

        {/* Evidence Blocks */}
        {activeSources.length > 0 && (
          <div style={{ maxWidth: '1400px', width: '100%', margin: '20px auto 0' }}>
             <h3 style={{ fontSize: '11px', color: '#64748b', textTransform: 'uppercase' }}>Retrieved Evidence</h3>
             <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '10px' }}>
               {activeSources.map((src, i) => (
                 <div key={i} style={{ background: '#1e293b', border: '1px solid #334155', padding: '10px', borderRadius: '6px' }}>
                   <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', marginBottom: '5px', color: '#64748b' }}>
                     <span>Page {src.page}</span>
                     <span>Score: {(src.score * 100).toFixed(1)}%</span>
                   </div>
                   <p style={{ margin: 0, fontSize: '11px', color: '#94a3b8', fontStyle: 'italic' }}>"{src.text.substring(0, 100)}..."</p>
                 </div>
               ))}
             </div>
          </div>
        )}
      </div>
    </div>
  );
}