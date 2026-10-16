// src/IntelliDocsDashboard.jsx
import React, { useState, useEffect } from 'react';

const API = 'http://localhost:8000/api/v1';

const LANGUAGES = [
  { code: 'en',  label: 'English' },
  { code: 'hi',  label: 'हिन्दी (Hindi)' },
  { code: 'mr',  label: 'मराठी (Marathi)' },
  { code: 'bn',  label: 'বাংলা (Bengali)' },
  { code: 'as',  label: 'অসমীয়া (Assamese)' },
  { code: 'gu',  label: 'ગુજરાતી (Gujarati)' },
  { code: 'pa',  label: 'ਪੰਜਾਬੀ (Punjabi)' },
  { code: 'or',  label: 'ଓଡ଼ିଆ (Odia)' },
  { code: 'ta',  label: 'தமிழ் (Tamil)' },
  { code: 'te',  label: 'తెలుగు (Telugu)' },
  { code: 'kn',  label: 'ಕನ್ನಡ (Kannada)' },
  { code: 'ml',  label: 'മലയാളം (Malayalam)' },
  { code: 'ur',  label: 'اردو (Urdu)' },
  { code: 'ne',  label: 'नेपाली (Nepali)' },
  { code: 'sa',  label: 'संस्कृतम् (Sanskrit)' },
  { code: 'mai', label: 'मैथिली (Maithili)' },
  { code: 'brx', label: 'बड़ो (Bodo)' },
  { code: 'doi', label: 'डोगरी (Dogri)' },
  { code: 'gom', label: 'कोंकणी (Konkani)' },
  { code: 'ks',  label: 'کٲشُر (Kashmiri)' },
];

export default function IntelliDocsDashboard() {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [documentId, setDocumentId] = useState(null);
  const [query, setQuery] = useState('');
  const [loadingAnswer, setLoadingAnswer] = useState(false);
  const [chatHistory, setChatHistory] = useState([]);
  const [activeSources, setActiveSources] = useState([]);
  const [lang, setLang] = useState('en');
  const [sessionId, setSessionId] = useState(null);
  const [uploads, setUploads] = useState([]);

  // Initialize session + load history + load uploads on mount
  useEffect(() => {
    const init = async () => {
      let sid = localStorage.getItem('intellidocs_session');
      if (!sid) {
        try {
          const res = await fetch(`${API}/sessions`, { method: 'POST' });
          const data = await res.json();
          sid = data.session_id;
          localStorage.setItem('intellidocs_session', sid);
        } catch (e) {
          console.error('Failed to create session:', e);
          return;
        }
      }
      setSessionId(sid);
      await Promise.all([loadHistory(sid), loadUploads()]);
    };
    init();
  }, []);

  const loadHistory = async (sid) => {
    try {
      const res = await fetch(`${API}/sessions/${sid}/messages`);
      const data = await res.json();
      const msgs = (data.messages || []).map((m) => ({
        role: m.role,
        content: m.content,
        citations: m.citations || [],
      }));
      setChatHistory(msgs);
    } catch (e) {
      console.error('Load history failed:', e);
    }
  };

  const loadUploads = async () => {
    try {
      const res = await fetch(`${API}/uploads`);
      const data = await res.json();
      setUploads(data.uploads || []);
    } catch (e) {
      console.error('Load uploads failed:', e);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) setFile(e.target.files[0]);
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
        setDocumentId(data.document_id);
        alert(`Ingestion Successful! ID: ${data.document_id}`);
        await loadUploads();
      } else {
        alert(`Upload failed: ${JSON.stringify(data.detail || data)}`);
      }
    } catch (err) {
      alert('Unable to reach backend.');
    } finally {
      setUploading(false);
    }
  };

const handleAskQuestion = async (e) => {
  e.preventDefault();
  if (!query.trim()) return;

  setLoadingAnswer(true);
  const formData = new FormData();
  formData.append('query', query);
  formData.append('lang', lang);          // ← MUST be the state value
  formData.append('session_id', sessionId);
  if (sessionId) formData.append('session_id', sessionId);
  // If sessionId is null, backend creates one and returns it

  try {
    const res = await fetch(`${API}/chat`, { method: 'POST', body: formData });
    if (!res.ok) {
      const errText = await res.text();
      console.error('Chat failed:', res.status, errText);
      alert(`Chat error ${res.status}. See console.`);
      setLoadingAnswer(false);
      return;
    }
    const data = await res.json();

    // Save the session ID if the backend just created one
    if (data.session_id && data.session_id !== sessionId) {
      localStorage.setItem('intellidocs_session', data.session_id);
      setSessionId(data.session_id);
    }

    setChatHistory((prev) => [
      ...prev,
      { role: 'user', content: query },
      { role: 'assistant', content: data.answer, citations: data.citations },
    ]);
    setActiveSources(data.retrieved_sources || []);
    setQuery('');
  } catch (err) {
    console.error(err);
  } finally {
    setLoadingAnswer(false);
  }
};

  const handleClearChat = async () => {
    if (!sessionId) return;
    if (!window.confirm('Clear all messages in this chat?')) return;
    try {
      const res = await fetch(`${API}/sessions/${sessionId}/messages`, {
        method: 'DELETE',
      });
      const data = await res.json();
      console.log('Cleared:', data);
      setChatHistory([]);
      setActiveSources([]);
    } catch (e) {
      alert('Clear failed: ' + e);
    }
  };

  const handleNewSession = async () => {
    if (!window.confirm('Start a fresh session? Current chat will be cleared from view.')) return;
    try {
      const res = await fetch(`${API}/sessions`, { method: 'POST' });
      const data = await res.json();
      localStorage.setItem('intellidocs_session', data.session_id);
      setSessionId(data.session_id);
      setChatHistory([]);
      setActiveSources([]);
    } catch (e) {
      alert('New session failed: ' + e);
    }
  };

  const boxStyle = {
    background: '#1e293b',
    border: '1px solid #334155',
    padding: '20px',
    borderRadius: '8px',
    marginBottom: '20px',
  };

  return (
    <div style={{ background: '#0f172a', color: '#f8fafc', minHeight: '100vh', padding: '30px', fontFamily: 'sans-serif' }}>
      <header style={{ borderBottom: '1px solid #334155', paddingBottom: '15px', marginBottom: '25px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h1 style={{ color: '#34d399', margin: 0, fontSize: '24px' }}>
          IntelliDocs <span style={{ color: '#94a3b8', fontSize: '14px', fontWeight: 'normal' }}>| Core Intelligence Studio</span>
        </h1>
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          <select
            value={lang}
            onChange={(e) => setLang(e.target.value)}
            style={{
              background: '#1e293b',
              color: '#f8fafc',
              border: '1px solid #475569',
              borderRadius: '6px',
              padding: '8px 12px',
              fontSize: '13px',
              cursor: 'pointer',
            }}
          >
            {LANGUAGES.map((l) => (
              <option key={l.code} value={l.code}>{l.label}</option>
            ))}
          </select>
          <button
            onClick={handleNewSession}
            style={{
              background: '#475569',
              color: '#fff',
              border: 'none',
              padding: '8px 14px',
              borderRadius: '6px',
              cursor: 'pointer',
              fontSize: '13px',
              fontWeight: 'bold',
            }}
          >
            New Session
          </button>
        </div>
      </header>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '20px' }}>
        {/* Left panel: Document Ingestion */}
        <div style={boxStyle}>
          <h2 style={{ fontSize: '16px', marginTop: 0, color: '#e2e8f0' }}>1. Document Ingestion</h2>
          <div style={{ border: '2px dashed #475569', padding: '20px', textAlign: 'center', borderRadius: '6px', background: '#0f172a' }}>
            <input type="file" accept=".pdf,.png,.jpg,.jpeg" onChange={handleFileChange} id="file-upload-picker" style={{ display: 'none' }} />
            <label htmlFor="file-upload-picker" style={{ cursor: 'pointer', color: '#94a3b8', fontSize: '14px' }}>
              {file ? <strong style={{ color: '#34d399' }}>{file.name}</strong> : 'Click to load local context profile'}
            </label>
          </div>
          <button
            onClick={handleUpload}
            disabled={!file || uploading}
            style={{
              width: '100%',
              marginTop: '15px',
              background: uploading ? '#475569' : '#059669',
              color: '#fff',
              border: 'none',
              padding: '10px',
              borderRadius: '6px',
              cursor: uploading ? 'not-allowed' : 'pointer',
              fontWeight: 'bold',
            }}
          >
            {uploading ? 'Processing Architecture Vector Indexes...' : 'Ingest to Vector Engine'}
          </button>

          {uploads.length > 0 && (
            <div style={{ marginTop: '20px' }}>
              <h3 style={{ fontSize: '11px', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '1px' }}>
                Upload History ({uploads.length})
              </h3>
              <ul style={{ padding: 0, margin: '8px 0 0 0', listStyle: 'none', maxHeight: '180px', overflowY: 'auto' }}>
                {uploads.map((u) => (
                  <li key={u.document_id} style={{
                    fontSize: '11px',
                    padding: '6px 8px',
                    marginBottom: '4px',
                    background: '#0f172a',
                    borderRadius: '4px',
                    border: '1px solid #334155',
                    color: '#94a3b8',
                  }}>
                    <div style={{ color: '#34d399', fontWeight: 'bold' }}>{u.filename}</div>
                    <div>{u.pages_processed} pages · {new Date(u.uploaded_at).toLocaleDateString()}</div>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        {/* Right panel: Chat */}
        <div style={{ ...boxStyle, minHeight: '500px', display: 'flex', flexDirection: 'column' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
            <h2 style={{ fontSize: '16px', margin: 0, color: '#e2e8f0' }}>2. Grounded Conversation</h2>
            <button
              onClick={handleClearChat}
              disabled={chatHistory.length === 0}
              style={{
                background: chatHistory.length === 0 ? '#334155' : '#dc2626',
                color: '#fff',
                border: 'none',
                padding: '6px 14px',
                borderRadius: '6px',
                cursor: chatHistory.length === 0 ? 'not-allowed' : 'pointer',
                fontSize: '12px',
                fontWeight: 'bold',
              }}
              title="Delete all messages in this session"
            >
              🗑 Clear Chat
            </button>
          </div>

          <div style={{
            flex: 1,
            maxHeight: '400px',
            overflowY: 'auto',
            background: '#0f172a',
            padding: '15px',
            borderRadius: '6px',
            minHeight: '250px',
          }}>
            {chatHistory.length === 0 && (
              <p style={{ color: '#475569', fontSize: '13px', fontStyle: 'italic' }}>
                Workspace active. Submit context profiles to query.
              </p>
            )}
            {chatHistory.map((msg, index) => (
              <div
                key={index}
                style={{
                  marginBottom: '15px',
                  padding: '10px 12px',
                  borderRadius: '6px',
                  background: msg.role === 'user' ? '#334155' : '#1e293b',
                  border: msg.role === 'user' ? 'none' : '1px solid #475569',
                  maxWidth: '85%',
                  marginLeft: msg.role === 'user' ? 'auto' : '0',
                }}
              >
                <p style={{ margin: 0, fontSize: '13px', lineHeight: '1.5', whiteSpace: 'pre-wrap' }}>
                  {msg.content}
                </p>
                {msg.citations && msg.citations.length > 0 && (
                  <div style={{ marginTop: '8px', display: 'flex', gap: '5px', alignItems: 'center', flexWrap: 'wrap' }}>
                    <span style={{ fontSize: '10px', color: '#64748b', fontWeight: 'bold' }}>SOURCES:</span>
                    {msg.citations.map((c, i) => (
                      <span key={i} style={{
                        fontSize: '10px',
                        background: '#064e3b',
                        color: '#34d399',
                        padding: '2px 6px',
                        borderRadius: '4px',
                        fontFamily: 'monospace',
                      }}>
                        Page {c}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
            {loadingAnswer && (
              <p style={{ fontSize: '12px', color: '#94a3b8', fontStyle: 'italic' }}>
                Synthesizing verified model answers...
              </p>
            )}
          </div>

          <form onSubmit={handleAskQuestion} style={{ display: 'flex', gap: '10px', marginTop: '15px' }}>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Query document text layers..."
              disabled={!sessionId}
              style={{
                flex: 1,
                background: '#0f172a',
                border: '1px solid #475569',
                borderRadius: '6px',
                padding: '10px',
                color: '#fff',
                fontSize: '13px',
              }}
            />
            <button
              type="submit"
              disabled={loadingAnswer || !query.trim() || !sessionId}
              style={{
                background: loadingAnswer ? '#475569' : '#2563eb',
                color: '#fff',
                border: 'none',
                padding: '10px 20px',
                borderRadius: '6px',
                cursor: loadingAnswer ? 'not-allowed' : 'pointer',
                fontWeight: 'bold',
              }}
            >
              {loadingAnswer ? '...' : 'Query'}
            </button>
          </form>
        </div>
      </div>

      {activeSources.length > 0 && (
        <div style={{ ...boxStyle, marginTop: '20px' }}>
          <h3 style={{ fontSize: '12px', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '1px', marginTop: 0 }}>
            Grounded Evidence Blocks
          </h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '15px', marginTop: '10px' }}>
            {activeSources.map((src, i) => (
              <div key={i} style={{ background: '#0f172a', border: '1px solid #475569', padding: '10px', borderRadius: '6px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', marginBottom: '8px' }}>
                  <span style={{ background: '#334155', padding: '2px 6px', borderRadius: '4px' }}>Page {src.page}</span>
                  <span style={{ color: '#64748b' }}>Score: {(src.score * 100).toFixed(1)}%</span>
                </div>
                <p style={{ margin: 0, fontSize: '12px', color: '#94a3b8', fontStyle: 'italic', lineHeight: '1.4', maxHeight: '100px', overflowY: 'auto' }}>
                  "{src.text}"
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}