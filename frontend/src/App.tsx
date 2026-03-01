import { useState, useRef, useCallback } from 'react'
import {
  Send, Loader2, Trash2, Upload, Search, Database, BarChart3, FileText,
  ChevronDown, Zap, ThumbsUp, ThumbsDown, FlaskConical, List, RefreshCw, X
} from 'lucide-react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// ─── Types ────────────────────────────────────────────────────────────────────

interface RAGSource {
  content: string
  score: number
  metadata: Record<string, unknown>
}

interface JudgeResult {
  score: number       // 0-10
  reasoning: string
  model: string
}

interface Message {
  role: 'user' | 'assistant'
  content: string
  message_id?: string
  sources?: RAGSource[]
  eval_result?: JudgeResult
}

interface APIResponse {
  content: string
  model: string
  usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number }
  cost_usd: number
  latency_ms: number
  sources?: RAGSource[]
}

interface RAGResult {
  id: string
  content: string
  score: number
  metadata: Record<string, unknown>
  semantic_score?: number
  bm25_score?: number
}

interface RAGQueryResponse {
  query: string
  results: RAGResult[]
  total_results: number
  search_mode: string
  latency_ms: number
}

interface RAGDocumentItem {
  id: string
  content: string
  metadata: Record<string, unknown>
}

interface CacheStats {
  enabled: boolean
  backend?: string
  total_entries?: number
  hit_rate?: number
  hits?: number
  misses?: number
}

// ─── Shared Components ────────────────────────────────────────────────────────

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-gray-50 rounded-lg p-3">
      <p className="text-xs text-gray-500 mb-1">{label}</p>
      <p className="text-sm font-medium text-gray-800">{value}</p>
    </div>
  )
}

// ─── SourcesPanel ─────────────────────────────────────────────────────────────

function SourcesPanel({ sources }: { sources: RAGSource[] }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="mt-2">
      <button
        onClick={() => setOpen(o => !o)}
        className="flex items-center gap-1 text-xs text-purple-600 hover:text-purple-800 font-medium transition-colors"
      >
        <Database className="w-3 h-3" />
        {sources.length} source{sources.length !== 1 ? 's' : ''}
        <ChevronDown className={`w-3 h-3 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      {open && (
        <div className="mt-2 space-y-2">
          {sources.map((src, i) => (
            <div key={i} className="bg-purple-50 border border-purple-200 rounded-lg p-3">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-semibold text-purple-700">Source {i + 1}</span>
                <span className="text-xs font-mono text-purple-500 bg-purple-100 px-1.5 py-0.5 rounded">
                  {src.score.toFixed(4)}
                </span>
              </div>
              <p className="text-xs text-gray-700 leading-relaxed">{src.content}</p>
              {!!src.metadata.source && (
                <p className="text-xs text-gray-400 mt-1">— {String(src.metadata.source)}</p>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ─── RateThisPanel ────────────────────────────────────────────────────────────

function RateThisPanel({
  messageId,
  question,
  response,
  evalResult,
  onEval,
}: {
  messageId?: string
  question: string
  response: string
  evalResult?: JudgeResult
  onEval: (result: JudgeResult) => void
}) {
  const [submitted, setSubmitted] = useState<'up' | 'down' | null>(null)
  const [judging, setJudging] = useState(false)
  const [reasoning, setReasoningOpen] = useState(false)

  const submitFeedback = async (rating: 'thumbs_up' | 'thumbs_down') => {
    setSubmitted(rating === 'thumbs_up' ? 'up' : 'down')
    try {
      await fetch(`${API_URL}/v1/evals/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message_id: messageId ?? 'unknown',
          question,
          response,
          rating,
          run_judge: false,
        }),
      })
    } catch {
      // non-blocking; feedback failure shouldn't disrupt chat
    }
  }

  const runJudge = async () => {
    setJudging(true)
    try {
      const res = await fetch(`${API_URL}/v1/evals/judge`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, response }),
      })
      if (!res.ok) throw new Error('Judge failed')
      const data = await res.json()
      onEval({ score: data.score, reasoning: data.reasoning, model: data.model })
    } catch {
      // silently ignore
    } finally {
      setJudging(false)
    }
  }

  const scoreColor =
    evalResult == null ? ''
    : evalResult.score >= 7 ? 'text-green-600 bg-green-50 border-green-200'
    : evalResult.score >= 4 ? 'text-amber-600 bg-amber-50 border-amber-200'
    : 'text-red-600 bg-red-50 border-red-200'

  return (
    <div className="mt-2 flex items-center gap-2 flex-wrap">
      {/* Thumbs */}
      <button
        onClick={() => submitFeedback('thumbs_up')}
        title="Good response"
        className={`p-1.5 rounded-lg transition-colors border ${
          submitted === 'up'
            ? 'bg-green-100 text-green-600 border-green-300'
            : 'bg-gray-50 text-gray-400 border-gray-200 hover:bg-green-50 hover:text-green-500 hover:border-green-200'
        }`}
      >
        <ThumbsUp className="w-3 h-3" />
      </button>
      <button
        onClick={() => submitFeedback('thumbs_down')}
        title="Poor response"
        className={`p-1.5 rounded-lg transition-colors border ${
          submitted === 'down'
            ? 'bg-red-100 text-red-600 border-red-300'
            : 'bg-gray-50 text-gray-400 border-gray-200 hover:bg-red-50 hover:text-red-500 hover:border-red-200'
        }`}
      >
        <ThumbsDown className="w-3 h-3" />
      </button>

      {/* AI Judge button */}
      {!evalResult && (
        <button
          onClick={runJudge}
          disabled={judging}
          title="Score with AI judge"
          className="flex items-center gap-1 px-2 py-1 rounded-lg text-xs border bg-gray-50 text-gray-500 border-gray-200 hover:bg-indigo-50 hover:text-indigo-600 hover:border-indigo-200 disabled:opacity-50 transition-colors"
        >
          {judging ? <Loader2 className="w-3 h-3 animate-spin" /> : <FlaskConical className="w-3 h-3" />}
          Rate
        </button>
      )}

      {/* Score badge */}
      {evalResult && (
        <button
          onClick={() => setReasoningOpen(o => !o)}
          className={`flex items-center gap-1 px-2 py-1 rounded-lg text-xs border font-medium transition-colors ${scoreColor}`}
        >
          <FlaskConical className="w-3 h-3" />
          {evalResult.score.toFixed(1)}/10
          <ChevronDown className={`w-3 h-3 transition-transform ${reasoning ? 'rotate-180' : ''}`} />
        </button>
      )}

      {/* Reasoning */}
      {evalResult && reasoning && (
        <div className={`w-full mt-1 p-2 rounded-lg text-xs border ${scoreColor}`}>
          {evalResult.reasoning}
          <span className="ml-1 text-gray-400">— {evalResult.model}</span>
        </div>
      )}
    </div>
  )
}

// ─── Chat Tab ─────────────────────────────────────────────────────────────────

function ChatTab() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [model, setModel] = useState('gpt-4o-mini')
  const [loading, setLoading] = useState(false)
  const [lastResponse, setLastResponse] = useState<APIResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [ragEnabled, setRagEnabled] = useState(false)
  const [rerankEnabled, setRerankEnabled] = useState(false)
  const [streamEnabled, setStreamEnabled] = useState(false)

  const sendMessage = useCallback(async () => {
    if (!input.trim() || loading) return
    const userMessage: Message = {
      role: 'user',
      content: input,
      message_id: crypto.randomUUID(),
    }
    const newMessages = [...messages, userMessage]
    setMessages(newMessages)
    setInput('')
    setLoading(true)
    setError(null)

    const url = `${API_URL}/v1/chat/completions?use_rag=${ragEnabled}&use_rerank=${rerankEnabled}&stream=${streamEnabled}`

    if (!streamEnabled) {
      // ── Non-streaming path ──────────────────────────────────────────────
      try {
        const response = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ messages: newMessages, model, temperature: 0.7, max_tokens: 500 })
        })
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}))
          throw new Error(errorData.detail || `API error: ${response.status}`)
        }
        const data: APIResponse = await response.json()
        setMessages([...newMessages, {
          role: 'assistant',
          content: data.content,
          message_id: crypto.randomUUID(),
          sources: data.sources ?? undefined,
        }])
        setLastResponse(data)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to get response')
        setMessages(newMessages.slice(0, -1))
      } finally {
        setLoading(false)
      }
    } else {
      // ── Streaming path ──────────────────────────────────────────────────
      const assistantIdx = newMessages.length
      setMessages([...newMessages, { role: 'assistant', content: '', message_id: crypto.randomUUID() }])

      try {
        const response = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ messages: newMessages, model, temperature: 0.7, max_tokens: 500 })
        })
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}))
          throw new Error(errorData.detail || `API error: ${response.status}`)
        }

        const reader = response.body!.getReader()
        const decoder = new TextDecoder()
        let buffer = ''
        let fullContent = ''
        let doneSeen = false

        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop() ?? ''

          for (const line of lines) {
            if (!line.startsWith('data: ')) continue
            const raw = line.slice(6).trim()
            if (raw === '[DONE]') { doneSeen = true; continue }
            try {
              const chunk = JSON.parse(raw)
              if (chunk.error) throw new Error(chunk.error)

              // Sources event (arrives after [DONE])
              if (doneSeen && chunk.sources) {
                setMessages(prev => prev.map((m, i) =>
                  i === assistantIdx ? { ...m, sources: chunk.sources } : m
                ))
                continue
              }

              if (chunk.content) {
                fullContent += chunk.content
                setMessages(prev => prev.map((m, i) =>
                  i === assistantIdx ? { ...m, content: fullContent } : m
                ))
              }

              if (chunk.usage) {
                setLastResponse({
                  content: fullContent,
                  model,
                  usage: chunk.usage,
                  cost_usd: 0,
                  latency_ms: 0,
                })
              }
            } catch {
              // ignore parse errors on individual chunks
            }
          }
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Streaming failed')
        setMessages(prev => prev.slice(0, assistantIdx))
      } finally {
        setLoading(false)
      }
    }
  }, [input, loading, messages, model, ragEnabled, rerankEnabled, streamEnabled])

  const updateMessageEval = (idx: number, result: JudgeResult) => {
    setMessages(prev => prev.map((m, i) => i === idx ? { ...m, eval_result: result } : m))
  }

  return (
    <div className="flex flex-col h-full">
      {/* Model selector + toggles */}
      <div className="flex items-center justify-between p-3 border-b border-gray-100 bg-gray-50">
        <div className="flex items-center gap-2">
          <select
            value={model}
            onChange={(e) => setModel(e.target.value)}
            className="px-3 py-1.5 border border-gray-300 rounded-lg text-sm bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="gpt-4o-mini">GPT-4o Mini</option>
            <option value="gpt-4o">GPT-4o</option>
            <option value="claude-sonnet">Claude Sonnet</option>
            <option value="claude-opus">Claude Opus</option>
            <option value="ollama/phi3">Ollama Phi-3 (Local)</option>
            <option value="ollama/llama3">Ollama Llama3 (Local)</option>
            <option value="ollama/mistral">Ollama Mistral (Local)</option>
          </select>
          <button
            onClick={() => setRagEnabled(r => !r)}
            title="Toggle RAG context injection"
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold border transition-colors ${
              ragEnabled
                ? 'bg-purple-500 text-white border-purple-500 shadow-sm'
                : 'bg-white text-gray-500 border-gray-300 hover:bg-purple-50 hover:text-purple-600 hover:border-purple-300'
            }`}
          >
            <Database className="w-3.5 h-3.5" />
            RAG
          </button>
          <button
            onClick={() => setRerankEnabled(r => !r)}
            title="Apply cross-encoder reranking to RAG results (requires RAG)"
            disabled={!ragEnabled}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold border transition-colors disabled:opacity-40 disabled:cursor-not-allowed ${
              rerankEnabled && ragEnabled
                ? 'bg-green-500 text-white border-green-500 shadow-sm'
                : 'bg-white text-gray-500 border-gray-300 hover:bg-green-50 hover:text-green-600 hover:border-green-300'
            }`}
          >
            <Search className="w-3.5 h-3.5" />
            Rerank
          </button>
          <button
            onClick={() => setStreamEnabled(s => !s)}
            title="Toggle streaming response"
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold border transition-colors ${
              streamEnabled
                ? 'bg-amber-400 text-white border-amber-400 shadow-sm'
                : 'bg-white text-gray-500 border-gray-300 hover:bg-amber-50 hover:text-amber-600 hover:border-amber-300'
            }`}
          >
            <Zap className="w-3.5 h-3.5" />
            Stream
          </button>
        </div>
        {messages.length > 0 && (
          <button onClick={() => { setMessages([]); setLastResponse(null); setError(null) }}
            className="p-1.5 text-gray-500 hover:text-red-500 hover:bg-red-50 rounded-lg transition-colors">
            <Trash2 className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-gray-400 mt-16">
            <div className="text-5xl mb-3">🔥</div>
            <p className="font-medium">No messages yet</p>
            <p className="text-sm mt-1">Send a message to start chatting</p>
            {(ragEnabled || streamEnabled) && (
              <p className="text-xs text-gray-400 mt-2 flex items-center justify-center gap-2">
                {ragEnabled && <span className="text-purple-500 font-medium">RAG</span>}
                {ragEnabled && rerankEnabled && <span className="text-green-500 font-medium">+ Rerank</span>}
                {streamEnabled && <span className="text-amber-500 font-medium">Stream</span>}
              </p>
            )}
          </div>
        )}
        {messages.map((msg, i) => {
          // Find the preceding user message for judge context
          const precedingUser = i > 0 && messages[i - 1].role === 'user'
            ? messages[i - 1].content
            : ''
          return (
            <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-xl ${msg.role === 'user' ? '' : 'w-full'}`}>
                <div className={`px-4 py-3 rounded-xl text-sm whitespace-pre-wrap leading-relaxed ${
                  msg.role === 'user' ? 'bg-blue-500 text-white' : 'bg-white border border-gray-200 text-gray-900 shadow-sm'
                }`}>
                  {msg.content}
                </div>
                {msg.role === 'assistant' && (
                  <div className="px-1">
                    {msg.sources && msg.sources.length > 0 && (
                      <SourcesPanel sources={msg.sources} />
                    )}
                    <RateThisPanel
                      messageId={msg.message_id}
                      question={precedingUser}
                      response={msg.content}
                      evalResult={msg.eval_result}
                      onEval={(result) => updateMessageEval(i, result)}
                    />
                  </div>
                )}
              </div>
            </div>
          )
        })}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-white border border-gray-200 px-4 py-3 rounded-xl shadow-sm">
              <Loader2 className="w-4 h-4 animate-spin text-gray-400" />
            </div>
          </div>
        )}
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-xl text-sm">
            <strong>Error:</strong> {error}
          </div>
        )}
      </div>

      {/* Metrics bar */}
      {lastResponse && (
        <div className="border-t border-gray-100 bg-gray-50 px-4 py-2 flex items-center justify-between text-xs text-gray-500">
          <span className="font-medium text-gray-700">{lastResponse.model}</span>
          <span>{lastResponse.usage.total_tokens} tokens</span>
          <span className="text-green-600 font-medium">${lastResponse.cost_usd.toFixed(6)}</span>
          <span>{lastResponse.latency_ms.toFixed(0)}ms</span>
        </div>
      )}

      {/* Input */}
      <div className="border-t border-gray-200 bg-white p-3">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage() } }}
            placeholder="Type your message… (Enter to send)"
            disabled={loading}
            className="flex-1 px-4 py-2.5 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100"
          />
          <button onClick={sendMessage} disabled={loading || !input.trim()}
            className="px-4 py-2.5 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-gray-300 transition-colors">
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
          </button>
        </div>
      </div>
    </div>
  )
}

// ─── RAG Tab ──────────────────────────────────────────────────────────────────

type SearchMode = 'hybrid' | 'semantic' | 'bm25'
type RAGPanel = 'ingest' | 'search' | 'documents'

function RAGTab() {
  const [panel, setPanel] = useState<RAGPanel>('ingest')

  const [docText, setDocText] = useState('')
  const [docSource, setDocSource] = useState('')
  const [ingestLoading, setIngestLoading] = useState(false)
  const [ingestResult, setIngestResult] = useState<Record<string, unknown> | null>(null)
  const [ingestError, setIngestError] = useState<string | null>(null)

  const [query, setQuery] = useState('')
  const [topK, setTopK] = useState(5)
  const [searchMode, setSearchMode] = useState<SearchMode>('hybrid')
  const [queryLoading, setQueryLoading] = useState(false)
  const [queryResult, setQueryResult] = useState<RAGQueryResponse | null>(null)
  const [queryError, setQueryError] = useState<string | null>(null)

  const [documents, setDocuments] = useState<RAGDocumentItem[]>([])
  const [docsTotal, setDocsTotal] = useState(0)
  const [docsLoading, setDocsLoading] = useState(false)
  const [docsError, setDocsError] = useState<string | null>(null)
  const [deletingId, setDeletingId] = useState<string | null>(null)

  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = (ev) => setDocText(ev.target?.result as string)
    reader.readAsText(file)
    setDocSource(file.name)
  }

  const handleIngest = async () => {
    if (!docText.trim()) return
    setIngestLoading(true)
    setIngestError(null)
    setIngestResult(null)

    try {
      const response = await fetch(`${API_URL}/v1/rag/ingest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          documents: [{ content: docText, metadata: { source: docSource || 'manual-input' } }]
        })
      })
      if (!response.ok) {
        const err = await response.json().catch(() => ({}))
        throw new Error(err.detail || `Error ${response.status}`)
      }
      const data = await response.json()
      setIngestResult(data)
      setDocText('')
      setDocSource('')
    } catch (err) {
      setIngestError(err instanceof Error ? err.message : 'Ingestion failed')
    } finally {
      setIngestLoading(false)
    }
  }

  const handleQuery = async () => {
    if (!query.trim()) return
    setQueryLoading(true)
    setQueryError(null)
    setQueryResult(null)

    try {
      const response = await fetch(`${API_URL}/v1/rag/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, top_k: topK, search_mode: searchMode })
      })
      if (!response.ok) {
        const err = await response.json().catch(() => ({}))
        throw new Error(err.detail || `Error ${response.status}`)
      }
      const data: RAGQueryResponse = await response.json()
      setQueryResult(data)
    } catch (err) {
      setQueryError(err instanceof Error ? err.message : 'Query failed')
    } finally {
      setQueryLoading(false)
    }
  }

  const loadDocuments = async () => {
    setDocsLoading(true)
    setDocsError(null)
    try {
      const res = await fetch(`${API_URL}/v1/rag/documents?limit=50`)
      if (!res.ok) throw new Error(`Error ${res.status}`)
      const data = await res.json()
      setDocuments(data.documents)
      setDocsTotal(data.total)
    } catch (err) {
      setDocsError(err instanceof Error ? err.message : 'Failed to load documents')
    } finally {
      setDocsLoading(false)
    }
  }

  const deleteDocument = async (id: string) => {
    setDeletingId(id)
    try {
      await fetch(`${API_URL}/v1/rag/documents/${id}`, { method: 'DELETE' })
      setDocuments(prev => prev.filter(d => d.id !== id))
      setDocsTotal(t => t - 1)
    } finally {
      setDeletingId(null)
    }
  }

  const searchModeColors: Record<SearchMode, string> = {
    hybrid: 'bg-purple-100 text-purple-700 border-purple-300',
    semantic: 'bg-blue-100 text-blue-700 border-blue-300',
    bm25: 'bg-orange-100 text-orange-700 border-orange-300',
  }

  const panelBtn = (id: RAGPanel, label: string, icon: React.ReactNode) => (
    <button
      key={id}
      onClick={() => { setPanel(id); if (id === 'documents') loadDocuments() }}
      className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold border transition-colors ${
        panel === id
          ? 'bg-blue-500 text-white border-blue-500'
          : 'bg-white text-gray-500 border-gray-300 hover:bg-blue-50 hover:text-blue-600 hover:border-blue-300'
      }`}
    >
      {icon}{label}
    </button>
  )

  return (
    <div className="flex flex-col h-full">
      {/* Panel selector */}
      <div className="flex items-center gap-2 p-3 border-b border-gray-100 bg-gray-50">
        {panelBtn('ingest', 'Ingest', <Upload className="w-3.5 h-3.5" />)}
        {panelBtn('search', 'Search', <Search className="w-3.5 h-3.5" />)}
        {panelBtn('documents', 'Documents', <List className="w-3.5 h-3.5" />)}
      </div>

      <div className="flex-1 overflow-auto p-4">

        {/* ── Ingest Panel ── */}
        {panel === 'ingest' && (
          <div className="max-w-lg">
            <div className="bg-white border border-gray-200 rounded-xl p-4 shadow-sm">
              <h3 className="font-semibold text-gray-800 flex items-center gap-2 mb-3">
                <Upload className="w-4 h-4 text-blue-500" /> Ingest Documents
              </h3>
              <input ref={fileInputRef} type="file" accept=".txt,.md,.pdf" onChange={handleFileUpload} className="hidden" />
              <button onClick={() => fileInputRef.current?.click()}
                className="w-full px-3 py-2 border-2 border-dashed border-gray-300 rounded-lg text-sm text-gray-500 hover:border-blue-400 hover:text-blue-500 transition-colors mb-2">
                <FileText className="w-4 h-4 inline mr-2" />Upload .txt / .md file
              </button>
              <textarea
                value={docText}
                onChange={(e) => setDocText(e.target.value)}
                placeholder="Or paste document text here…"
                rows={6}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <input
                type="text"
                value={docSource}
                onChange={(e) => setDocSource(e.target.value)}
                placeholder="Source label (optional)"
                className="w-full mt-2 px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <button onClick={handleIngest} disabled={ingestLoading || !docText.trim()}
                className="w-full mt-3 px-4 py-2 bg-blue-500 text-white rounded-lg text-sm hover:bg-blue-600 disabled:bg-gray-300 transition-colors flex items-center justify-center gap-2">
                {ingestLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
                {ingestLoading ? 'Ingesting…' : 'Ingest'}
              </button>
              {ingestResult && (
                <div className="mt-3 p-3 bg-green-50 border border-green-200 rounded-lg text-xs text-green-700 space-y-1">
                  <p className="font-semibold">✓ Ingested successfully</p>
                  <p>Chunks: <strong>{String(ingestResult.chunks_created)}</strong></p>
                  <p>BM25 indexed: <strong>{String(ingestResult.bm25_indexed)}</strong></p>
                  <p>Latency: <strong>{String(Number(ingestResult.total_latency_ms).toFixed(0))}ms</strong></p>
                </div>
              )}
              {ingestError && (
                <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg text-xs text-red-700">{ingestError}</div>
              )}
            </div>
          </div>
        )}

        {/* ── Search Panel ── */}
        {panel === 'search' && (
          <div className="flex flex-col gap-3">
            <div className="bg-white border border-gray-200 rounded-xl p-4 shadow-sm">
              <h3 className="font-semibold text-gray-800 flex items-center gap-2 mb-3">
                <Search className="w-4 h-4 text-purple-500" /> Search
              </h3>
              <div className="flex gap-2 mb-3">
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter') handleQuery() }}
                  placeholder="Ask a question about your documents…"
                  className="flex-1 px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-purple-500"
                />
                <button onClick={handleQuery} disabled={queryLoading || !query.trim()}
                  className="px-4 py-2 bg-purple-500 text-white rounded-lg text-sm hover:bg-purple-600 disabled:bg-gray-300 transition-colors flex items-center gap-2">
                  {queryLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
                </button>
              </div>
              <div className="flex items-center gap-3">
                <span className="text-xs text-gray-500 font-medium">Mode:</span>
                {(['hybrid', 'semantic', 'bm25'] as SearchMode[]).map(mode => (
                  <button key={mode} onClick={() => setSearchMode(mode)}
                    className={`px-3 py-1 rounded-full text-xs font-medium border transition-colors ${
                      searchMode === mode ? searchModeColors[mode] : 'bg-gray-50 text-gray-500 border-gray-200 hover:bg-gray-100'
                    }`}>
                    {mode === 'hybrid' ? '⚡ Hybrid' : mode === 'semantic' ? '🧠 Semantic' : '🔤 BM25'}
                  </button>
                ))}
                <div className="ml-auto flex items-center gap-2">
                  <span className="text-xs text-gray-500">Top K:</span>
                  <select value={topK} onChange={(e) => setTopK(Number(e.target.value))}
                    className="px-2 py-1 border border-gray-300 rounded text-xs bg-white focus:outline-none">
                    {[3, 5, 10].map(k => <option key={k} value={k}>{k}</option>)}
                  </select>
                </div>
              </div>
            </div>

            {queryError && (
              <div className="p-4 bg-red-50 border border-red-200 rounded-xl text-sm text-red-700">{queryError}</div>
            )}
            {queryResult && (
              <div className="flex flex-col gap-2">
                <div className="flex items-center justify-between px-1">
                  <span className="text-sm font-medium text-gray-700">
                    {queryResult.total_results} results
                    <span className={`ml-2 px-2 py-0.5 rounded-full text-xs ${searchModeColors[queryResult.search_mode as SearchMode]}`}>
                      {queryResult.search_mode}
                    </span>
                  </span>
                  <span className="text-xs text-gray-400">{queryResult.latency_ms.toFixed(0)}ms</span>
                </div>
                {queryResult.results.map((result, i) => (
                  <div key={result.id} className="bg-white border border-gray-200 rounded-xl p-4 shadow-sm">
                    <div className="flex items-start justify-between gap-2 mb-2">
                      <span className="text-xs font-semibold text-gray-400">#{i + 1}</span>
                      <div className="flex items-center gap-2 flex-shrink-0">
                        <span className="text-xs px-2 py-0.5 bg-gray-100 rounded-full font-mono text-gray-600">
                          {result.score.toFixed(4)}
                        </span>
                        {result.semantic_score != null && (
                          <span className="text-xs px-2 py-0.5 bg-blue-50 rounded-full text-blue-600">
                            sem: {result.semantic_score.toFixed(3)}
                          </span>
                        )}
                        {result.bm25_score != null && (
                          <span className="text-xs px-2 py-0.5 bg-orange-50 rounded-full text-orange-600">
                            bm25: {result.bm25_score.toFixed(3)}
                          </span>
                        )}
                      </div>
                    </div>
                    <p className="text-sm text-gray-800 leading-relaxed line-clamp-4">{result.content}</p>
                    {!!result.metadata.source && (
                      <p className="text-xs text-gray-400 mt-2">Source: {String(result.metadata.source)}</p>
                    )}
                  </div>
                ))}
                {queryResult.results.length === 0 && (
                  <div className="text-center py-8 text-gray-400 text-sm">
                    No results found. Try ingesting some documents first.
                  </div>
                )}
              </div>
            )}
            {!queryResult && !queryError && !queryLoading && (
              <div className="flex-1 flex items-center justify-center text-gray-400 py-16">
                <div className="text-center">
                  <Database className="w-10 h-10 mx-auto mb-3 opacity-30" />
                  <p className="text-sm">Ingest documents, then search</p>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── Documents Panel ── */}
        {panel === 'documents' && (
          <div className="flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-700">
                {docsTotal > 0 ? `${docsTotal} chunks` : 'Document chunks'}
              </span>
              <button onClick={loadDocuments} disabled={docsLoading}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-white border border-gray-300 rounded-lg text-xs text-gray-600 hover:bg-gray-50 transition-colors">
                {docsLoading ? <Loader2 className="w-3 h-3 animate-spin" /> : <RefreshCw className="w-3 h-3" />}
                Refresh
              </button>
            </div>

            {docsError && (
              <div className="p-3 bg-red-50 border border-red-200 rounded-xl text-sm text-red-700">{docsError}</div>
            )}

            {documents.length === 0 && !docsLoading && !docsError && (
              <div className="text-center py-16 text-gray-400">
                <Database className="w-10 h-10 mx-auto mb-3 opacity-30" />
                <p className="text-sm">No documents ingested yet</p>
              </div>
            )}

            {documents.map(doc => (
              <div key={doc.id} className="bg-white border border-gray-200 rounded-xl p-4 shadow-sm">
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0">
                    <p className="text-xs font-mono text-gray-400 mb-1 truncate">{doc.id}</p>
                    <p className="text-sm text-gray-800 leading-relaxed line-clamp-3">{doc.content}</p>
                    {!!doc.metadata.source && (
                      <p className="text-xs text-gray-400 mt-1">Source: {String(doc.metadata.source)}</p>
                    )}
                  </div>
                  <button
                    onClick={() => deleteDocument(doc.id)}
                    disabled={deletingId === doc.id}
                    title="Delete chunk"
                    className="flex-shrink-0 p-1.5 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded-lg transition-colors disabled:opacity-50"
                  >
                    {deletingId === doc.id ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <X className="w-3.5 h-3.5" />}
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

// ─── Stats Tab ─────────────────────────────────────────────────────────────────

function StatsTab() {
  const [cacheStats, setCacheStats] = useState<CacheStats | null>(null)
  const [ragStats, setRagStats] = useState<Record<string, unknown> | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchStats = async () => {
    setLoading(true)
    setError(null)
    try {
      const [cacheRes, ragRes] = await Promise.all([
        fetch(`${API_URL}/v1/cache/stats`),
        fetch(`${API_URL}/v1/rag/stats`)
      ])
      if (cacheRes.ok) setCacheStats(await cacheRes.json())
      if (ragRes.ok) setRagStats(await ragRes.json())
    } catch {
      setError('Failed to fetch stats')
    } finally {
      setLoading(false)
    }
  }

  const clearCache = async () => {
    await fetch(`${API_URL}/v1/cache/clear`, { method: 'POST' })
    fetchStats()
  }

  return (
    <div className="p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold text-gray-800 flex items-center gap-2">
          <BarChart3 className="w-4 h-4 text-green-500" /> System Stats
        </h3>
        <button onClick={fetchStats} disabled={loading}
          className="px-3 py-1.5 bg-green-500 text-white rounded-lg text-sm hover:bg-green-600 disabled:bg-gray-300 transition-colors flex items-center gap-2">
          {loading ? <Loader2 className="w-3 h-3 animate-spin" /> : null}
          Refresh
        </button>
      </div>

      {error && <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">{error}</div>}

      {/* Cache Stats */}
      <div className="bg-white border border-gray-200 rounded-xl p-4 shadow-sm">
        <h4 className="font-medium text-gray-700 mb-3">Semantic Cache</h4>
        {cacheStats ? (
          <div className="grid grid-cols-2 gap-3">
            <StatCard label="Status" value={cacheStats.enabled ? '✅ Active' : '❌ Disabled'} />
            <StatCard label="Backend" value={cacheStats.backend || '—'} />
            <StatCard label="Entries" value={String(cacheStats.total_entries ?? '—')} />
            <StatCard label="Hit Rate" value={cacheStats.hit_rate != null ? `${(cacheStats.hit_rate * 100).toFixed(1)}%` : '—'} />
            <StatCard label="Hits" value={String(cacheStats.hits ?? '—')} />
            <StatCard label="Misses" value={String(cacheStats.misses ?? '—')} />
          </div>
        ) : (
          <p className="text-sm text-gray-400">Click Refresh to load stats</p>
        )}
        {cacheStats?.enabled && (
          <button onClick={clearCache} className="mt-3 px-3 py-1.5 bg-red-50 text-red-600 border border-red-200 rounded-lg text-xs hover:bg-red-100 transition-colors">
            Clear Cache
          </button>
        )}
      </div>

      {/* RAG Stats */}
      <div className="bg-white border border-gray-200 rounded-xl p-4 shadow-sm">
        <h4 className="font-medium text-gray-700 mb-3">RAG Pipeline</h4>
        {ragStats ? (
          <div className="space-y-2">
            <StatCard label="Status" value={ragStats.enabled ? '✅ Active' : '❌ Disabled'} />
            {!!ragStats.enabled && (
              <>
                <StatCard label="Embedding Model" value={String(ragStats.embedding_model ?? '—')} />
                <StatCard label="Chunking Strategy" value={String(ragStats.chunking_strategy ?? '—')} />
                {ragStats.hybrid_search && typeof ragStats.hybrid_search === 'object' && (
                  <>
                    <StatCard label="Hybrid Search"
                      value={(ragStats.hybrid_search as Record<string, unknown>).enabled ? '✅ Enabled' : '❌ Disabled'} />
                    <StatCard label="Fusion Method"
                      value={String((ragStats.hybrid_search as Record<string, unknown>).fusion_method ?? '—')} />
                  </>
                )}
              </>
            )}
          </div>
        ) : (
          <p className="text-sm text-gray-400">Click Refresh to load stats</p>
        )}
      </div>
    </div>
  )
}

// ─── Evals Tab ────────────────────────────────────────────────────────────────

interface FeedbackRecord {
  id: string
  message_id: string
  question: string
  response: string
  rating?: string
  judge_result?: { score: number; reasoning: string; model: string }
  created_at: number
}

interface ABStats {
  experiment_id: string
  routing: string
  total_requests: number
  variants: {
    id: string
    requests: number
    avg_latency_ms: number
    avg_input_tokens: number
    avg_output_tokens: number
    avg_cost_usd: number
  }[]
}

function EvalsTab() {
  const [feedbacks, setFeedbacks] = useState<FeedbackRecord[]>([])
  const [feedbackTotal, setFeedbackTotal] = useState(0)
  const [feedbackLoading, setFeedbackLoading] = useState(false)

  const [experiments, setExperiments] = useState<{ id: string }[]>([])
  const [selectedExp, setSelectedExp] = useState<string>('')
  const [abStats, setAbStats] = useState<ABStats | null>(null)
  const [abLoading, setAbLoading] = useState(false)
  const [abError, setAbError] = useState<string | null>(null)

  const loadFeedback = async () => {
    setFeedbackLoading(true)
    try {
      const res = await fetch(`${API_URL}/v1/evals/feedback?limit=50`)
      if (!res.ok) return
      const data = await res.json()
      setFeedbacks(data.feedbacks)
      setFeedbackTotal(data.total)
    } finally {
      setFeedbackLoading(false)
    }
  }

  const loadExperiments = async () => {
    const res = await fetch(`${API_URL}/v1/evals/ab/experiments`)
    if (!res.ok) return
    const data = await res.json()
    setExperiments(data.experiments)
    if (data.experiments.length > 0 && !selectedExp) {
      setSelectedExp(data.experiments[0].id)
    }
  }

  const loadAbStats = async (expId: string) => {
    if (!expId) return
    setAbLoading(true)
    setAbError(null)
    try {
      const res = await fetch(`${API_URL}/v1/evals/ab/experiments/${expId}/stats`)
      if (!res.ok) throw new Error(`Error ${res.status}`)
      setAbStats(await res.json())
    } catch (err) {
      setAbError(err instanceof Error ? err.message : 'Failed to load stats')
    } finally {
      setAbLoading(false)
    }
  }

  const scoreColor = (score: number) =>
    score >= 7 ? 'text-green-600 bg-green-50 border-green-200'
    : score >= 4 ? 'text-amber-600 bg-amber-50 border-amber-200'
    : 'text-red-600 bg-red-50 border-red-200'

  return (
    <div className="flex gap-4 h-full p-4 overflow-auto">
      {/* Left: Feedback log */}
      <div className="w-1/2 flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold text-gray-800 flex items-center gap-2">
            <FlaskConical className="w-4 h-4 text-indigo-500" />
            Feedback Log
            {feedbackTotal > 0 && (
              <span className="text-xs text-gray-400 font-normal">({feedbackTotal})</span>
            )}
          </h3>
          <button onClick={loadFeedback} disabled={feedbackLoading}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-white border border-gray-300 rounded-lg text-xs text-gray-600 hover:bg-gray-50 transition-colors">
            {feedbackLoading ? <Loader2 className="w-3 h-3 animate-spin" /> : <RefreshCw className="w-3 h-3" />}
            Refresh
          </button>
        </div>

        {feedbacks.length === 0 && (
          <div className="text-center py-12 text-gray-400">
            <FlaskConical className="w-8 h-8 mx-auto mb-2 opacity-30" />
            <p className="text-sm">No feedback yet — rate responses in Chat</p>
          </div>
        )}

        <div className="space-y-2">
          {feedbacks.map(fb => (
            <div key={fb.id} className="bg-white border border-gray-200 rounded-xl p-3 shadow-sm">
              <div className="flex items-start justify-between gap-2 mb-1">
                <p className="text-xs text-gray-600 line-clamp-2 flex-1">{fb.question || '—'}</p>
                <div className="flex items-center gap-1 flex-shrink-0">
                  {fb.rating && (
                    <span className={`text-xs px-1.5 py-0.5 rounded border ${
                      fb.rating === 'thumbs_up' ? 'bg-green-50 text-green-600 border-green-200' : 'bg-red-50 text-red-600 border-red-200'
                    }`}>
                      {fb.rating === 'thumbs_up' ? '👍' : '👎'}
                    </span>
                  )}
                  {fb.judge_result && (
                    <span className={`text-xs px-1.5 py-0.5 rounded border font-mono ${scoreColor(fb.judge_result.score)}`}>
                      {fb.judge_result.score.toFixed(1)}
                    </span>
                  )}
                </div>
              </div>
              {fb.judge_result?.reasoning && (
                <p className="text-xs text-gray-400 line-clamp-2">{fb.judge_result.reasoning}</p>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Right: A/B Stats */}
      <div className="w-1/2 flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold text-gray-800 flex items-center gap-2">
            <BarChart3 className="w-4 h-4 text-indigo-500" /> A/B Experiments
          </h3>
          <button onClick={loadExperiments}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-white border border-gray-300 rounded-lg text-xs text-gray-600 hover:bg-gray-50 transition-colors">
            <RefreshCw className="w-3 h-3" /> Refresh
          </button>
        </div>

        {experiments.length > 0 && (
          <div className="flex items-center gap-2">
            <select
              value={selectedExp}
              onChange={(e) => { setSelectedExp(e.target.value); loadAbStats(e.target.value) }}
              className="flex-1 px-3 py-1.5 border border-gray-300 rounded-lg text-sm bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              {experiments.map(e => <option key={e.id} value={e.id}>{e.id}</option>)}
            </select>
            <button onClick={() => loadAbStats(selectedExp)} disabled={abLoading || !selectedExp}
              className="px-3 py-1.5 bg-indigo-500 text-white rounded-lg text-xs hover:bg-indigo-600 disabled:bg-gray-300 transition-colors">
              {abLoading ? <Loader2 className="w-3 h-3 animate-spin" /> : 'Load'}
            </button>
          </div>
        )}

        {experiments.length === 0 && (
          <div className="text-center py-12 text-gray-400">
            <BarChart3 className="w-8 h-8 mx-auto mb-2 opacity-30" />
            <p className="text-sm">No experiments yet</p>
            <p className="text-xs mt-1">POST /v1/evals/ab/experiments to create one</p>
          </div>
        )}

        {abError && <div className="p-3 bg-red-50 border border-red-200 rounded-xl text-sm text-red-700">{abError}</div>}

        {abStats && (
          <div className="space-y-3">
            <div className="flex items-center gap-3 text-xs text-gray-500">
              <span>Routing: <strong className="text-gray-700">{abStats.routing}</strong></span>
              <span>Total requests: <strong className="text-gray-700">{abStats.total_requests}</strong></span>
            </div>
            {abStats.variants.map(v => (
              <div key={v.id} className="bg-white border border-gray-200 rounded-xl p-4 shadow-sm">
                <p className="text-sm font-semibold text-gray-800 mb-3">{v.id}</p>
                <div className="grid grid-cols-2 gap-2">
                  <StatCard label="Requests" value={String(v.requests)} />
                  <StatCard label="Avg Latency" value={`${v.avg_latency_ms}ms`} />
                  <StatCard label="Avg Input Tokens" value={String(v.avg_input_tokens)} />
                  <StatCard label="Avg Cost" value={`$${v.avg_cost_usd.toFixed(6)}`} />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

// ─── App ──────────────────────────────────────────────────────────────────────

type Tab = 'chat' | 'rag' | 'stats' | 'evals'

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('chat')

  const tabs: { id: Tab; label: string; icon: React.ReactNode }[] = [
    { id: 'chat', label: 'Chat', icon: <Send className="w-4 h-4" /> },
    { id: 'rag', label: 'RAG', icon: <Database className="w-4 h-4" /> },
    { id: 'stats', label: 'Stats', icon: <BarChart3 className="w-4 h-4" /> },
    { id: 'evals', label: 'Evals', icon: <FlaskConical className="w-4 h-4" /> },
  ]

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-5xl mx-auto px-4 py-3 flex items-center justify-between">
          <div>
            <h1 className="text-lg font-bold text-gray-900">FAIForge</h1>
            <p className="text-xs text-gray-400">Production AI Boilerplate · v2.4</p>
          </div>
          <nav className="flex gap-1">
            {tabs.map(tab => (
              <button key={tab.id} onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  activeTab === tab.id
                    ? 'bg-blue-500 text-white shadow-sm'
                    : 'text-gray-600 hover:bg-gray-100'
                }`}>
                {tab.icon}{tab.label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      {/* Content */}
      <main className="flex-1 max-w-5xl mx-auto w-full flex flex-col" style={{ height: 'calc(100vh - 65px)' }}>
        {activeTab === 'chat' && <ChatTab />}
        {activeTab === 'rag' && <RAGTab />}
        {activeTab === 'stats' && <StatsTab />}
        {activeTab === 'evals' && <EvalsTab />}
      </main>
    </div>
  )
}
