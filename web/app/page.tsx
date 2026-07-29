"use client";

import {
  ArrowUp,
  CaretDown,
  ChatCircleDots,
  CircleNotch,
  Database,
  FileArrowUp,
  FileText,
  MagnifyingGlass,
  Paperclip,
  Sparkle,
} from "@phosphor-icons/react";
import {
  useCallback,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  useState,
  type ChangeEvent,
  type FormEvent,
  type KeyboardEvent,
  type RefObject,
} from "react";

import { AnswerPanel } from "../components/AnswerPanel";
import { ApprovalDialog } from "../components/ApprovalDialog";
import { KnowledgeSidebar, uploadAccept } from "../components/KnowledgeSidebar";
import { apiFetch, sseUrl } from "../lib/api";
import {
  initialRunUiState,
  isRunActive,
  runUiReducer,
} from "../lib/run-state";
import type {
  ChunkRecord,
  Citation,
  DocumentRecord,
  RetrievalResult,
  StreamEvent,
} from "../lib/types";

type KnowledgeBaseResponse = {
  document_count: number;
  chunk_count: number;
};

type DocumentsResponse = { items: DocumentRecord[] };
type MetricsResponse = { success_rate: number | null };
type StartRunResponse = { run_id: string; events_url: string };
type UploadResponse = { document: DocumentRecord };
type ChunkResponse = { chunk: ChunkRecord };
type Surface = "chat" | "workspace";

export default function WorkbenchPage() {
  const streamRef = useRef<EventSource | null>(null);
  const activeRunRef = useRef<string | null>(null);
  const seenEventSequences = useRef<Set<number>>(new Set());
  const sourceRequestRef = useRef(0);
  const composerRef = useRef<HTMLTextAreaElement>(null);
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [documentCount, setDocumentCount] = useState(0);
  const [chunkCount, setChunkCount] = useState(0);
  const [query, setQuery] = useState("");
  const [lastQuestion, setLastQuestion] = useState<string | null>(null);
  const [activeSurface, setActiveSurface] = useState<Surface>("chat");
  const [aggregateSuccessRate, setAggregateSuccessRate] = useState<number | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [apiOnline, setApiOnline] = useState(true);
  const [runState, dispatch] = useReducer(runUiReducer, initialRunUiState);
  const isRunning = isRunActive(runState.phase);
  const hasConversation = Boolean(
    lastQuestion || runState.runId || runState.events.length || runState.answer || runState.runError,
  );

  const refreshWorkspace = useCallback(async () => {
    try {
      const [knowledgeBase, documentResponse, metricResponse] = await Promise.all([
        apiFetch<KnowledgeBaseResponse>("/api/knowledge-base"),
        apiFetch<DocumentsResponse>("/api/documents"),
        apiFetch<MetricsResponse>("/api/metrics"),
      ]);
      setDocumentCount(knowledgeBase.document_count);
      setChunkCount(knowledgeBase.chunk_count);
      setDocuments(documentResponse.items);
      setAggregateSuccessRate(metricResponse.success_rate);
      setApiOnline(true);
    } catch {
      setApiOnline(false);
    }
  }, []);

  useEffect(() => {
    const initialLoad = window.setTimeout(() => {
      void refreshWorkspace();
    }, 0);
    return () => {
      window.clearTimeout(initialLoad);
      activeRunRef.current = null;
      streamRef.current?.close();
    };
  }, [refreshWorkspace]);

  const retrievalResults = useMemo(() => collectRetrievalResults(runState.events), [runState.events]);

  const handleStreamEvent = useCallback((event: StreamEvent) => {
    if (event.sequence !== undefined) {
      if (seenEventSequences.current.has(event.sequence)) return;
      seenEventSequences.current.add(event.sequence);
    }
    dispatch({ type: "stream_event", event });
  }, []);

  const startRun = useCallback(
    async (question: string) => {
      if (!question || activeRunRef.current) return;
      streamRef.current?.close();
      activeRunRef.current = null;
      seenEventSequences.current = new Set();
      sourceRequestRef.current += 1;
      setLastQuestion(question);
      setActiveSurface("chat");
      dispatch({ type: "run_requested" });

      try {
        const run = await apiFetch<StartRunResponse>("/api/runs", {
          method: "POST",
          body: JSON.stringify({ question, user_id: "default" }),
        });
        activeRunRef.current = run.run_id;
        dispatch({ type: "run_started", runId: run.run_id });

        const stream = new EventSource(sseUrl(run.events_url));
        streamRef.current = stream;
        stream.onopen = () => dispatch({ type: "stream_connected" });
        stream.onmessage = (message) => {
          if (activeRunRef.current !== run.run_id) return;
          try {
            const incoming = JSON.parse(message.data) as StreamEvent;
            handleStreamEvent(incoming);
            if (incoming.type === "run_finished") {
              activeRunRef.current = null;
              stream.close();
              void refreshWorkspace();
            }
          } catch {
            activeRunRef.current = null;
            stream.close();
            dispatch({ type: "stream_failed", error: "收到无法解析的运行事件" });
            void refreshWorkspace();
          }
        };
        stream.onerror = () => {
          if (activeRunRef.current !== run.run_id) return;
          if (stream.readyState === EventSource.CLOSED) {
            activeRunRef.current = null;
            dispatch({ type: "stream_failed", error: "实时连接已关闭，未收到任务完成事件" });
            void refreshWorkspace();
            return;
          }
          dispatch({ type: "stream_reconnecting" });
        };
      } catch (error) {
        activeRunRef.current = null;
        dispatch({
          type: "run_start_failed",
          error: error instanceof Error ? error.message : "无法启动本次运行",
        });
      }
    },
    [handleStreamEvent, refreshWorkspace],
  );

  function handleRun(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const question = query.trim();
    if (!question || isRunning) return;
    void startRun(question);
  }

  function handleComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key !== "Enter" || event.shiftKey || event.nativeEvent.isComposing) return;
    event.preventDefault();
    event.currentTarget.form?.requestSubmit();
  }

  function handleRetry() {
    const question = lastQuestion || query.trim();
    if (question && !isRunning) void startRun(question);
  }

  function startNewConversation() {
    if (isRunning) return;
    streamRef.current?.close();
    activeRunRef.current = null;
    seenEventSequences.current = new Set();
    sourceRequestRef.current += 1;
    setLastQuestion(null);
    setQuery("");
    setActiveSurface("chat");
    dispatch({ type: "reset" });
    window.setTimeout(() => composerRef.current?.focus(), 0);
  }

  function useSuggestion(suggestion: string) {
    setQuery(suggestion);
    window.setTimeout(() => composerRef.current?.focus(), 0);
  }

  async function handleUpload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    setUploadError(null);
    setIsUploading(true);
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("user_id", "default");
      await apiFetch<UploadResponse>("/api/documents/upload", { method: "POST", body: form });
      await refreshWorkspace();
    } catch (error) {
      setUploadError(error instanceof Error ? error.message : "资料上传失败");
    } finally {
      setIsUploading(false);
    }
  }

  async function handleApproval(approved: boolean) {
    if (!runState.approval || !runState.runId || runState.approvalPending) return;
    dispatch({ type: "approval_pending", value: true });
    try {
      await apiFetch(`/api/runs/${runState.runId}/approvals/${runState.approval.id}`, {
        method: "POST",
        body: JSON.stringify({ approved }),
      });
    } catch (error) {
      dispatch({
        type: "approval_failed",
        error: error instanceof Error ? error.message : "提交审批失败",
      });
    }
  }

  async function inspectCitation(citation: Citation) {
    const requestId = sourceRequestRef.current + 1;
    sourceRequestRef.current = requestId;
    dispatch({ type: "source_selected", citation });
    if (citation.source_type !== "kb_chunk") return;
    try {
      const response = await apiFetch<ChunkResponse>(`/api/chunks/${citation.source_id}`);
      if (sourceRequestRef.current === requestId) {
        dispatch({ type: "source_loaded", chunk: response.chunk });
      }
    } catch (error) {
      if (sourceRequestRef.current === requestId) {
        dispatch({
          type: "source_failed",
          error: error instanceof Error ? error.message : "无法定位原始 chunk",
        });
      }
    }
  }

  function inspectRetrievalResult(result: RetrievalResult) {
    void inspectCitation({
      id: `kb:${result.chunk_id}`,
      source_type: "kb_chunk",
      source_id: result.chunk_id,
      title: result.title,
      locator: result.source_uri,
      excerpt: result.content_preview,
      metadata: { document_id: result.document_id, chunk_index: result.chunk_index, score: result.score },
    });
  }

  return (
    <main className="workbench-shell">
      <KnowledgeSidebar
        documents={documents}
        documentCount={documentCount}
        chunkCount={chunkCount}
        isUploading={isUploading}
        uploadError={uploadError}
        activeSurface={activeSurface}
        isRunning={isRunning}
        onNewConversation={startNewConversation}
        onFocusComposer={() => {
          setActiveSurface("chat");
          window.setTimeout(() => composerRef.current?.focus(), 0);
        }}
        onSurfaceChange={setActiveSurface}
        onUpload={handleUpload}
      />

      <section className="workbench-main">
        <header className="app-header">
          <div className="header-spacer" aria-hidden="true" />
          <div className="surface-switch" role="tablist" aria-label="工作区视图">
            <button
              className={activeSurface === "chat" ? "is-active" : ""}
              type="button"
              role="tab"
              aria-selected={activeSurface === "chat"}
              onClick={() => setActiveSurface("chat")}
            >
              <ChatCircleDots size={18} weight="regular" aria-hidden="true" />
              对话
            </button>
            <button
              className={activeSurface === "workspace" ? "is-active" : ""}
              type="button"
              role="tab"
              aria-selected={activeSurface === "workspace"}
              onClick={() => setActiveSurface("workspace")}
            >
              <Database size={18} weight="regular" aria-hidden="true" />
              资料
            </button>
          </div>
          <div className={`api-presence ${apiOnline ? "is-online" : "is-offline"}`}>
            <CircleNotch size={22} weight="regular" aria-hidden="true" />
            <span className="sr-only">{apiOnline ? "系统已连接" : "等待 API"}</span>
          </div>
        </header>

        {activeSurface === "workspace" ? (
          <WorkspaceSurface
            documents={documents}
            documentCount={documentCount}
            chunkCount={chunkCount}
            isUploading={isUploading}
            uploadError={uploadError}
            onUpload={handleUpload}
          />
        ) : (
          <section className={`chat-surface ${hasConversation ? "has-conversation" : "is-welcome"}`}>
            {hasConversation ? (
              <div className="conversation-thread" aria-label="Agent 对话">
                {lastQuestion ? <article className="user-message"><p>{lastQuestion}</p></article> : null}
                <AnswerPanel
                  answer={runState.answer}
                  events={runState.events}
                  retrievalResults={retrievalResults}
                  metrics={runState.metrics}
                  aggregateSuccessRate={aggregateSuccessRate}
                  isRunning={isRunning}
                  runId={runState.runId}
                  runPhase={runState.phase}
                  runError={runState.runError}
                  selectedCitation={runState.selectedCitation}
                  selectedChunk={runState.selectedChunk}
                  sourceLoading={runState.sourceLoading}
                  sourceError={runState.sourceError}
                  onCitation={(citation) => void inspectCitation(citation)}
                  onRetrievalResult={inspectRetrievalResult}
                  onCloseSource={() => dispatch({ type: "source_closed" })}
                  onRetry={handleRetry}
                />
              </div>
            ) : (
              <WelcomePanel
                documentCount={documentCount}
              />
            )}

            <ChatComposer
              composerRef={composerRef}
              query={query}
              isRunning={isRunning}
              isUploading={isUploading}
              uploadError={uploadError}
              documentCount={documentCount}
              hasConversation={hasConversation}
              onChange={setQuery}
              onKeyDown={handleComposerKeyDown}
              onSubmit={handleRun}
              onUpload={handleUpload}
            />
            {!hasConversation ? <SuggestionList onSuggestion={useSuggestion} /> : null}
          </section>
        )}
      </section>

      <ApprovalDialog
        approval={runState.approval}
        pending={runState.approvalPending}
        error={runState.approvalError}
        onDecision={handleApproval}
      />
    </main>
  );
}

function WelcomePanel({
  documentCount,
}: {
  documentCount: number;
}) {
  return (
    <div className="welcome-panel">
      <div className="welcome-mark" aria-hidden="true"><Sparkle size={28} weight="fill" /></div>
      <h1>你好，我是 PKB Agent。</h1>
      <p>
        {documentCount
          ? `已连接 ${documentCount} 份资料。问我一个问题，我会把答案和证据一起带回来。`
          : "先上传一份资料，或直接开始提问。每个结论都能回到它的原始证据。"}
      </p>
    </div>
  );
}

function SuggestionList({ onSuggestion }: { onSuggestion: (suggestion: string) => void }) {
  return (
    <div className="suggestion-list" aria-label="提问建议">
      <button type="button" onClick={() => onSuggestion("总结我的知识库中最重要的结论，并列出对应证据。")}>
        <FileText size={20} weight="regular" aria-hidden="true" /> 总结资料中的关键结论
      </button>
      <button type="button" onClick={() => onSuggestion("找出资料中相互矛盾或需要进一步确认的内容。")}>
        <Sparkle size={20} weight="regular" aria-hidden="true" /> 找出需要核验的内容
      </button>
      <button type="button" onClick={() => onSuggestion("用通俗的语言解释这份资料的核心设计取舍。")}>
        <MagnifyingGlass size={20} weight="regular" aria-hidden="true" /> 解释核心设计取舍
      </button>
    </div>
  );
}

function ChatComposer({
  composerRef,
  query,
  isRunning,
  isUploading,
  uploadError,
  documentCount,
  hasConversation,
  onChange,
  onKeyDown,
  onSubmit,
  onUpload,
}: {
  composerRef: RefObject<HTMLTextAreaElement | null>;
  query: string;
  isRunning: boolean;
  isUploading: boolean;
  uploadError: string | null;
  documentCount: number;
  hasConversation: boolean;
  onChange: (value: string) => void;
  onKeyDown: (event: KeyboardEvent<HTMLTextAreaElement>) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  onUpload: (event: ChangeEvent<HTMLInputElement>) => void;
}) {
  return (
    <div className={`composer-dock ${hasConversation ? "is-conversation" : ""}`}>
      <form className="chat-composer" onSubmit={onSubmit}>
        <div className="composer-shell">
          <label className="composer-attach" title="上传资料">
            <input type="file" accept={uploadAccept} onChange={onUpload} disabled={isUploading || isRunning} />
            <Paperclip size={21} weight="regular" aria-hidden="true" />
            <span className="sr-only">上传资料</span>
          </label>
          <textarea
            ref={composerRef}
            value={query}
            onChange={(event) => onChange(event.target.value)}
            onKeyDown={onKeyDown}
            placeholder="问问你的知识库"
            rows={1}
            disabled={isRunning}
            aria-label="向 PKB Agent 提问"
          />
          <div className="composer-controls">
            <span className="composer-context" title="当前知识库上下文">
              <Database size={16} weight="regular" aria-hidden="true" />
              {documentCount ? `${documentCount} 份资料` : "知识库"}
              <CaretDown size={14} weight="bold" aria-hidden="true" />
            </span>
            <button
              className="composer-send"
              type="submit"
              aria-label={isRunning ? "Agent 正在运行" : "发送问题"}
              disabled={!query.trim() || isRunning}
            >
              {isRunning ? <CircleNotch size={19} weight="bold" className="is-spinning" /> : <ArrowUp size={19} weight="bold" />}
            </button>
          </div>
        </div>
      </form>
      {isUploading ? <p className="composer-status"><CircleNotch size={15} className="is-spinning" /> 正在上传并建立索引…</p> : null}
      {uploadError ? <p className="composer-status is-error">{uploadError}</p> : null}
      <p className="composer-disclaimer">PKB Agent 可能会出错，请核对引用与原文。</p>
    </div>
  );
}

function WorkspaceSurface({
  documents,
  documentCount,
  chunkCount,
  isUploading,
  uploadError,
  onUpload,
}: {
  documents: DocumentRecord[];
  documentCount: number;
  chunkCount: number;
  isUploading: boolean;
  uploadError: string | null;
  onUpload: (event: ChangeEvent<HTMLInputElement>) => void;
}) {
  return (
    <section className="workspace-surface" aria-label="知识库资料">
      <header>
        <p>知识库</p>
        <h1>把资料变成可追溯的回答</h1>
        <span>上传后会自动分块、索引，并在每次回答中保留证据链。</span>
      </header>
      <div className="workspace-stat-row">
        <article><strong>{documentCount}</strong><span>份资料</span></article>
        <article><strong>{chunkCount}</strong><span>个证据片段</span></article>
        <article><strong>FastAPI + SSE</strong><span>实时运行通道</span></article>
      </div>
      <label className={`workspace-upload ${isUploading ? "is-uploading" : ""}`}>
        <input type="file" accept={uploadAccept} onChange={onUpload} disabled={isUploading} />
        <FileArrowUp size={23} weight="regular" aria-hidden="true" />
        <span><strong>{isUploading ? "正在建立索引…" : "上传资料"}</strong><small>PDF、Markdown、文本、网页与代码文件</small></span>
      </label>
      {uploadError ? <p className="workspace-error">{uploadError}</p> : null}
      <div className="workspace-documents">
        <div className="workspace-list-heading"><span>最近资料</span><span>{documents.length}</span></div>
        {documents.length ? documents.map((document) => (
          <article className="workspace-document" key={document.id}>
            <FileText size={20} weight="regular" aria-hidden="true" />
            <div><strong>{document.title || "未命名资料"}</strong><span>{document.version || "v—"} · {document.chunk_count ?? 0} 个片段</span></div>
            <em className={document.source_status === "expired" ? "is-expired" : ""}>
              {document.source_status === "expired" ? document.source_label || "已过期" : "已同步"}
            </em>
          </article>
        )) : (
          <div className="workspace-empty"><FileText size={23} weight="light" /><p>还没有资料。上传第一份资料后，就可以在对话中提问。</p></div>
        )}
      </div>
    </section>
  );
}

function collectRetrievalResults(events: StreamEvent[]): RetrievalResult[] {
  const seen = new Set<string>();
  const results: RetrievalResult[] = [];
  for (const event of events) {
    if (event.type !== "tool_result" || event.tool !== "rag_search" || !event.observation) continue;
    try {
      const payload = JSON.parse(event.observation) as { results?: RetrievalResult[] };
      for (const result of payload.results || []) {
        if (result.chunk_id && !seen.has(result.chunk_id)) {
          seen.add(result.chunk_id);
          results.push(result);
        }
      }
    } catch {
      // A deliberately truncated observation remains useful to the Agent but
      // cannot be expanded as a structured retrieval list in the browser.
    }
  }
  return results;
}
