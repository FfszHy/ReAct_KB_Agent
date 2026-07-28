"use client";

import { useCallback, useEffect, useMemo, useRef, useState, type ChangeEvent, type FormEvent } from "react";

import { AnswerPanel } from "../components/AnswerPanel";
import { ApprovalDialog } from "../components/ApprovalDialog";
import { KnowledgeSidebar } from "../components/KnowledgeSidebar";
import { RetrievalPanel } from "../components/RetrievalPanel";
import { RunMetrics } from "../components/RunMetrics";
import { SourceInspector } from "../components/SourceInspector";
import { TraceTimeline } from "../components/TraceTimeline";
import { apiFetch, sseUrl } from "../lib/api";
import type {
  AnswerPayload,
  Approval,
  ChunkRecord,
  Citation,
  DocumentRecord,
  RetrievalResult,
  RunMetrics as RunMetricsType,
  RunSnapshot,
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

const initialPrompt = "基于我的知识库，总结当前系统的关键设计决策，并说明它们之间的关系。";

export default function WorkbenchPage() {
  const streamRef = useRef<EventSource | null>(null);
  const seenEventSequences = useRef<Set<number>>(new Set());
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [documentCount, setDocumentCount] = useState(0);
  const [chunkCount, setChunkCount] = useState(0);
  const [query, setQuery] = useState(initialPrompt);
  const [runId, setRunId] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [events, setEvents] = useState<StreamEvent[]>([]);
  const [answer, setAnswer] = useState<AnswerPayload | null>(null);
  const [metrics, setMetrics] = useState<RunMetricsType | null>(null);
  const [aggregateSuccessRate, setAggregateSuccessRate] = useState<number | null>(null);
  const [approval, setApproval] = useState<Approval | null>(null);
  const [approvalPending, setApprovalPending] = useState(false);
  const [selectedCitation, setSelectedCitation] = useState<Citation | null>(null);
  const [selectedChunk, setSelectedChunk] = useState<ChunkRecord | null>(null);
  const [sourceLoading, setSourceLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [apiOnline, setApiOnline] = useState(true);

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
    void refreshWorkspace();
    return () => streamRef.current?.close();
  }, [refreshWorkspace]);

  const retrievalResults = useMemo(() => collectRetrievalResults(events), [events]);

  const handleStreamEvent = useCallback(
    (event: StreamEvent) => {
      if (event.sequence !== undefined) {
        if (seenEventSequences.current.has(event.sequence)) return;
        seenEventSequences.current.add(event.sequence);
      }
      setEvents((previous) => [...previous, event]);
      if (event.metrics) setMetrics(event.metrics);
      if (event.type === "approval_required" && event.approval) {
        setApproval(event.approval);
        setApprovalPending(false);
      }
      if (event.type === "approval_resolved") {
        setApproval((current) => (current?.id === event.approval_id ? null : current));
        setApprovalPending(false);
      }
      if (event.type === "answer") {
        setAnswer(event.answer_payload || null);
      }
      if (event.type === "error") {
        setRunError(event.error || "运行中出现异常");
      }
      if (event.type === "run_finished") {
        const snapshot = event.state as RunSnapshot | null | undefined;
        if (snapshot?.answer) setAnswer(snapshot.answer);
        if (snapshot?.metrics) setMetrics(snapshot.metrics);
        setIsRunning(false);
        setApproval(null);
        streamRef.current?.close();
        void refreshWorkspace();
      }
    },
    [refreshWorkspace],
  );

  async function handleRun(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const question = query.trim();
    if (!question || isRunning) return;
    streamRef.current?.close();
    setRunError(null);
    setRunId(null);
    seenEventSequences.current = new Set();
    setEvents([]);
    setAnswer(null);
    setMetrics(null);
    setApproval(null);
    setSelectedCitation(null);
    setSelectedChunk(null);
    setIsRunning(true);

    try {
      const run = await apiFetch<StartRunResponse>("/api/runs", {
        method: "POST",
        body: JSON.stringify({ question, user_id: "default" }),
      });
      setRunId(run.run_id);
      const stream = new EventSource(sseUrl(run.events_url));
      streamRef.current = stream;
      stream.onmessage = (message) => {
        try {
          handleStreamEvent(JSON.parse(message.data) as StreamEvent);
        } catch {
          setRunError("收到无法解析的运行事件");
        }
      };
      stream.onerror = () => {
        if (stream.readyState === EventSource.CLOSED) return;
        setRunError("实时连接暂时中断，正在尝试恢复…");
      };
    } catch (error) {
      setIsRunning(false);
      setRunError(error instanceof Error ? error.message : "无法启动本次运行");
    }
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
    if (!approval || !runId || approvalPending) return;
    setApprovalPending(true);
    try {
      await apiFetch(`/api/runs/${runId}/approvals/${approval.id}`, {
        method: "POST",
        body: JSON.stringify({ approved }),
      });
    } catch (error) {
      setApprovalPending(false);
      setRunError(error instanceof Error ? error.message : "提交审批失败");
    }
  }

  async function inspectCitation(citation: Citation) {
    setSelectedCitation(citation);
    setSelectedChunk(null);
    if (citation.source_type === "kb_chunk") {
      setSourceLoading(true);
      try {
        const response = await apiFetch<ChunkResponse>(`/api/chunks/${citation.source_id}`);
        setSelectedChunk(response.chunk);
      } catch (error) {
        setRunError(error instanceof Error ? error.message : "无法定位原始 chunk");
      } finally {
        setSourceLoading(false);
      }
    }
    window.setTimeout(() => document.getElementById("source-inspector")?.scrollIntoView({ behavior: "smooth", block: "center" }), 40);
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
        onUpload={handleUpload}
      />

      <section className="workbench-main">
        <header className="topbar">
          <div>
            <p className="eyebrow">KNOWLEDGE / CONTROL ROOM</p>
            <h2>让 Agent 的每一步都有据可查</h2>
          </div>
          <div className="topbar-status">
            <span className={`connection-status ${apiOnline ? "is-online" : "is-offline"}`}>
              {apiOnline ? "系统已连接" : "等待 API"}
            </span>
            <span className="run-id">{runId ? `RUN ${runId.slice(0, 8)}` : "尚未运行"}</span>
          </div>
        </header>

        <section className="question-stage">
          <div className="flow-rail" aria-label="主流程">
            <span className={documents.length ? "is-complete" : ""}>01 上传资料</span>
            <span className={documents.length ? "is-complete" : ""}>02 建立知识库</span>
            <span className={isRunning ? "is-active" : answer ? "is-complete" : ""}>03 提问并核验</span>
          </div>
          <form className="question-form" onSubmit={handleRun}>
            <label htmlFor="question">你想基于资料确认什么？</label>
            <textarea
              id="question"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="例如：这些设计决策的取舍是什么？"
              rows={3}
              disabled={isRunning}
            />
            <div className="question-actions">
              <div className="question-hint">
                <span>↳</span> 回答中的每个引用都可定位至原始 chunk
              </div>
              <button className="button button-primary ask-button" type="submit" disabled={!query.trim() || isRunning}>
                {isRunning ? "Agent 正在推理" : "开始提问"} <span>→</span>
              </button>
            </div>
          </form>
          {runError ? <p className="run-error">{runError}</p> : null}
        </section>

        <div className="workbench-grid">
          <div className="answer-column">
            <AnswerPanel answer={answer} isRunning={isRunning} onCitation={(citation) => void inspectCitation(citation)} />
            <RetrievalPanel results={retrievalResults} onResult={inspectRetrievalResult} />
            <SourceInspector
              citation={selectedCitation}
              chunk={selectedChunk}
              loading={sourceLoading}
              onClose={() => {
                setSelectedCitation(null);
                setSelectedChunk(null);
              }}
            />
          </div>
          <aside className="observability-column">
            <RunMetrics metrics={metrics} aggregateSuccessRate={aggregateSuccessRate} isRunning={isRunning} />
            <TraceTimeline events={events} isRunning={isRunning} />
          </aside>
        </div>
      </section>

      <ApprovalDialog approval={approval} pending={approvalPending} onDecision={handleApproval} />
    </main>
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
