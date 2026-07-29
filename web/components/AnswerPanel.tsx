"use client";

import {
  ArrowCounterClockwise,
  ArrowSquareOut,
  CheckCircle,
  CircleNotch,
  FileText,
  Globe,
  Sparkle,
  WarningCircle,
} from "@phosphor-icons/react";

import { RunDetails } from "./RunDetails";
import { SourceInspector } from "./SourceInspector";
import type {
  AnswerPayload,
  ChunkRecord,
  Citation,
  RetrievalResult,
  RunMetrics,
  StreamEvent,
} from "../lib/types";
import type { RunPhase } from "../lib/run-state";

type Props = {
  answer: AnswerPayload | null;
  events: StreamEvent[];
  retrievalResults: RetrievalResult[];
  metrics: RunMetrics | null;
  aggregateSuccessRate: number | null;
  isRunning: boolean;
  runId: string | null;
  runPhase: RunPhase;
  runError: string | null;
  selectedCitation: Citation | null;
  selectedChunk: ChunkRecord | null;
  sourceLoading: boolean;
  sourceError: string | null;
  onCitation: (citation: Citation) => void;
  onRetrievalResult: (result: RetrievalResult) => void;
  onCloseSource: () => void;
  onRetry: () => void;
};

export function AnswerPanel({
  answer,
  events,
  retrievalResults,
  metrics,
  aggregateSuccessRate,
  isRunning,
  runId,
  runPhase,
  runError,
  selectedCitation,
  selectedChunk,
  sourceLoading,
  sourceError,
  onCitation,
  onRetrievalResult,
  onCloseSource,
  onRetry,
}: Props) {
  const hasRunActivity = events.length > 0 || Boolean(metrics) || isRunning || Boolean(runError);
  const isGrounded = answer?.status === "grounded";

  return (
    <article className={`assistant-message is-${runPhase}`} aria-live="polite">
      <header className="assistant-message-header">
        <span className="assistant-avatar" aria-hidden="true"><Sparkle size={18} weight="fill" /></span>
        <strong>PKB Agent</strong>
        {isRunning ? <span className="assistant-live"><CircleNotch size={14} className="is-spinning" /> 正在思考</span> : null}
        {answer ? (
          <span className={`answer-status ${isGrounded ? "is-grounded" : "is-limited"}`}>
            {isGrounded ? <CheckCircle size={15} weight="fill" /> : <WarningCircle size={15} weight="fill" />}
            {isGrounded ? "已核验" : "证据不足"}
          </span>
        ) : null}
      </header>

      {answer ? (
        <div className="assistant-response">
          <p className="answer-body">{answer.answer}</p>

          {answer.claims.length ? (
            <div className="claim-list" aria-label="回答的证据声明">
              {answer.claims.map((claim, index) => (
                <article className={`claim-card is-${claim.kind}`} key={`${claim.text}-${index}`}>
                  <span>{claim.kind === "fact" ? "原文事实" : "模型推断"}</span>
                  <p>{claim.text}</p>
                  {claim.citations.length ? <small>{claim.citations.join(" · ")}</small> : null}
                </article>
              ))}
            </div>
          ) : null}

          {answer.citations.length ? (
            <section className="citation-section" aria-label="回答引用">
              <div className="inline-section-heading"><span>引用</span><small>{answer.citations.length} 个来源</small></div>
              <div className="citation-grid">
                {answer.citations.map((citation) => (
                  <button
                    className={`citation-card ${selectedCitation?.id === citation.id ? "is-selected" : ""}`}
                    key={citation.id}
                    onClick={() => onCitation(citation)}
                    type="button"
                    aria-pressed={selectedCitation?.id === citation.id}
                  >
                    {citation.source_type === "kb_chunk" ? <FileText size={17} weight="regular" aria-hidden="true" /> : <Globe size={17} weight="regular" aria-hidden="true" />}
                    <span><strong>{citation.title || citation.locator || "未命名来源"}</strong><small>{citation.excerpt || "查看原始证据"}</small></span>
                    <ArrowSquareOut size={16} weight="regular" aria-hidden="true" />
                  </button>
                ))}
              </div>
            </section>
          ) : null}
        </div>
      ) : (
        <ThinkingResponse isRunning={isRunning} />
      )}

      {runError ? (
        <section className="agent-run-error" role="alert">
          <WarningCircle size={20} weight="fill" aria-hidden="true" />
          <div><strong>本轮运行需要处理</strong><p>{runError}</p></div>
          <button className="retry-button" type="button" onClick={onRetry} disabled={isRunning}>
            <ArrowCounterClockwise size={16} weight="bold" aria-hidden="true" />
            重试
          </button>
        </section>
      ) : null}

      <SourceInspector
        citation={selectedCitation}
        chunk={selectedChunk}
        loading={sourceLoading}
        error={sourceError}
        onClose={onCloseSource}
      />

      {hasRunActivity ? (
        <RunDetails
          key={runId || "pending-run"}
          events={events}
          results={retrievalResults}
          metrics={metrics}
          aggregateSuccessRate={aggregateSuccessRate}
          isRunning={isRunning}
          onResult={onRetrievalResult}
        />
      ) : null}
    </article>
  );
}

function ThinkingResponse({ isRunning }: { isRunning: boolean }) {
  return (
    <section className="assistant-thinking">
      {isRunning ? (
        <><CircleNotch size={18} className="is-spinning" aria-hidden="true" /><p>正在从知识库中建立证据链…</p></>
      ) : (
        <p>准备好后，我会把回答、引用和运行过程收拢在这里。</p>
      )}
    </section>
  );
}
