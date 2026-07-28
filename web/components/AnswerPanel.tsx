"use client";

import type { AnswerPayload, Citation } from "../lib/types";

type Props = {
  answer: AnswerPayload | null;
  isRunning: boolean;
  onCitation: (citation: Citation) => void;
};

export function AnswerPanel({ answer, isRunning, onCitation }: Props) {
  if (!answer) {
    return (
      <section className="answer-panel empty-answer">
        <div className="empty-answer-orbit" aria-hidden="true">
          <span>01</span>
          <span>02</span>
          <span>03</span>
        </div>
        <p className="eyebrow">GROUNDED ANSWERS</p>
        <h2>{isRunning ? "Agent 正在建立证据链" : "从一条问题开始"}</h2>
        <p>
          {isRunning
            ? "计划、检索、工具结果和最终回答会在右侧实时呈现。"
            : "上传资料后提问。工作台会把回答、证据、审批和运行成本收拢到同一条可回放流程中。"}
        </p>
      </section>
    );
  }

  const isGrounded = answer.status === "grounded";
  return (
    <section className="answer-panel" aria-live="polite">
      <div className="answer-heading">
        <div>
          <p className="eyebrow">FINAL / VERIFIED</p>
          <h2>带证据的回答</h2>
        </div>
        <span className={`answer-status ${isGrounded ? "is-grounded" : "is-limited"}`}>
          {isGrounded ? "已核验" : "证据不足"}
        </span>
      </div>

      <p className="answer-body">{answer.answer}</p>

      {answer.claims.length ? (
        <div className="claim-grid">
          {answer.claims.map((claim, index) => (
            <article className={`claim-card is-${claim.kind}`} key={`${claim.text}-${index}`}>
              <span>{claim.kind === "fact" ? "原文事实" : "模型推断"}</span>
              <p>{claim.text}</p>
              <small>{claim.citations.join(" · ")}</small>
            </article>
          ))}
        </div>
      ) : null}

      {answer.citations.length ? (
        <div className="citation-section">
          <div className="section-heading">
            <span>引用证据</span>
            <span>{answer.citations.length}</span>
          </div>
          <div className="citation-grid">
            {answer.citations.map((citation, index) => (
              <button
                className="citation-card"
                key={citation.id}
                onClick={() => onCitation(citation)}
                type="button"
              >
                <span className="citation-index">{String(index + 1).padStart(2, "0")}</span>
                <span className="citation-type">
                  {citation.source_type === "kb_chunk" ? "知识库 chunk" : "网页证据"}
                </span>
                <strong>{citation.title || citation.locator || "未命名来源"}</strong>
                <p>{citation.excerpt || "查看原始证据"}</p>
                <span className="citation-action">定位原文 <b>↗</b></span>
              </button>
            ))}
          </div>
        </div>
      ) : null}
    </section>
  );
}
