"use client";

import type { ReactNode } from "react";

import type { ChunkRecord, Citation } from "../lib/types";

type Props = {
  citation: Citation | null;
  chunk: ChunkRecord | null;
  loading: boolean;
  onClose: () => void;
};

export function SourceInspector({ citation, chunk, loading, onClose }: Props) {
  if (!citation) return null;
  const text = chunk?.content || citation.excerpt || "此证据没有可展示的文本。";
  const freshness = freshnessLabel(citation);
  const locator = citation.locator || chunk?.document?.source_uri;

  return (
    <section className="source-inspector" id="source-inspector" aria-label="原始证据定位">
      <div className="source-inspector-header">
        <div>
          <p className="eyebrow">SOURCE / LOCATED</p>
          <h2>{citation.title || chunk?.document?.title || "原始证据"}</h2>
        </div>
        <button className="icon-button" type="button" onClick={onClose} aria-label="关闭原始证据">
          ×
        </button>
      </div>

      <div className="source-meta-row">
        <span>{citation.source_type === "kb_chunk" ? "知识库原始 chunk" : "网页快照"}</span>
        <span className={`freshness ${freshness.kind}`}>{freshness.label}</span>
      </div>
      <p className="source-location">
        {citation.source_type === "kb_chunk"
          ? `定位：Chunk #${chunk?.chunk_index ?? citation.metadata?.chunk_index ?? "—"}`
          : "定位：本轮抓取的网页快照"}
        {locator ? " · " : ""}
        {locator ? <SourceLink href={locator} /> : null}
      </p>

      <article className="source-text">
        {loading ? <span className="source-loading">正在定位原始文本…</span> : highlightEvidence(text, citation.excerpt)}
      </article>
      <p className="source-note">高亮部分是本回答实际引用的原文。引用 ID：{citation.id}</p>
    </section>
  );
}

function SourceLink({ href }: { href: string }) {
  if (!href.startsWith("http")) return <span>{href}</span>;
  return (
    <a href={href} target="_blank" rel="noreferrer">
      打开来源 ↗
    </a>
  );
}

function highlightEvidence(text: string, excerpt?: string | null): ReactNode {
  const needle = (excerpt || "").replace(/…$/, "").trim();
  if (!needle || needle.length < 14) return text;
  const index = text.indexOf(needle);
  if (index < 0) return text;
  return (
    <>
      {text.slice(0, index)}
      <mark>{text.slice(index, index + needle.length)}</mark>
      {text.slice(index + needle.length)}
    </>
  );
}

function freshnessLabel(citation: Citation): { label: string; kind: string } {
  const metadata = citation.metadata || {};
  if (metadata.expired === true) {
    const expires = new Date(String(metadata.expires_at || ""));
    const days = Number.isNaN(expires.getTime())
      ? null
      : Math.max(1, Math.floor((Date.now() - expires.getTime()) / 86_400_000));
    return { label: days ? `该网页已过期 ${days} 天` : "该网页已过期", kind: "is-expired" };
  }
  if (citation.source_type === "web_page") return { label: "网页快照仍在有效期", kind: "is-fresh" };
  return { label: "已锁定本次引用版本", kind: "is-fresh" };
}
