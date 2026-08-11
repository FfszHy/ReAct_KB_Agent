"use client";

import { useEffect, useRef, type ReactNode } from "react";
import { useGSAP } from "@gsap/react";
import gsap from "gsap";
import { ArrowSquareOut, CircleNotch, FileText, Globe, WarningCircle, X } from "@phosphor-icons/react";

import type { ChunkRecord, Citation } from "../lib/types";

gsap.registerPlugin(useGSAP);

type Props = {
  citation: Citation | null;
  chunk: ChunkRecord | null;
  loading: boolean;
  error?: string | null;
  onClose: () => void;
};

export function SourceInspector({ citation, chunk, loading, error, onClose }: Props) {
  const root = useRef<HTMLElement>(null);
  const citationId = citation?.id;

  useEffect(() => {
    if (!citationId || !root.current) return;
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    root.current.focus({ preventScroll: true });
    root.current.scrollIntoView({ behavior: reducedMotion ? "auto" : "smooth", block: "nearest" });
  }, [citationId]);

  useGSAP(
    () => {
      if (!citation || !root.current) return;
      const media = gsap.matchMedia();
      media.add("(prefers-reduced-motion: no-preference)", () => {
        gsap.fromTo(root.current, { autoAlpha: 0, y: 8, scale: 0.99 }, { autoAlpha: 1, y: 0, scale: 1, duration: 0.2, ease: "power3.out" });
      });
      return () => media.revert();
    },
    { scope: root, dependencies: [citation?.id], revertOnUpdate: true },
  );

  if (!citation) return null;
  const isKnowledgeBaseCitation = citation.source_type === "kb_chunk" || citation.source_type === "kb_document" || citation.source_type === "kb_catalog";
  const text = chunk?.content || citation.excerpt || "此证据没有可展示的文本。";
  const freshness = freshnessLabel(citation);
  const locator = citation.locator || chunk?.document?.source_uri;

  return (
    <section className="source-inspector" ref={root} tabIndex={-1} aria-label="回答中的原始证据定位">
      <div className="source-inspector-header">
        <div>
          <span className="source-kicker">{isKnowledgeBaseCitation ? <FileText size={16} weight="regular" /> : <Globe size={16} weight="regular" />} 原始证据</span>
          <h3>{citation.title || chunk?.document?.title || "原始证据"}</h3>
        </div>
        <button className="icon-button" type="button" onClick={onClose} aria-label="关闭原始证据"><X size={18} weight="regular" /></button>
      </div>

      <div className="source-meta-row">
        <span>{citation.source_type === "kb_chunk" ? `Chunk #${chunk?.chunk_index ?? citation.metadata?.chunk_index ?? "—"}` : citation.source_type === "kb_document" || citation.source_type === "kb_catalog" ? "知识库资料目录" : "本轮网页快照"}</span>
        <span className={`freshness ${freshness.kind}`}>{freshness.label}</span>
      </div>
      {locator ? <p className="source-location"><SourceLink href={locator} /></p> : null}

      <article className="source-text">
        {loading ? <span className="source-loading"><CircleNotch size={16} className="is-spinning" /> 正在定位原始文本…</span> : error ? <span className="source-error"><WarningCircle size={16} weight="fill" /> {error}</span> : highlightEvidence(text, citation.excerpt)}
      </article>
      <p className="source-note">{citation.source_type === "kb_document" || citation.source_type === "kb_catalog" ? "此条信息来自本轮读取的知识库资料目录。" : "高亮部分是本回答实际引用的原文。"}</p>
    </section>
  );
}

function SourceLink({ href }: { href: string }) {
  if (!href.startsWith("http")) return <span>{href}</span>;
  return <a href={href} target="_blank" rel="noreferrer">打开来源 <ArrowSquareOut size={15} weight="regular" aria-hidden="true" /></a>;
}

function highlightEvidence(text: string, excerpt?: string | null): ReactNode {
  const needle = (excerpt || "").replace(/…$/, "").trim();
  if (!needle || needle.length < 14) return text;
  const index = text.indexOf(needle);
  if (index < 0) return text;
  return <>{text.slice(0, index)}<mark>{text.slice(index, index + needle.length)}</mark>{text.slice(index + needle.length)}</>;
}

function freshnessLabel(citation: Citation): { label: string; kind: string } {
  const metadata = citation.metadata || {};
  if (metadata.expired === true) {
    const expires = new Date(String(metadata.expires_at || ""));
    const days = Number.isNaN(expires.getTime()) ? null : Math.max(1, Math.floor((Date.now() - expires.getTime()) / 86_400_000));
    return { label: days ? `该网页已过期 ${days} 天` : "该网页已过期", kind: "is-expired" };
  }
  if (citation.source_type === "web_page") return { label: "网页快照仍在有效期", kind: "is-fresh" };
  return { label: "已锁定本次引用版本", kind: "is-fresh" };
}
