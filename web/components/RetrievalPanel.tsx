"use client";

import { ArrowSquareOut, FileText, MagnifyingGlass } from "@phosphor-icons/react";

import type { RetrievalResult } from "../lib/types";

type Props = {
  results: RetrievalResult[];
  onResult: (result: RetrievalResult) => void;
};

export function RetrievalPanel({ results, onResult }: Props) {
  return (
    <section className="retrieval-panel" aria-label="本轮检索结果">
      <div className="detail-panel-heading" data-run-pane-item>
        <div><MagnifyingGlass size={18} weight="regular" aria-hidden="true" /><span>检索结果</span></div>
        <small>{results.length} 条候选</small>
      </div>
      {results.length ? (
        <div className="retrieval-list">
          {results.map((result, index) => (
            <button className="retrieval-item" type="button" key={`${result.chunk_id}-${index}`} onClick={() => onResult(result)} data-run-pane-item>
              <span className="retrieval-rank">{String(result.rank ?? index + 1).padStart(2, "0")}</span>
              <FileText size={18} weight="regular" aria-hidden="true" />
              <div>
                <strong>{result.title || "未命名资料"}</strong>
                <p>{result.content_preview || "查看原始 chunk"}</p>
                <small>Chunk {result.chunk_index ?? "—"} · 相关度 {formatScore(result.score)}</small>
              </div>
              <ArrowSquareOut size={16} weight="regular" aria-hidden="true" />
            </button>
          ))}
        </div>
      ) : (
        <p className="no-retrieval" data-run-pane-item>本轮尚未返回候选证据。</p>
      )}
    </section>
  );
}

function formatScore(value?: number | null): string {
  return value === undefined || value === null ? "—" : value.toFixed(2);
}
