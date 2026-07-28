"use client";

import { useState } from "react";

import type { RetrievalResult } from "../lib/types";

type Props = {
  results: RetrievalResult[];
  onResult: (result: RetrievalResult) => void;
};

export function RetrievalPanel({ results, onResult }: Props) {
  const [open, setOpen] = useState(false);
  return (
    <section className="retrieval-panel">
      <button className="retrieval-toggle" type="button" onClick={() => setOpen((value) => !value)}>
        <span>
          <b>检索结果</b>
          <small>{results.length ? `${results.length} 个候选证据` : "本轮尚未返回候选"}</small>
        </span>
        <span>{open ? "−" : "+"}</span>
      </button>
      {open ? (
        <div className="retrieval-list">
          {results.length ? (
            results.map((result, index) => (
              <button className="retrieval-item" type="button" key={`${result.chunk_id}-${index}`} onClick={() => onResult(result)}>
                <span>#{result.rank ?? index + 1}</span>
                <div>
                  <strong>{result.title || "未命名资料"}</strong>
                  <p>{result.content_preview || "查看原始 chunk"}</p>
                  <small>
                    Chunk {result.chunk_index ?? "—"} · 相关度 {formatScore(result.score)}
                  </small>
                </div>
              </button>
            ))
          ) : (
            <p className="no-retrieval">检索完成后，可以在这里展开未被最终引用的候选片段。</p>
          )}
        </div>
      ) : null}
    </section>
  );
}

function formatScore(score?: number): string {
  return typeof score === "number" ? score.toFixed(3) : "—";
}
