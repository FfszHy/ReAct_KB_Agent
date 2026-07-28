"use client";

import type { ChangeEvent } from "react";

import type { DocumentRecord } from "../lib/types";

type Props = {
  documents: DocumentRecord[];
  documentCount: number;
  chunkCount: number;
  isUploading: boolean;
  uploadError?: string | null;
  onUpload: (event: ChangeEvent<HTMLInputElement>) => void;
};

export function KnowledgeSidebar({
  documents,
  documentCount,
  chunkCount,
  isUploading,
  uploadError,
  onUpload,
}: Props) {
  return (
    <aside className="knowledge-sidebar" aria-label="知识库资料">
      <div className="brand-lockup">
        <span className="brand-mark">K</span>
        <div>
          <p className="eyebrow">EVIDENCE / OPS</p>
          <h1>PKB Workbench</h1>
        </div>
      </div>

      <section className="knowledge-summary">
        <div className="summary-topline">
          <span className="pulse-dot" />
          <span>默认知识库</span>
          <span className="summary-ready">READY</span>
        </div>
        <p>资料进库后，每个回答都必须回到可核验的原始证据。</p>
        <div className="summary-numbers">
          <div>
            <strong>{documentCount}</strong>
            <span>份资料</span>
          </div>
          <div>
            <strong>{chunkCount}</strong>
            <span>证据片段</span>
          </div>
        </div>
      </section>

      <label className={`upload-zone ${isUploading ? "is-uploading" : ""}`}>
        <input
          type="file"
          accept=".pdf,.txt,.md,.markdown,.rst,.csv,.json,.yaml,.yml,.html,.py,.js,.ts"
          onChange={onUpload}
          disabled={isUploading}
        />
        <span className="upload-icon">＋</span>
        <strong>{isUploading ? "正在建立索引…" : "上传资料"}</strong>
        <small>PDF、Markdown、文本与代码文件</small>
      </label>
      {uploadError ? <p className="inline-error">{uploadError}</p> : null}

      <div className="section-heading">
        <span>资料版本</span>
        <span>{documents.length}</span>
      </div>
      <div className="document-list">
        {documents.length ? (
          documents.map((document) => <DocumentItem document={document} key={document.id} />)
        ) : (
          <div className="empty-documents">
            <span>01</span>
            <p>上传第一份资料，建立可检索的知识库。</p>
          </div>
        )}
      </div>

      <div className="sidebar-footer">
        <span>Python Runtime</span>
        <span>FastAPI · SSE</span>
      </div>
    </aside>
  );
}

function DocumentItem({ document }: { document: DocumentRecord }) {
  const title = document.title || "未命名资料";
  const status =
    document.source_status === "ready"
      ? "已同步"
      : document.source_status === "expired"
        ? document.source_label || "网页已过期"
        : document.source_status || "处理中";
  return (
    <article className="document-item" title={document.source_uri || title}>
      <div className="document-filetype">{fileGlyph(document.source_type)}</div>
      <div className="document-copy">
        <strong>{title}</strong>
        <span>
          {document.version || "v—"} · {document.chunk_count ?? 0} chunks
        </span>
      </div>
      <span className={`source-state ${document.source_status === "ready" ? "is-current" : ""}`}>
        {status}
      </span>
    </article>
  );
}

function fileGlyph(type?: string): string {
  if (type === "url") return "↗";
  if (type === "text") return "T";
  return "F";
}
