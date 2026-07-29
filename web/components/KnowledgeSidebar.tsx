"use client";

import {
  ChatCircleDots,
  Database,
  FileText,
  Globe,
  MagnifyingGlass,
  NotePencil,
  Plus,
  UserCircle,
} from "@phosphor-icons/react";
import type { ChangeEvent } from "react";

import type { DocumentRecord } from "../lib/types";

export const uploadAccept = ".pdf,.txt,.md,.markdown,.rst,.csv,.json,.yaml,.yml,.html,.py,.js,.ts";

type Surface = "chat" | "workspace";

type Props = {
  documents: DocumentRecord[];
  documentCount: number;
  chunkCount: number;
  isUploading: boolean;
  isRunning: boolean;
  activeSurface: Surface;
  uploadError?: string | null;
  onNewConversation: () => void;
  onFocusComposer: () => void;
  onSurfaceChange: (surface: Surface) => void;
  onUpload: (event: ChangeEvent<HTMLInputElement>) => void;
};

export function KnowledgeSidebar({
  documents,
  documentCount,
  chunkCount,
  isUploading,
  isRunning,
  activeSurface,
  uploadError,
  onNewConversation,
  onFocusComposer,
  onSurfaceChange,
  onUpload,
}: Props) {
  return (
    <aside className="knowledge-sidebar" aria-label="PKB Agent 导航">
      <div className="sidebar-brand-row">
        <button className="brand-lockup" type="button" onClick={onNewConversation} disabled={isRunning} aria-label="新建对话">
          <strong>PKB</strong><span>Agent</span>
        </button>
        <button className="sidebar-icon-button" type="button" onClick={onFocusComposer} aria-label="搜索或提问">
          <MagnifyingGlass size={22} weight="regular" aria-hidden="true" />
        </button>
      </div>

      <button className="new-conversation-button" type="button" onClick={onNewConversation} disabled={isRunning}>
        <NotePencil size={21} weight="regular" aria-hidden="true" />
        新对话
      </button>

      <nav className="sidebar-nav" aria-label="主要导航">
        <button className={activeSurface === "chat" ? "is-active" : ""} type="button" onClick={() => onSurfaceChange("chat")}>
          <ChatCircleDots size={21} weight="regular" aria-hidden="true" />
          对话
        </button>
        <button className={activeSurface === "workspace" ? "is-active" : ""} type="button" onClick={() => onSurfaceChange("workspace")}>
          <Database size={21} weight="regular" aria-hidden="true" />
          资料库
        </button>
      </nav>

      <div className="sidebar-section-label"><span>知识库</span><em>{documentCount}</em></div>
      <section className="knowledge-summary">
        <div><Database size={19} weight="regular" aria-hidden="true" /><span>默认知识库</span></div>
        <p>{documentCount ? `${documentCount} 份资料 · ${chunkCount} 个证据片段` : "上传资料后，即可开始可追溯的问答。"}</p>
      </section>

      <label className={`sidebar-upload ${isUploading ? "is-uploading" : ""}`}>
        <input type="file" accept={uploadAccept} onChange={onUpload} disabled={isUploading || isRunning} />
        <Plus size={20} weight="regular" aria-hidden="true" />
        <span>{isUploading ? "正在建立索引…" : "上传资料"}</span>
      </label>
      {uploadError ? <p className="sidebar-error">{uploadError}</p> : null}

      <div className="sidebar-section-label recent-label"><span>最近资料</span><em>{documents.length}</em></div>
      <div className="document-list">
        {documents.length ? (
          documents.map((document) => <DocumentItem document={document} key={document.id} />)
        ) : (
          <div className="empty-documents">
            <FileText size={20} weight="light" aria-hidden="true" />
            <p>这里会显示最近上传的资料。</p>
          </div>
        )}
      </div>

      <div className="sidebar-account">
        <UserCircle size={30} weight="regular" aria-hidden="true" />
        <div><strong>本地工作区</strong><span>Python Runtime</span></div>
      </div>
    </aside>
  );
}

function DocumentItem({ document }: { document: DocumentRecord }) {
  const isWebSource = document.source_type === "url";
  const status = document.source_status === "expired"
    ? document.source_label || "网页已过期"
    : document.source_status === "ready" ? "已同步" : document.source_status || "处理中";
  return (
    <article className="document-item" title={document.source_uri || document.title || "未命名资料"}>
      {isWebSource ? <Globe size={18} weight="regular" aria-hidden="true" /> : <FileText size={18} weight="regular" aria-hidden="true" />}
      <div>
        <strong>{document.title || "未命名资料"}</strong>
        <span>{document.version || "v—"} · {document.chunk_count ?? 0} 个片段</span>
      </div>
      <em className={document.source_status === "expired" ? "is-expired" : ""}>{status}</em>
    </article>
  );
}
