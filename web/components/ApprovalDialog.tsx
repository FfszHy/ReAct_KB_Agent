"use client";

import type { Approval } from "../lib/types";

type Props = {
  approval: Approval | null;
  pending: boolean;
  onDecision: (approved: boolean) => void;
};

export function ApprovalDialog({ approval, pending, onDecision }: Props) {
  if (!approval) return null;
  return (
    <div className="approval-backdrop" role="presentation">
      <section className="approval-dialog" aria-modal="true" role="dialog" aria-labelledby="approval-title">
        <div className="approval-symbol">!</div>
        <p className="eyebrow">HUMAN APPROVAL REQUIRED</p>
        <h2 id="approval-title">允许 Agent 执行 {toolLabel(approval.tool)}？</h2>
        <p className="approval-reason">{approval.reason}</p>
        <div className="approval-impact">
          <span>为什么需要确认</span>
          <p>{approval.impact}</p>
        </div>
        <pre className="approval-arguments">{JSON.stringify(approval.arguments, null, 2)}</pre>
        <div className="approval-actions">
          <button className="button button-quiet" type="button" disabled={pending} onClick={() => onDecision(false)}>
            拒绝此操作
          </button>
          <button className="button button-primary" type="button" disabled={pending} onClick={() => onDecision(true)}>
            {pending ? "正在提交…" : "允许继续"}
          </button>
        </div>
      </section>
    </div>
  );
}

function toolLabel(tool: string): string {
  if (tool === "web_fetch") return "网页读取";
  if (tool === "memory_write") return "记忆写入";
  return tool;
}
