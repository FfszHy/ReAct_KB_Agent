"use client";

import { Check, ShieldWarning, X } from "@phosphor-icons/react";

import type { Approval } from "../lib/types";

type Props = {
  approval: Approval | null;
  pending: boolean;
  error?: string | null;
  onDecision: (approved: boolean) => void;
};

export function ApprovalDialog({ approval, pending, error, onDecision }: Props) {
  if (!approval) return null;
  return (
    <div className="approval-backdrop" role="presentation">
      <section className="approval-dialog" aria-modal="true" role="dialog" aria-labelledby="approval-title">
        <span className="approval-symbol" aria-hidden="true"><ShieldWarning size={24} weight="fill" /></span>
        <p>需要人工确认</p>
        <h2 id="approval-title">允许 Agent 执行{toolLabel(approval.tool)}？</h2>
        <span className="approval-reason">{approval.reason}</span>
        <div className="approval-impact">
          <strong>为什么需要确认</strong>
          <p>{approval.impact}</p>
        </div>
        <pre className="approval-arguments">{JSON.stringify(approval.arguments, null, 2)}</pre>
        {error ? <p className="approval-error" role="alert">提交失败：{error}。请重试。</p> : null}
        <div className="approval-actions">
          <button className="approval-deny" type="button" disabled={pending} onClick={() => onDecision(false)}><X size={17} weight="bold" />拒绝</button>
          <button className="approval-allow" type="button" disabled={pending} onClick={() => onDecision(true)}><Check size={17} weight="bold" />{pending ? "正在提交…" : "允许继续"}</button>
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
