"use client";

import { useEffect, useMemo, useState } from "react";

import type { StreamEvent } from "../lib/types";

type Props = {
  events: StreamEvent[];
  isRunning: boolean;
};

const replayableTypes = new Set([
  "start",
  "plan",
  "tool_call",
  "approval_required",
  "approval_resolved",
  "tool_result",
  "answer_verification_failed",
  "answer",
  "error",
]);

export function TraceTimeline({ events, isRunning }: Props) {
  const timeline = useMemo(() => events.filter((event) => replayableTypes.has(event.type)), [events]);
  const [replaying, setReplaying] = useState(false);
  const [shown, setShown] = useState(timeline.length);

  useEffect(() => {
    if (!replaying) setShown(timeline.length);
  }, [timeline.length, replaying]);

  useEffect(() => {
    if (!replaying) return;
    if (shown >= timeline.length) {
      setReplaying(false);
      return;
    }
    const id = window.setTimeout(() => setShown((value) => value + 1), 520);
    return () => window.clearTimeout(id);
  }, [replaying, shown, timeline.length]);

  function startReplay() {
    if (!timeline.length) return;
    setShown(0);
    setReplaying(true);
  }

  return (
    <section className="trace-panel" aria-label="Agent Trace 时间线">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">AGENT TRACE</p>
          <h2>本次推理回放</h2>
        </div>
        <button className="replay-button" onClick={startReplay} type="button" disabled={!timeline.length || isRunning}>
          {replaying ? "回放中" : "↻ 回放"}
        </button>
      </div>
      <div className="trace-state">
        <span className={isRunning ? "trace-live" : ""}>{isRunning ? "LIVE" : "REPLAY"}</span>
        <span>{timeline.length ? `${timeline.length} 个事件` : "等待第一条事件"}</span>
      </div>
      <ol className="timeline">
        {timeline.slice(0, shown).map((event, index) => (
          <TraceEvent key={`${event.sequence || index}-${event.type}`} event={event} index={index} />
        ))}
        {isRunning ? (
          <li className="timeline-item is-waiting">
            <span className="timeline-node" />
            <div>
              <strong>Agent 正在继续</strong>
              <p>等待下一步计划或工具结果</p>
            </div>
          </li>
        ) : null}
      </ol>
    </section>
  );
}

function TraceEvent({ event, index }: { event: StreamEvent; index: number }) {
  const descriptor = describeEvent(event);
  return (
    <li className={`timeline-item is-${descriptor.tone}`}>
      <span className="timeline-node">{String(index + 1).padStart(2, "0")}</span>
      <div className="timeline-content">
        <div className="timeline-title-row">
          <span>{descriptor.label}</span>
          {event.duration_ms !== undefined ? <small>{event.duration_ms} ms</small> : null}
        </div>
        <strong>{descriptor.title}</strong>
        {descriptor.detail ? <p>{descriptor.detail}</p> : null}
        {event.type === "tool_call" && event.args ? (
          <code>{shortJson(event.args)}</code>
        ) : null}
      </div>
    </li>
  );
}

function describeEvent(event: StreamEvent): { label: string; title: string; detail?: string; tone: string } {
  switch (event.type) {
    case "start":
      return { label: "开始", title: "接收问题，建立本轮证据账本", tone: "start" };
    case "plan":
      return {
        label: "计划",
        title: "Agent 制定下一步计划",
        detail: event.thought || "正在决定如何获取可验证证据。",
        tone: "tool",
      };
    case "tool_call":
      return {
        label: event.tool === "rag_search" ? "检索" : "工具调用",
        title: toolLabel(event.tool || "tool"),
        detail: event.thought || "Agent 正在执行下一步计划。",
        tone: "tool",
      };
    case "tool_result":
      return {
        label: event.ok ? "结果" : "异常",
        title: `${toolLabel(event.tool || "tool")} ${event.ok ? "已返回" : "未完成"}`,
        detail: event.ok ? "结果已写入本轮推理上下文。" : "该工具结果未被作为证据使用。",
        tone: event.ok ? "result" : "error",
      };
    case "approval_required":
      return { label: "审批", title: `等待人工确认：${toolLabel(event.approval?.tool || "tool")}`, tone: "approval" };
    case "approval_resolved":
      return {
        label: "审批",
        title: event.resolution === "approved" ? "已允许继续执行" : "审批未通过或已超时",
        tone: event.resolution === "approved" ? "result" : "error",
      };
    case "answer_verification_failed":
      return { label: "校验", title: "回答未通过证据校验，正在修复", tone: "approval" };
    case "answer":
      return { label: "完成", title: "最终回答已生成并完成证据校验", tone: "answer" };
    case "error":
      return { label: "异常", title: "本轮运行中断", detail: event.error, tone: "error" };
    default:
      return { label: "事件", title: event.type, tone: "tool" };
  }
}

function toolLabel(tool: string): string {
  const labels: Record<string, string> = {
    rag_search: "检索知识库",
    rag_read: "读取原始 chunk",
    web_search: "网页搜索",
    web_fetch: "读取网页",
    memory_search: "检索记忆",
    memory_write: "写入记忆",
    calculator: "计算",
    now: "读取当前时间",
  };
  return labels[tool] || tool;
}

function shortJson(value: Record<string, unknown>): string {
  const text = JSON.stringify(value);
  return text.length > 180 ? `${text.slice(0, 177)}…` : text;
}
