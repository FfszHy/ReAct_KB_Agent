"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useGSAP } from "@gsap/react";
import gsap from "gsap";
import {
  ArrowCounterClockwise,
  CheckCircle,
  CircleNotch,
  ClockCounterClockwise,
  MagnifyingGlass,
  ShieldWarning,
  Sparkle,
  WarningCircle,
  Wrench,
} from "@phosphor-icons/react";

import type { StreamEvent } from "../lib/types";

gsap.registerPlugin(useGSAP);

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
  const root = useRef<HTMLOListElement>(null);
  const renderedCount = useRef(0);
  const timeline = useMemo(() => events.filter((event) => replayableTypes.has(event.type)), [events]);
  const [replay, setReplay] = useState<{ index: number; total: number } | null>(null);
  const replaying = replay !== null;
  const shown = replay ? Math.min(replay.index, timeline.length) : timeline.length;

  useEffect(() => {
    if (!replay) return;
    const total = Math.min(replay.total, timeline.length);
    const delay = replay.index >= total ? 0 : 360;
    const id = window.setTimeout(() => {
      setReplay((current) => {
        if (!current) return null;
        const target = Math.min(current.total, timeline.length);
        if (target === 0 || current.index + 1 >= target) return null;
        return { ...current, index: current.index + 1 };
      });
    }, delay);
    return () => window.clearTimeout(id);
  }, [replay, timeline.length]);

  useGSAP(
    () => {
      const nodes = Array.from(root.current?.querySelectorAll<HTMLElement>("[data-trace-event]") || []);
      if (nodes.length < renderedCount.current) renderedCount.current = 0;
      const entering = nodes.slice(renderedCount.current);
      renderedCount.current = nodes.length;
      if (!entering.length) return;

      const media = gsap.matchMedia();
      media.add("(prefers-reduced-motion: no-preference)", () => {
        gsap.fromTo(entering, { autoAlpha: 0, y: 8 }, { autoAlpha: 1, y: 0, duration: 0.2, ease: "power3.out", stagger: 0.035 });
      });
      return () => media.revert();
    },
    { scope: root, dependencies: [shown, timeline.length] },
  );

  function startReplay() {
    if (!timeline.length) return;
    renderedCount.current = 0;
    setReplay({ index: 0, total: timeline.length });
  }

  return (
    <section className="trace-timeline" aria-label="Agent Trace 时间线">
      <div className="detail-panel-heading" data-run-pane-item>
        <div><ClockCounterClockwise size={18} weight="regular" aria-hidden="true" /><span>Agent Trace</span></div>
        <button className="replay-button" onClick={startReplay} type="button" disabled={!timeline.length || isRunning || replaying}>
          <ArrowCounterClockwise size={16} weight="bold" aria-hidden="true" />
          {replaying ? "回放中" : "回放"}
        </button>
      </div>
      <div className="trace-state" data-run-pane-item>
        <span className={isRunning ? "is-live" : ""}>{isRunning ? <CircleNotch size={14} className="is-spinning" /> : <CheckCircle size={14} weight="fill" />}{isRunning ? "正在接收事件" : "本轮已记录"}</span>
        <small>{timeline.length ? `${timeline.length} 个事件` : "等待第一条事件"}</small>
      </div>
      <ol className="timeline" ref={root}>
        {timeline.slice(0, shown).map((event, index) => <TraceEvent key={`${event.sequence || index}-${event.type}`} event={event} index={index} />)}
        {isRunning ? (
          <li className="timeline-item is-waiting" data-trace-event>
            <span className="timeline-node"><CircleNotch size={16} className="is-spinning" /></span>
            <div><strong>Agent 正在继续</strong><p>等待下一步计划或工具结果</p></div>
          </li>
        ) : null}
      </ol>
    </section>
  );
}

function TraceEvent({ event, index }: { event: StreamEvent; index: number }) {
  const descriptor = describeEvent(event);
  return (
    <li className={`timeline-item is-${descriptor.tone}`} data-trace-event data-run-pane-item>
      <span className="timeline-node">{eventIcon(event, descriptor.tone)}</span>
      <div className="timeline-content">
        <div className="timeline-title-row"><span>{descriptor.label}</span>{event.duration_ms !== undefined ? <small>{event.duration_ms} ms</small> : null}</div>
        <strong>{descriptor.title}</strong>
        {descriptor.detail ? <p>{descriptor.detail}</p> : null}
        {event.type === "tool_call" && event.args ? <code>{shortJson(event.args)}</code> : null}
      </div>
      <em>{String(index + 1).padStart(2, "0")}</em>
    </li>
  );
}

function eventIcon(event: StreamEvent, tone: string) {
  if (event.type === "tool_call" && event.tool === "rag_search") return <MagnifyingGlass size={16} weight="regular" />;
  if (tone === "error") return <WarningCircle size={16} weight="fill" />;
  if (tone === "approval") return <ShieldWarning size={16} weight="fill" />;
  if (tone === "answer") return <Sparkle size={16} weight="fill" />;
  if (tone === "result") return <CheckCircle size={16} weight="fill" />;
  return <Wrench size={16} weight="regular" />;
}

function describeEvent(event: StreamEvent): { label: string; title: string; detail?: string; tone: string } {
  switch (event.type) {
    case "start":
      return { label: "开始", title: "接收问题，建立本轮证据账本", tone: "start" };
    case "plan":
      return { label: "计划", title: "Agent 制定下一步计划", detail: event.thought || "正在决定如何获取可验证证据。", tone: "tool" };
    case "tool_call":
      return { label: event.tool === "rag_search" ? "检索" : "工具调用", title: toolLabel(event.tool || "tool"), detail: event.thought || "Agent 正在执行下一步计划。", tone: "tool" };
    case "tool_result":
      return { label: event.ok ? "结果" : "异常", title: `${toolLabel(event.tool || "tool")} ${event.ok ? "已返回" : "未完成"}`, detail: event.ok ? "结果已写入本轮推理上下文。" : "该工具结果未被作为证据使用。", tone: event.ok ? "result" : "error" };
    case "approval_required":
      return { label: "审批", title: `等待人工确认：${toolLabel(event.approval?.tool || "tool")}`, tone: "approval" };
    case "approval_resolved":
      return { label: "审批", title: event.resolution === "approved" ? "已允许继续执行" : "审批未通过或已超时", tone: event.resolution === "approved" ? "result" : "error" };
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
