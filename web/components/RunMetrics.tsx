"use client";

import { ChartLineUp, Clock, Coins, Wrench } from "@phosphor-icons/react";
import type { ReactNode } from "react";

import type { RunMetrics as RunMetricsType, Usage } from "../lib/types";

type Props = {
  metrics: RunMetricsType | null;
  aggregateSuccessRate?: number | null;
  isRunning: boolean;
};

export function RunMetrics({ metrics, aggregateSuccessRate, isRunning }: Props) {
  const usage = metrics?.usage || {};
  const settled = !isRunning && isCostSettled(usage);
  const actualCost = usage.actual_cost;
  const currency = usage.cost_currency || "CNY";

  return (
    <section className="run-metrics" aria-label="本轮运行数据">
      <div className="run-metrics-heading">
        <span>本轮运行</span>
        <small>{isRunning ? "正在记录" : settled ? "已结算" : metrics ? "成本未记录" : "等待开始"}</small>
      </div>
      <dl className="metrics-grid">
        <Metric icon={<Clock size={17} weight="regular" />} label="耗时" value={formatDuration(metrics?.duration_ms)} hint={toolDurationHint(metrics)} />
        <Metric
          icon={<Coins size={17} weight="regular" />}
          label="实际成本"
          value={settled ? formatCost(actualCost, currency) : "—"}
          hint={settled ? completedUsageHint(usage) : costPendingHint(metrics, isRunning)}
        />
        <Metric
          icon={<Wrench size={17} weight="regular" />}
          label="工具成功率"
          value={formatPercent(metrics?.tool_success_rate)}
          hint={`${metrics?.successful_tool_call_count ?? 0}/${metrics?.tool_call_count ?? 0} 次调用成功`}
        />
        <Metric
          icon={<ChartLineUp size={17} weight="regular" />}
          label="近期成功率"
          value={formatPercent(aggregateSuccessRate)}
          hint={metrics ? (metrics.run_succeeded ? "本轮回答已通过证据核验" : "按已完成运行汇总") : "等待第一轮运行"}
        />
      </dl>
      <p className="token-line">
        <span>实际 Token</span><strong>{(usage.total_tokens || 0).toLocaleString()}</strong>
        <small>输入 {(usage.prompt_tokens || 0).toLocaleString()} · 输出 {(usage.completion_tokens || 0).toLocaleString()}</small>
      </p>
    </section>
  );
}

function Metric({
  icon,
  label,
  value,
  hint,
}: {
  icon: ReactNode;
  label: string;
  value: string;
  hint: string;
}) {
  return (
    <div className="metric-card" data-run-pane-item>
      <dt>{icon}<span>{label}</span></dt>
      <dd><strong>{value}</strong><small>{hint}</small></dd>
    </div>
  );
}

function isCostSettled(usage: Usage): boolean {
  if (usage.cost_status === "accruing") return false;
  return usage.actual_cost !== undefined;
}

function formatDuration(value?: number | null): string {
  if (value === undefined || value === null) return "—";
  if (value < 1000) return `${value} ms`;
  return `${(value / 1000).toFixed(value < 10_000 ? 1 : 0)} 秒`;
}

function formatPercent(value?: number | null): string {
  return value === undefined || value === null ? "—" : `${value}%`;
}

function formatCost(value: number | undefined, currency: string): string {
  if (value === undefined || value === null) return "—";
  const symbol = currency === "CNY" ? "¥" : currency === "USD" ? "$" : `${currency} `;
  return `${symbol}${new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 8 }).format(value)}`;
}

function completedUsageHint(usage: Usage): string {
  const hit = usage.prompt_cache_hit_tokens || 0;
  const miss = usage.prompt_cache_miss_tokens || 0;
  if (hit || miss) return `命中 ${hit.toLocaleString()} · 未命中 ${miss.toLocaleString()}`;
  return "按本轮 API 返回的实际用量结算";
}

function toolDurationHint(metrics: RunMetricsType | null): string {
  const value = metrics?.tool_duration_ms;
  return value === undefined || value === null ? "等待任务开始" : `工具 ${formatDuration(value)}`;
}

function costPendingHint(metrics: RunMetricsType | null, isRunning: boolean): string {
  if (isRunning) return "任务结束后结算";
  if (metrics) return "此历史运行没有成本记录";
  return "任务结束后结算";
}
