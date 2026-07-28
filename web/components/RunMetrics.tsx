"use client";

import type { RunMetrics } from "../lib/types";

type Props = {
  metrics: RunMetrics | null;
  aggregateSuccessRate?: number | null;
  isRunning: boolean;
};

export function RunMetrics({ metrics, aggregateSuccessRate, isRunning }: Props) {
  const usage = metrics?.usage || {};
  const currency = usage.cost_currency || "CNY";
  return (
    <section className="metrics-panel" aria-label="运行指标">
      <div className="panel-heading compact-heading">
        <div>
          <p className="eyebrow">RUN TELEMETRY</p>
          <h2>运行健康度</h2>
        </div>
        <span className={`run-chip ${isRunning ? "is-running" : ""}`}>{isRunning ? "运行中" : "已记录"}</span>
      </div>
      <div className="metrics-grid">
        <Metric label="耗时" value={formatDuration(metrics?.duration_ms)} hint="端到端" />
        <Metric label="预估成本" value={formatCost(usage.estimated_cost, currency)} hint={cacheHint(usage)} />
        <Metric
          label="工具成功率"
          value={metrics?.tool_success_rate === null || metrics?.tool_success_rate === undefined ? "—" : `${metrics.tool_success_rate}%`}
          hint={`${metrics?.successful_tool_call_count ?? 0}/${metrics?.tool_call_count ?? 0} 次调用`}
        />
        <Metric
          label="近期成功率"
          value={aggregateSuccessRate === null || aggregateSuccessRate === undefined ? "—" : `${aggregateSuccessRate}%`}
          hint={metrics?.run_succeeded ? "本轮证据已核验" : "按已完成运行计算"}
        />
      </div>
      <div className="token-line">
        <span>Token</span>
        <strong>{(usage.total_tokens || 0).toLocaleString()}</strong>
        <span>输入 {usage.prompt_tokens || 0} · 输出 {usage.completion_tokens || 0}</span>
      </div>
    </section>
  );
}

function Metric({ label, value, hint }: { label: string; value: string; hint: string }) {
  return (
    <div className="metric-card">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{hint}</small>
    </div>
  );
}

function formatDuration(value?: number | null): string {
  if (!value) return "—";
  if (value < 1000) return `${value}ms`;
  return `${(value / 1000).toFixed(value < 10_000 ? 1 : 0)}s`;
}

function formatCost(value: number | undefined, currency: string): string {
  if (value === undefined || value === null) return "—";
  const symbol = currency === "CNY" ? "¥" : currency === "USD" ? "$" : `${currency} `;
  return `${symbol}${value < 0.01 ? value.toFixed(4) : value.toFixed(2)}`;
}

function cacheHint(usage: RunMetrics["usage"]): string {
  const hit = usage?.prompt_cache_hit_tokens || 0;
  const miss = usage?.prompt_cache_miss_tokens || 0;
  if (hit || miss) return `缓存命中 ${hit.toLocaleString()} · 未命中 ${miss.toLocaleString()}`;
  return "缓存分项将在模型响应后显示";
}
