"use client";

import { useRef, useState } from "react";
import { useGSAP } from "@gsap/react";
import gsap from "gsap";
import { ChartLineUp, ClockCounterClockwise, MagnifyingGlass } from "@phosphor-icons/react";

import { RetrievalPanel } from "./RetrievalPanel";
import { RunMetrics } from "./RunMetrics";
import { TraceTimeline } from "./TraceTimeline";
import type { RetrievalResult, RunMetrics as RunMetricsType, StreamEvent } from "../lib/types";

gsap.registerPlugin(useGSAP);

type Pane = "retrieval" | "trace";

type Props = {
  events: StreamEvent[];
  results: RetrievalResult[];
  metrics: RunMetricsType | null;
  aggregateSuccessRate: number | null;
  isRunning: boolean;
  onResult: (result: RetrievalResult) => void;
};

export function RunDetails({
  events,
  results,
  metrics,
  aggregateSuccessRate,
  isRunning,
  onResult,
}: Props) {
  const root = useRef<HTMLElement>(null);
  const [activePane, setActivePane] = useState<Pane | null>(isRunning ? "trace" : null);

  useGSAP(
    () => {
      if (!activePane) return;
      const media = gsap.matchMedia();
      media.add("(prefers-reduced-motion: no-preference)", () => {
        const timeline = gsap.timeline({ defaults: { ease: "power3.out" } });
        timeline
          .fromTo("[data-run-pane]", { autoAlpha: 0, y: 8 }, { autoAlpha: 1, y: 0, duration: 0.2 })
          .fromTo(
            "[data-run-pane-item]",
            { autoAlpha: 0, y: 5 },
            { autoAlpha: 1, y: 0, duration: 0.16, stagger: 0.035 },
            "<0.04",
          );
      });
      return () => media.revert();
    },
    { scope: root, dependencies: [activePane], revertOnUpdate: true },
  );

  function togglePane(pane: Pane) {
    setActivePane((current) => current === pane ? null : pane);
  }

  return (
    <section className="run-details" ref={root} aria-label="本轮证据与运行详情">
      <RunMetrics metrics={metrics} aggregateSuccessRate={aggregateSuccessRate} isRunning={isRunning} />

      <div className="run-detail-switcher" aria-label="展开运行详情">
        <button
          className={activePane === "retrieval" ? "is-active" : ""}
          type="button"
          aria-expanded={activePane === "retrieval"}
          onClick={() => togglePane("retrieval")}
        >
          <MagnifyingGlass size={16} weight="regular" aria-hidden="true" />
          检索结果 <span>{results.length}</span>
        </button>
        <button
          className={activePane === "trace" ? "is-active" : ""}
          type="button"
          aria-expanded={activePane === "trace"}
          onClick={() => togglePane("trace")}
        >
          <ClockCounterClockwise size={16} weight="regular" aria-hidden="true" />
          Agent Trace <span>{events.length}</span>
          {isRunning ? <i aria-label="实时更新" /> : null}
        </button>
        {metrics ? <span className="run-complete-mark"><ChartLineUp size={15} weight="regular" aria-hidden="true" /> 本轮已记录</span> : null}
      </div>

      {activePane ? (
        <div className="run-detail-pane" data-run-pane>
          {activePane === "retrieval" ? (
            <RetrievalPanel results={results} onResult={onResult} />
          ) : (
            <TraceTimeline events={events} isRunning={isRunning} />
          )}
        </div>
      ) : null}
    </section>
  );
}
