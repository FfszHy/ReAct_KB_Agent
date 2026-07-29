import type {
  AnswerPayload,
  Approval,
  ChunkRecord,
  Citation,
  RunMetrics,
  RunSnapshot,
  StreamEvent,
} from "./types";

export type RunPhase = "idle" | "connecting" | "running" | "reconnecting" | "completed" | "failed";

export type RunUiState = {
  phase: RunPhase;
  runId: string | null;
  events: StreamEvent[];
  answer: AnswerPayload | null;
  metrics: RunMetrics | null;
  approval: Approval | null;
  approvalPending: boolean;
  approvalError: string | null;
  selectedCitation: Citation | null;
  selectedChunk: ChunkRecord | null;
  sourceLoading: boolean;
  sourceError: string | null;
  runError: string | null;
};

export const initialRunUiState: RunUiState = {
  phase: "idle",
  runId: null,
  events: [],
  answer: null,
  metrics: null,
  approval: null,
  approvalPending: false,
  approvalError: null,
  selectedCitation: null,
  selectedChunk: null,
  sourceLoading: false,
  sourceError: null,
  runError: null,
};

export type RunUiAction =
  | { type: "reset" }
  | { type: "run_requested" }
  | { type: "run_started"; runId: string }
  | { type: "run_start_failed"; error: string }
  | { type: "stream_reconnecting" }
  | { type: "stream_connected" }
  | { type: "stream_failed"; error: string }
  | { type: "stream_event"; event: StreamEvent }
  | { type: "approval_pending"; value: boolean }
  | { type: "approval_failed"; error: string }
  | { type: "source_selected"; citation: Citation }
  | { type: "source_loaded"; chunk: ChunkRecord | null }
  | { type: "source_failed"; error: string }
  | { type: "source_closed" };

export function runUiReducer(state: RunUiState, action: RunUiAction): RunUiState {
  switch (action.type) {
    case "reset":
      return initialRunUiState;
    case "run_requested":
      return { ...initialRunUiState, phase: "connecting" };
    case "run_started":
      return {
        ...initialRunUiState,
        phase: "running",
        runId: action.runId,
      };
    case "run_start_failed":
      return {
        ...initialRunUiState,
        phase: "failed",
        runError: action.error,
      };
    case "stream_reconnecting":
      return state.phase === "running"
        ? { ...state, phase: "reconnecting" }
        : state;
    case "stream_connected":
      return state.phase === "reconnecting"
        ? { ...state, phase: "running", runError: null }
        : state;
    case "stream_failed":
      return {
        ...state,
        phase: "failed",
        approval: null,
        approvalPending: false,
        approvalError: null,
        runError: action.error,
      };
    case "stream_event":
      return reduceStreamEvent(state, action.event);
    case "approval_pending":
      return {
        ...state,
        approvalPending: action.value,
        approvalError: action.value ? null : state.approvalError,
      };
    case "approval_failed":
      return { ...state, approvalPending: false, approvalError: action.error };
    case "source_selected":
      return {
        ...state,
        selectedCitation: action.citation,
        selectedChunk: null,
        sourceLoading: action.citation.source_type === "kb_chunk",
        sourceError: null,
      };
    case "source_loaded":
      return { ...state, selectedChunk: action.chunk, sourceLoading: false, sourceError: null };
    case "source_failed":
      return { ...state, sourceLoading: false, sourceError: action.error };
    case "source_closed":
      return {
        ...state,
        selectedCitation: null,
        selectedChunk: null,
        sourceLoading: false,
        sourceError: null,
      };
    default:
      return state;
  }
}

function reduceStreamEvent(state: RunUiState, event: StreamEvent): RunUiState {
  let next: RunUiState = {
    ...state,
    events: [...state.events, event],
    metrics: event.metrics || state.metrics,
  };

  if (event.type === "approval_required" && event.approval) {
    next = { ...next, approval: event.approval, approvalPending: false, approvalError: null };
  }
  if (event.type === "approval_resolved") {
    next = {
      ...next,
      approval: next.approval?.id === event.approval_id ? null : next.approval,
      approvalPending: false,
      approvalError: null,
    };
  }
  if (event.type === "answer") {
    next = { ...next, answer: event.answer_payload || next.answer };
  }
  if (event.type === "error") {
    next = { ...next, runError: event.error || "运行中出现异常" };
  }
  if (event.type !== "run_finished") return next;

  const snapshot = event.state as RunSnapshot | null | undefined;
  const failed = event.status === "error" || snapshot?.status === "error";
  return {
    ...next,
    phase: failed ? "failed" : "completed",
    answer: snapshot?.answer || next.answer,
    metrics: snapshot?.metrics || next.metrics,
    approval: null,
    approvalPending: false,
    approvalError: null,
    runError: event.error || snapshot?.error || next.runError,
  };
}

export function isRunActive(phase: RunPhase): boolean {
  return phase === "connecting" || phase === "running" || phase === "reconnecting";
}
