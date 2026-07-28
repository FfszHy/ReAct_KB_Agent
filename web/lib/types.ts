export type SourceMetadata = Record<string, unknown>;

export type Citation = {
  id: string;
  source_type: "kb_chunk" | "web_page" | string;
  source_id: string;
  title?: string | null;
  locator?: string | null;
  excerpt?: string | null;
  metadata?: SourceMetadata;
};

export type AnswerPayload = {
  status: "grounded" | "insufficient_evidence" | string;
  answer: string;
  claims: Array<{
    text: string;
    kind: "fact" | "inference" | string;
    citations: string[];
  }>;
  citations: Citation[];
};

export type Usage = {
  prompt_tokens?: number;
  prompt_cache_hit_tokens?: number;
  prompt_cache_miss_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  estimated_cost?: number;
  cost_currency?: string;
};

export type RunMetrics = {
  duration_ms?: number | null;
  tool_duration_ms?: number | null;
  tool_call_count?: number;
  successful_tool_call_count?: number;
  tool_success_rate?: number | null;
  run_succeeded?: boolean;
  run_success_rate?: number;
  usage?: Usage;
};

export type DocumentRecord = {
  id: string;
  title: string;
  source_uri?: string | null;
  source_type?: string;
  content_hash?: string | null;
  chunk_count?: number;
  char_count?: number;
  created_at?: string;
  version?: string;
  source_status?: string;
  source_label?: string;
};

export type ChunkRecord = {
  id?: string;
  chunk_id?: string;
  document_id?: string;
  chunk_index?: number;
  content?: string;
  token_count?: number;
  document?: { title?: string; source_uri?: string; source_type?: string };
};

export type Approval = {
  id: string;
  run_id: string;
  tool: string;
  arguments: Record<string, unknown>;
  reason: string;
  impact: string;
  requested_at: string;
};

export type StreamEvent = {
  type: string;
  sequence?: number;
  emitted_at?: string;
  step?: number;
  tool?: string;
  args?: Record<string, unknown>;
  thought?: string | null;
  ok?: boolean;
  truncated?: boolean;
  duration_ms?: number;
  observation?: string;
  answer?: string;
  answer_payload?: AnswerPayload;
  verification?: Record<string, unknown>;
  approval?: Approval;
  approval_id?: string;
  resolution?: string;
  metrics?: RunMetrics;
  error?: string;
  state?: RunSnapshot | null;
};

export type RunSnapshot = {
  run_id: string;
  question: string;
  status: string;
  final_answer?: string | null;
  answer?: AnswerPayload | null;
  verification?: Record<string, unknown>;
  metrics?: RunMetrics;
  retrieved_evidence?: Citation[];
  error?: string | null;
};

export type RetrievalResult = {
  rank?: number;
  chunk_id: string;
  document_id?: string;
  title?: string;
  source_uri?: string;
  chunk_index?: number;
  score?: number;
  content_preview?: string;
};
