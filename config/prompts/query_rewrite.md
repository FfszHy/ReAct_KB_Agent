# Query rewriting guidance for retrieval.

Before issuing a `rag_search` or `web_search` call, mentally rewrite the user's
natural question into one or more retrieval-optimal queries:

- Drop filler and conversational pronouns; keep only content terms.
- Preserve explicit versions, exact API/field names, negation, and conditions
  that distinguish the requested behavior. Do not rewrite the question into
  an assumed answer or confirmed diagnosis.
- Expand ambiguous acronyms when context makes the expansion clear.
- For multi-part questions, split into one focused query per part.
- Keep the current retrieval query's purpose. It may target a gap discovered
  after earlier results; the original user question supplies context, not a
  reason to replace a focused follow-up with a broad restart.
- Use only as many distinct queries as needed; do not fill a query quota with
  paraphrases or unrelated subtopics.
- Prefer specific entities and key terms over full sentences.
- Keep each rewritten query short (≤ ~12 words) and keyword-dense.

Examples:
- "What did we decide about the auth approach last month?"
  → queries: `auth approach decision`, `authentication design`
- "How do I configure pgvector distance operators?"
  → query: `pgvector distance operator configuration`

At runtime this prompt is invoked only by the retrieval QueryPlanner, directly
before `rag_search` or `web_search`. Follow the JSON output contract supplied
by that caller; do not add explanations or Markdown fences.
