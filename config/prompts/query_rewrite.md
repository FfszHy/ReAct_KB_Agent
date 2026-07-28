# Query rewriting guidance for retrieval.

Before issuing a `rag_search` or `web_search` call, mentally rewrite the user's
natural question into one or more retrieval-optimal queries:

- Drop filler and conversational pronouns; keep only content terms.
- Expand ambiguous acronyms when context makes the expansion clear.
- For multi-part questions, split into one focused query per part.
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
