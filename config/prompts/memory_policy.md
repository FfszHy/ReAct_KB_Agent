# Memory policy for PKB-Agent.

The agent's `memory_write` tool is the ONLY way the agent persists anything
across runs. Use it sparingly and deliberately.

The runtime requires confirmation for memory writes by default and rejects
secret-looking content even if a write is approved.

## What to write to memory

- Stable facts the user explicitly asks to remember.
- Resolved preferences or constraints that affect future answers.
- Concise, reusable conclusions that required non-trivial work to derive
  (so future runs avoid recomputing them).

## What NOT to write to memory

- Ephemeral conversation state (handled by the run trace, not memory).
- Raw document content that already lives in the KB (write a reference instead).
- Sensitive secrets, credentials, or PII the user did not explicitly persist.
- Speculative guesses.

## Granularity

- One memory note = one self-contained fact / decision.
- Keep each note under ~4000 chars; link related notes by id when needed.
- Tag notes with a short `kind` (e.g. `preference`, `fact`, `decision`, `reference`).

## Retrieval

- `memory_search` uses semantic similarity over memory notes. Prefer specific
  queries; if nothing relevant is found, do not assume absence — note the gap.
- Memory is advisory: it informs answers but does not override explicit user
  instructions in the current turn.
