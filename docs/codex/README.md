# Codex ↔ Claude Communication Channel

File-based handoff between two AI assistants working on the same repo. Both agents read/write through git — the user pulls/pushes as needed.

## Protocol

- **Claude → Codex (tasks):** briefs live in `briefs/NNN-<topic>.md`. Each brief is **self-contained** — Codex won't see Claude's conversation history, so the brief must include all context (project goal, schema, paths, acceptance criteria, anti-goals).
- **Codex → Claude (status + findings):** replies live in `status/NNN-<topic>.md`. Codex creates the file on pickup, updates it as work progresses, and writes a final summary on completion.
- **Numbering:** monotonic 3-digit prefix. The latest unhandled brief is the highest number whose status file is missing or has `status: pending|in_progress`.

## Ownership boundaries (avoid merge conflicts)

| Files | Owner |
|---|---|
| `train*.py`, `pretrain_mae.py`, `track_d_pipeline.py` | Claude |
| `experiments/configs/*.yaml`, `experiments/ledger.jsonl` | Claude |
| `prepare.py`, `scripts/run_*.sh`, `scripts/plots/*` | Claude |
| `paper/figures/*` | Claude (regeneratable) |
| `briefs/*.md` (after first write) | Read-only for both |
| `status/*.md` | Codex |
| `scripts/codex/*` | Codex |
| `data/pseudo_labels/*` | Codex |
| `paper/main.md`, `paper/sections/*` | Codex (if a brief assigns it) |
| `papers/README.md` (citation registry) | Codex (if a brief assigns it) |
| `docs/superpowers/specs/*`, `docs/superpowers/plans/*` | Claude |

Both can **read** everything. Neither edits the other's primary files without a brief explicitly authorizing it.

## Lifecycle of a brief

1. Claude writes `briefs/NNN-<topic>.md` with the structure below and commits.
2. Codex (when invoked by the user) scans `briefs/` for the highest-numbered brief without a matching `status/` file, or with status `pending`.
3. Codex creates `status/NNN-<topic>.md` with `status: in_progress`, writes an opening note, and starts work.
4. Codex appends to the same status file as work progresses (every meaningful checkpoint).
5. On completion: Codex sets `status: done`, writes a final summary block at the bottom, and commits.
6. Claude reads the status file, integrates the deliverable, marks the brief consumed by leaving a closing line in the status file under `## Claude acknowledged`.

## Brief structure (template)

```markdown
# Brief NNN: <topic>

**Owner:** Codex
**Created:** YYYY-MM-DD by Claude
**Estimated effort:** S / M / L

## Context (self-contained — Codex has not seen prior conversation)
<everything Codex needs to understand the project and this task>

## Goal
<one-sentence outcome>

## Step-by-step
<concrete checklist>

## Acceptance criteria
<bullet list — what makes this "done">

## Anti-goals (don't do these)
<bullet list>

## Where to write
<exact paths>
```

## Status file structure (template)

```markdown
---
brief: NNN-<topic>
status: in_progress  # pending | in_progress | done | blocked
started: YYYY-MM-DD HH:MM
updated: YYYY-MM-DD HH:MM
---

## YYYY-MM-DD HH:MM — opening note
<what Codex understood from the brief, planned approach>

## YYYY-MM-DD HH:MM — progress
<checkpoints as work proceeds>

## YYYY-MM-DD HH:MM — final summary
<what was produced, where it lives, key numbers, anything Claude needs to know>

## Claude acknowledged
<Claude appends a one-line ack after consuming the deliverable>
```

## Conflict resolution

If both agents end up needing to edit the same file:
- Stop. Don't overwrite each other.
- Whichever agent notices first: write to `status/conflict-<file>.md` describing what each wants to change.
- User mediates.

## Cost discipline

Codex tasks that hit paid APIs (OpenAI vision, etc.) must:
- Report estimated $ in the brief
- Confirm before exceeding 2× the estimate
- Write actual $ spent into the final status summary
