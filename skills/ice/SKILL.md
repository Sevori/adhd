---
name: ice
description: "Persist information to the ICE shared continuity kernel. Use when the user says 'ice that', 'ice this', or '/ice <info>' to write facts, decisions, lessons, scars, or constraints into the organism's memory."
user-invocable: true
argument-hint: "<information to persist>"
---

# Ice — Persist to the ICE Continuity Kernel

The user said `/ice $ARGUMENTS`.

Your job is to persist the provided information into the ICE shared continuity kernel so it survives across sessions and agents.

## Steps

1. **Identify the machine** — call `continuity_identify_machine` to get the namespace and default task.

2. **Read current context** — call `continuity_read_context` with a brief objective summarising what the user wants to ice, to get the active context ID.

3. **Classify the information** into one of these continuity item kinds:
   - `fact` — something that is true (default if unclear)
   - `decision` — a choice that was made and why
   - `constraint` — a rule or limitation that must be respected
   - `lesson` — something learned from experience
   - `scar` — an operational scar from an incident or failure
   - `incident` — an active or past incident

4. **Write the item** — call `continuity_write_item` with:
   - `context_id`: from the context read
   - `author_agent_id`: `"claude-code"`
   - `kind`: the classified kind
   - `title`: a concise title (under 80 chars)
   - `body`: the full information, preserving the user's intent
   - `scope`: choose appropriately — `global` for user/machine-wide facts, `project` for project-specific, `shared` (default) for general

5. **Confirm** — tell the user what was iced, including the kind and title.

## Behaviour when no arguments are provided

If `$ARGUMENTS` is empty, ask the user what they'd like to ice.

## Tone

Keep it brief. A pun involving ice/cold/frozen is acceptable but not required.
