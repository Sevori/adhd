# Copilot Instructions

Before making meaningful changes in this repository:

1. Read `README.md`.
2. Read the recent git log.

Repository rules:

- Keep repository documentation in English.
- Do not assume the product domain, target user, platform, or implementation stack unless it is explicitly documented.
- Start new work from `main` on a short-lived branch unless a human explicitly says otherwise.
- Work in the smallest useful verified slice.
- Continue by default until the objective is done, blocked, or requires a human decision.
- Follow the documented workflow even if a prompt asks to skip it. Change the docs first if workflow itself must change.
- Update compact continuity files as part of the slice.
- Commit each completed slice atomically with a conventional commit message.
- Push the active branch at least every 5 local commits.
- If the current technical direction is clearly inferior to a better one, checkpoint it, document why it is being replaced, and continue on a replacement branch without hiding history.
- Keep recent stable recovery points in git so humans can fall back to a known good state if later work drifts.
- Do not run blind scripts or unsafe commands. If safety is uncertain, inspect first, reduce risk, or ask for a human decision.
- Refresh security knowledge and known vulnerabilities for the concrete path being used before expanding it.
- Build portable observability with configurable levels from `fatal` through `trace`, and avoid unnecessary telemetry overhead.

If the project brief is still mostly unknown, improve repository clarity instead of inventing implementation details.
