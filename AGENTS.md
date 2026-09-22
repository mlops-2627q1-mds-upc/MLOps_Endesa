# Agent Instructions for MLOps_Endesa

These instructions apply to the entire repository. Read
[CONTRIBUTING.md](CONTRIBUTING.md) and the
[EDN index](docs/edn/README.md) before making changes. The working agreement is the
canonical workflow policy; this file describes how agents should apply it.

## Scope and collaboration

- Work within the user's authorized task. Preserve unrelated changes; inspect
  `git status` and the relevant files before editing. Do not reset, clean, or
  overwrite another contributor's work.
- Follow GitHub Flow using a short-lived task branch. Link the issue when one
  exists. Before issue tracking is configured, use a descriptive branch and record
  the bootstrap context without inventing an issue number.
- Keep changes reviewable and use the repository's PR template. Respect the
  recorded merge and review policy. Do not supply an agent's own approval as the
  required human peer review or change authorship to imply another person worked
  on the change.
- Proceed with authorized local work without repeatedly requesting confirmation.
  Creating documents does not by itself authorize posting messages, publishing
  artifacts, configuring external services, spending money, or deploying.
- Use only the user's explicitly authorized identity for authenticated services.
  Do not expose or commit credentials.

## Evidence and engineering decisions

- Write repository documentation in clear English for teammates and course
  instructors. Explain the problem, choice, rationale, and evidence without claims
  about grades or unsupported claims of quality.
- Keep prose short and specific to this project. Prefer concrete constraints,
  alternatives, and trade-offs to promotional language or generic assurances.
  Remove repeated caveats and unnecessary headings; keep tables for comparisons
  and numbered lists for procedures.
- Keep each document focused: README for the project, CONTRIBUTING for the working
  rules, EDN for decisions, and issues/setup notes for operational progress.
- Credit human proposals and choices accurately. Describe AI's actual support
  briefly in the EDN; retain the course's involvement, response, and assessment
  fields. Prompt transcripts are optional and should appear only when useful.
  Do not invent deliberation, consensus, or a participant's assessment to make a
  record more persuasive, or reduce substantive AI work to proofreading.
- Distinguish a documented policy, implemented configuration, successful check,
  and verified deployment. State limitations and checks not performed.
- Record consequential decisions using the EDN fields. Name only participants
  actually involved. Record AI involvement and the observed human response;
  never infer consensus, assessment, approval, meetings, or experiment results.
- Keep the EDN index and rubric evidence links consistent with actual artifacts.
  Do not mark remote setup complete based on local files alone.
- Attribute course examples and external sources when adapting them. Check the
  current project code and requirements before reusing demo code.

## MLOps changes and validation

- Keep raw datasets, model weights, local environments, credentials, and temporary
  outputs out of Git. Preserve DVC metadata and dependency locks when introduced.
  Check ignore rules when adding generated artifacts or new tooling.
- Preserve traceability between code, data, configuration, runs, and model
  artifacts. Do not silently overwrite shared experiment or deployment state.
- For forecasting changes, check chronological split boundaries, preprocessing
  leakage, units, timestamp semantics, and comparable evaluation conditions.
  Do not tune against the final test set or label an unverified evaluation as
  leakage-free.
- Run checks appropriate to the change. For documentation, check links,
  consistency, and templates. For behavior changes, run or add meaningful tests
  for the affected behavior. Do not start full training or costly compute as a
  side effect of a documentation or routine review task.
- Report what changed, what was checked, and what remains pending. Do not claim
  model improvement, reproducibility, CI enforcement, or deployment success
  without the corresponding evidence.
