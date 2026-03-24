# Agent Pipeline

A 6-agent sequential pipeline for Claude Code that processes code change requests through structured phases: scanning, planning, review, implementation, code review, and documentation.

## How It Works

When you open Claude Code in this repository and give it a task, the `CLAUDE.md` file at the repo root instructs it to follow this pipeline:

```
User Prompt → Scanner → Planner ⇄ Reviewer → Implementor ⇄ Code Reviewer → Documenter
```

Each agent is a specialized sub-agent with a specific role, constrained inputs, and structured JSON output. Artifacts are passed between agents via files in the `artifacts/` directory.

## Agents

| # | Agent | File | Role |
|---|-------|------|------|
| 1 | Scanner | `scanner.md` | Scans codebase, identifies affected areas, flags ambiguities |
| 2 | Planner | `planner.md` | Creates step-by-step implementation plan from scan report |
| 3 | Reviewer | `reviewer.md` | Critically evaluates the plan (adversarial — approval must be earned) |
| 4 | Implementor | `implementor.md` | Executes the approved plan by writing actual code |
| 5 | Code Reviewer | `code_reviewer.md` | Reviews code changes for correctness and plan fidelity |
| 6 | Documenter | `documenter.md` | Adds inline docs, changelog entries, and architecture notes |

## Key Features

- **Context isolation**: Each agent only sees what it needs. The Planner and Reviewer cannot access the codebase. The Implementor and Code Reviewer cannot see the original user prompt.
- **Revision loops**: The Reviewer can send the plan back to the Planner (max 3 cycles). The Code Reviewer can send code back to the Implementor (max 3 cycles).
- **Structured artifacts**: All inter-agent communication is via JSON files stored in `artifacts/`.
- **Progress logging**: A human-readable log is maintained at `artifacts/pipeline_log.md`.

## Artifacts Directory

The `artifacts/` directory is created at runtime and is gitignored. It contains:

| File | Producer | Description |
|------|----------|-------------|
| `user_prompt.md` | Orchestrator | The raw user prompt |
| `scan_report.json` | Scanner | Codebase analysis results |
| `implementation_plan.json` | Planner | Step-by-step change plan |
| `plan_review.json` | Reviewer | Plan approval or revision request |
| `implementation_result.json` | Implementor | Record of code changes made |
| `code_review.json` | Code Reviewer | Code approval or revision request |
| `documentation_package.json` | Documenter | Documentation change summary |
| `pipeline_log.md` | Orchestrator | Human-readable progress log |

## Usage

1. Open a terminal in this repository
2. Run `claude` to start Claude Code
3. Provide your change request as a prompt
4. Claude Code will automatically follow the pipeline defined in `CLAUDE.md`
5. Review the results and artifacts in `agent_pipeline/artifacts/`

## Customization

Each agent's behavior can be customized by editing its `.md` file. The orchestration logic is in the root `CLAUDE.md`. Key things you can adjust:

- **Max revision cycles**: Change the limit (default: 3) in `CLAUDE.md`
- **Agent prompts**: Edit individual `.md` files to change agent behavior
- **Output schemas**: Modify the JSON schemas in agent `.md` files
- **Context isolation**: Adjust what each agent receives in `CLAUDE.md`'s dependency map
