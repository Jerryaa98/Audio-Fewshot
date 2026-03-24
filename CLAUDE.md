# Multi-Agent Pipeline Orchestrator

When the user provides a task or change request, execute the 6-agent sequential pipeline described below. Do NOT make changes directly — always route through the pipeline.

## Pipeline Overview

```
User Prompt → Scanner → Planner ⇄ Reviewer → Implementor ⇄ Code Reviewer → Documenter → Done
                          ↑______|  (max 3)       ↑__________|  (max 3)
```

**Agents:**
1. **Scanner** — Scans codebase, finds affected areas, flags ambiguities
2. **Planner** — Creates step-by-step implementation plan
3. **Reviewer** — Critically evaluates the plan (adversarial)
4. **Implementor** — Executes the approved plan (writes code)
5. **Code Reviewer** — Reviews code changes for correctness
6. **Documenter** — Adds documentation to changed code

## Agent Dependency Map

| Agent | Receives | Output File | Codebase Access |
|-------|----------|-------------|-----------------|
| Scanner | user_prompt | scan_report.json | YES |
| Planner | user_prompt + scan_report | implementation_plan.json | NO |
| Reviewer | user_prompt + scan_report + implementation_plan | plan_review.json | NO |
| Implementor | scan_report + implementation_plan | implementation_result.json | YES |
| Code Reviewer | scan_report + implementation_plan + implementation_result | code_review.json | YES |
| Documenter | scan_report + implementation_plan + code_review | documentation_package.json | YES |

## Orchestration Protocol

When the user gives a task, follow these steps exactly:

### Step 0: Initialize
1. Create `agent_pipeline/artifacts/` directory if it doesn't exist
2. Write the user's prompt verbatim to `agent_pipeline/artifacts/user_prompt.md`
3. Initialize `agent_pipeline/artifacts/pipeline_log.md` with a header line

### Step 1: Run Scanner
1. Read `agent_pipeline/scanner.md`
2. Spawn an Agent (subagent_type: "general-purpose") with the scanner prompt + user prompt text appended under a `# YOUR INPUT` section
3. Extract the JSON output from the agent's response
4. Write it to `agent_pipeline/artifacts/scan_report.json`
5. Log: `Scanner [COMPLETED]` to pipeline_log.md

### Step 2: Run Planner (with revision loop)
1. Read `agent_pipeline/planner.md`
2. Read `agent_pipeline/artifacts/user_prompt.md` and `agent_pipeline/artifacts/scan_report.json`
3. Spawn an Agent with the planner prompt + user prompt + scan report appended under `# YOUR INPUTS`
4. Extract JSON, write to `agent_pipeline/artifacts/implementation_plan.json`
5. Log: `Planner [COMPLETED]`

### Step 3: Run Reviewer (with revision loop)
```
plan_cycle = 0
LOOP:
  1. Read agent_pipeline/reviewer.md
  2. Read user_prompt.md + scan_report.json + implementation_plan.json
  3. Spawn Agent with reviewer prompt + all three artifacts under # YOUR INPUTS
  4. Extract JSON, write to agent_pipeline/artifacts/plan_review.json
  5. IF verdict == "approved":
       Log: Reviewer [APPROVED]
       BREAK
  6. IF plan_cycle >= 3:
       Log: Reviewer [MAX CYCLES REACHED - proceeding with current plan]
       BREAK
  7. plan_cycle += 1
  8. Copy implementation_plan.json -> implementation_plan_v{plan_cycle}.json
  9. Log: Reviewer [CYCLE {plan_cycle} - REVISION REQUESTED]
  10. Re-run Planner with: planner prompt + user_prompt + scan_report + plan_review (revision request)
  11. Write new output to implementation_plan.json
  12. GOTO LOOP
```

### Step 4: Run Implementor (with revision loop)
1. Read `agent_pipeline/implementor.md`
2. Read `scan_report.json` + `implementation_plan.json` (NOT user_prompt — context isolation)
3. Spawn an Agent with the implementor prompt + scan report + implementation plan under `# YOUR INPUTS`
4. Extract JSON, write to `agent_pipeline/artifacts/implementation_result.json`
5. Log: `Implementor [COMPLETED]`

### Step 5: Run Code Reviewer (with revision loop)
```
code_cycle = 0
LOOP:
  1. Read agent_pipeline/code_reviewer.md
  2. Read scan_report.json + implementation_plan.json + implementation_result.json
  3. Spawn Agent with code reviewer prompt + all three artifacts under # YOUR INPUTS
  4. Extract JSON, write to agent_pipeline/artifacts/code_review.json
  5. IF verdict == "approved":
       Log: Code Reviewer [APPROVED]
       BREAK
  6. IF code_cycle >= 3:
       Log: Code Reviewer [MAX CYCLES REACHED - proceeding]
       BREAK
  7. code_cycle += 1
  8. Log: Code Reviewer [CYCLE {code_cycle} - REVISION REQUESTED]
  9. Re-run Implementor with: implementor prompt + scan_report + implementation_plan + code_review (revision request)
  10. Write new output to implementation_result.json
  11. GOTO LOOP
```

### Step 6: Run Documenter
1. Read `agent_pipeline/documenter.md`
2. Read `scan_report.json` + `implementation_plan.json` + `code_review.json` (NOT user_prompt — context isolation)
3. Spawn an Agent with the documenter prompt + all three artifacts under `# YOUR INPUTS`
4. Extract JSON, write to `agent_pipeline/artifacts/documentation_package.json`
5. Log: `Documenter [COMPLETED]`

### Step 7: Report to User
1. Read `agent_pipeline/artifacts/pipeline_log.md` and present a summary
2. List all files created/modified/deleted
3. Highlight any warnings, blocked steps, or revision cycles that occurred
4. Note any minor suggestions from reviewers

## Context Isolation Rules

These are CRITICAL and must be strictly followed:

- **Planner and Reviewer**: When spawning these agents, prepend to the prompt: `"IMPORTANT: You must NOT use Glob, Grep, Read, Edit, Write, or any file system tools. Work ONLY from the artifacts provided below."` They must not access the codebase.
- **Implementor**: Does NOT receive the user prompt. The plan is its only specification.
- **Code Reviewer**: Does NOT receive the user prompt. It evaluates against the plan only. It must NOT use Edit or Write tools — read-only access.
- **Documenter**: Does NOT receive the user prompt or revision history.

## JSON Extraction

When an agent returns its response, extract JSON using this approach:
1. Look for content between ` ```json ` and ` ``` ` markers
2. If no markers found, look for content between the first `{` and the last `}`
3. Parse the extracted text as JSON
4. If parsing fails, re-run the agent once with: `"Your previous output was not valid JSON. You MUST output a single valid JSON block between ```json markers. Try again."`
5. If second attempt fails, halt the pipeline and report the error to the user

## Error Handling

- **Agent failure**: If an Agent tool call fails, report the error and halt
- **Missing artifact**: If a required artifact file is missing, halt and report which artifact is missing
- **Max revision cycles**: After 3 failed review cycles, proceed with a warning — do not loop forever
- **Blocked steps**: If the implementor reports blocked steps, include them prominently in the final report

## Pipeline Log Format

Maintain `agent_pipeline/artifacts/pipeline_log.md` with this format:

```markdown
# Pipeline Log
## Task: <first 100 chars of user prompt>
### Scanner [COMPLETED]
- <key findings count>
### Planner [COMPLETED]
- <step count> steps planned
### Reviewer [APPROVED] / [CYCLE N - REVISION REQUESTED]
- <issues if any>
### Implementor [COMPLETED]
- <files changed count>
### Code Reviewer [APPROVED] / [CYCLE N - REVISION REQUESTED]
- <issues if any>
### Documenter [COMPLETED]
- <docs added count>
```
