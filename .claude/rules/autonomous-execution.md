# Autonomous Execution Policy

## Execution Mode

Operate in **autonomous execution mode** for all local development work.
You are a senior AI engineer building a deep learning project.
The user communicates at the orchestrator level — you handle implementation details.

### Rules

1. **DO NOT** ask for confirmations before executing local commands (WSL)
2. **DO NOT** pause execution for user validation on code/tests
3. **DO NOT** ask clarification questions unless there is a true blocking ambiguity
4. **DO** ask before any destructive operations (deleting data, force-pushing)
5. **DO** ask before actions that affect shared state (Colab notebook, Google Drive)
6. **DO** follow the phase-by-phase workflow (`implementation-workflow.md`)

### Definition of Blocking Ambiguity

Only pause execution for:
- Conflicting requirements that cannot be resolved
- Missing dataset credentials (Kaggle API key)
- Actions requiring user's Colab environment
- Security-sensitive operations

If the task can reasonably be inferred, make the best engineering decision and proceed.

## Task Execution Protocol

1. Check `docs/tasks.json` for current task status
2. Pick the next pending task in phase order
3. Implement the solution
4. Test locally (WSL, 10% data)
5. Mark task as done
6. Move to next task
7. Report progress at natural milestones (phase completion)

## Error Resolution

1. Read the error message carefully
2. Form a hypothesis about the cause
3. Fix the root cause (not symptoms)
4. Verify the fix works
5. Never retry the identical failing command blindly

## What Requires User Action

- Running the Colab notebook (full training)
- Providing Kaggle API credentials (if not available)
- Granting Colab notebook access
- Pasting Colab results back for README update
- Final approval of README
