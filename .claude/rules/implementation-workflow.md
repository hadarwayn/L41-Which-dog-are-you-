# Implementation Workflow

## Phase Order (MANDATORY — Never Skip)

```
P1: Planning ─────────► P2: Project Setup ─────► P3: Data Pipeline
                                                        │
P6: WSL Testing ◄───── P5: Training Infra ◄──── P4: Models
        │
P7: Colab Notebook ──► P8: Experiments ────────► P9: Visualization
        │
P10: README v1 ──────► [User runs Colab] ──────► P11: README v2
        │
P12: Final Cleanup
```

## Before Each Phase

1. Review `docs/tasks.json` for task dependencies
2. Ensure all prerequisite phases are complete
3. Update task status as you work

## Phase-Specific Rules

### P3: Data Pipeline
- **MUST** balance classes and standardize resolution BEFORE any training
- Verify pipeline with sample visualization
- Create 10% stratified subset for WSL testing

### P4: Models
- Each model in its own file under `src/models/`
- All models must have identical interface: `forward(x) → logits`
- Use `get_model(name)` factory pattern

### P5: Training Infrastructure
- Must work on both CPU (WSL) and GPU (Colab)
- Save metrics as JSON for later comparison
- Checkpoint after each transfer learning stage

### P6: WSL Local Testing
- Run ALL 6 models on 10% subset
- Use fewer epochs (3-5 per stage)
- Generate preliminary comparison results
- Fix any bugs before proceeding to Colab

### P7: Colab Notebook
- Must be self-contained (installs deps, mounts Drive)
- Include Results Export cell at the end
- Add markdown explanations between code cells

### P10: README v1 (Preliminary)
- Written from WSL 10% test results
- Label all results as "Preliminary (10% subset)"
- Include full CNN explanations and architecture descriptions

### P11: README v2 (Final)
- User pastes Colab results
- Replace preliminary results with final numbers
- Add celebrity and animal experiment sections

## Completion Criteria Per Task

A task is DONE only when:
- Code compiles and runs without errors
- Output matches expected format
- No regressions in existing functionality
- Task is marked "done" in tasks.json
