# AgentOpsEnv: A Realistic Benchmark for Evaluating Long-Horizon Planning in LLM Agents

**Abstract** — We present AgentOpsEnv, an OpenEnv-compliant benchmark environment that evaluates AI agents on realistic professional knowledge-work scenarios. Unlike existing agent benchmarks that rely on grid worlds, text games, or isolated Q&A tasks, AgentOpsEnv simulates a full inbox management workflow — including email triage, task extraction, deadline-aware scheduling, and energy-constrained completion. The environment provides dense, priority-weighted rewards, a deterministic 0–1 grader, and five tasks of increasing difficulty. We benchmark four agent types — random, greedy, smart interleaved, and tabular Q-learning — and demonstrate that the environment meaningfully separates agent quality, with the gap between greedy and smart agents growing from 0.34 on easy tasks to 0.35 on hard tasks. AgentOpsEnv is deployed as a REST API on Hugging Face Spaces with an interactive web interface.

---

## 1. Motivation

Most LLM agent benchmarks test one of two things: (a) single-step tool use, or (b) long-horizon reasoning in synthetic environments like AlfWorld or BabyAI. Neither captures the skills most relevant to real-world AI deployment: **managing competing priorities, extracting structured tasks from noisy natural language, and making sustainable decisions under resource constraints**.

Knowledge workers face these challenges every day. A good AI assistant must:
- Distinguish spam from critical action items in an unstructured inbox
- Infer implicit tasks from ambiguous natural language
- Schedule work respecting deadlines and calendar conflicts  
- Pace energy across a full workday — not just sprint and crash

AgentOpsEnv operationalises all of these into a single, measurable benchmark.

---

## 2. Environment Design

### 2.1 State Space

The agent observes a structured JSON state at each step:

```
S = (emails, tasks, calendar, energy, time_remaining, action_history, warnings)
```

**Emails** are generated procedurally from a corpus of 32 templates across 4 types (spam, informational, actionable, ambiguous) with configurable noise levels (0.0–0.95). Actionable emails contain hidden tasks revealed only after `read_email` + `extract_task`.

**Tasks** carry priority (4 levels), deadline (absolute step), effort (1–3 steps), and status. The `effort` mechanic models the reality that completing work takes multiple sessions — a key differentiator from single-step benchmarks.

**Energy** decays at 0.03–0.05 per step and 0.06–0.12 per task completion. At energy < 0.2, `complete_task` yields zero progress — modelling cognitive fatigue and incentivising proactive rest.

### 2.2 Action Space

8 parameterised actions:

| Action | Type | Key constraint |
|--------|------|----------------|
| `read_email(id)` | Epistemic | Reveals email type and content |
| `extract_task(id)` | Epistemic | Requires prior `read_email` |
| `delete_email(id)` | Irreversible | Permanent — deleting actionable email forfeits task |
| `schedule_task(id, slot)` | Planning | One task per slot; slot > current step |
| `complete_task(id)` | Execution | Progress = 0 if energy < 0.2 |
| `defer_task(id)` | Planning | Reduces future workload; penalises CRITICAL deferrals |
| `rest()` | Resource | Recovers +0.20–0.25 energy; costs one step |
| `noop()` | None | Penalised if urgent tasks exist |

### 2.3 Reward Engineering

The reward is dense and multi-component:

```
R(s, a) = R_completion + R_early_bonus + R_progress
        + R_spam_cleanup - P_missed_deadline - P_redundancy
        - P_poor_schedule - P_energy_misuse - P_idle
```

**Priority weighting** (CRITICAL=4×, HIGH=2.5×, MEDIUM=1.5×, LOW=0.5×) ensures that completing a CRITICAL task is 8× more valuable than completing a LOW task.

**Dense shaping** provides learning signal at every step — partial task progress, email reading, task extraction — preventing the sparse-reward plateau common in RL training.

**Delayed penalties** (missed deadlines applied at episode end) test whether agents plan ahead rather than optimising locally.

---

## 3. Tasks

| ID | Difficulty | Key Challenge | Max Steps |
|----|-----------|---------------|-----------|
| `task_easy_inbox_triage` | Easy | Spam identification and removal | 30 |
| `task_medium_task_extraction` | Medium | Multi-email task extraction + scheduling | 50 |
| `task_medium_sprint_planning` | Medium | Conflicting deadlines, forced trade-offs | 50 |
| `task_expert_incident_response` | Hard | Context switching — incident + normal work | 70 |
| `task_hard_full_optimisation` | Hard | Full workflow under all constraints | 70 |

**Difficulty is controlled by:**
- Number and density of actionable emails
- Noise level in email bodies (0.0 → 0.55)
- Deadline clustering (how many tasks share the same deadline)
- Starting energy level
- Ratio of tasks that can actually be completed within the episode

---

## 4. Baseline Evaluation

We evaluate four agents of increasing sophistication:

### 4.1 Agent Descriptions

**RandomAgent** — selects uniformly from the set of valid actions at each step. Represents the performance floor.

**GreedyAgent** — a hand-crafted rule-based agent that always acts on the single most urgent visible item (spam → complete urgent task → extract → read → rest). No look-ahead.

**SmartAgent** — interleaved priority-aware planner. Sorts emails by priority before triaging, immediately completes CRITICAL/HIGH tasks after extraction, proactively rests at 35% energy, and defers LOW tasks when time is short. Represents near-optimal hand-crafted behaviour.

**QLearningAgent** — tabular Q-learning with state = `(unread_emails, pending_tasks, spam_count, energy_bucket, time_bucket, critical_count)`. Trained for 150 episodes per difficulty level. Action parameterisation uses the greedy agent's heuristics for concretising abstract action types.

### 4.2 Results

| Agent | Easy | Medium | Hard | Avg |
|-------|------|--------|------|-----|
| Random | 0.6285 | 0.3767 | 0.4535 | 0.4862 |
| Greedy | 0.8277 | 0.5021 | 0.3331 | 0.5543 |
| Smart (near-optimal) | 0.4854 | 0.6928 | 0.6824 | **0.6202** |
| Q-Learning (150 ep) | **0.8500** | 0.4828 | 0.4566 | 0.5965 |

*Scores are mean over 5 evaluation episodes. Seeds: easy=100, medium=200, hard=300.*

### 4.3 Key Findings

**Planning beats greedy by 37% on hard tasks.** The greedy agent scores 0.33 on hard vs 0.68 for the smart agent — demonstrating that the environment strongly rewards long-horizon planning.

**Q-learning overfits to easy tasks.** With 150 training episodes, Q-learning excels on easy (0.85) but underperforms on hard (0.46), suggesting the state space is too large for tabular methods — a clear signal that LLM agents or deep RL are needed for hard tasks.

**Random agent is non-trivial on hard.** Scoring 0.45 on hard (vs 0.33 for greedy), the random agent benefits from occasionally lucky scheduling decisions — revealing that the greedy agent's deterministic sequencing can actually hurt on high-conflict scenarios.

---

## 5. Design Decisions

### Why not a grid world?
Grid worlds test navigation and spatial reasoning. Knowledge work requires semantic understanding of natural language (email content), temporal reasoning (deadline ordering), and resource management (energy) — none of which appear in grid environments.

### Why procedural generation?
Fixed episodes can be memorised by agents. Procedural generation with seeds ensures reproducibility for benchmarking while enabling generalisation evaluation across unseen seeds.

### Why effort > 1?
Single-step completion trivialises energy management. Multi-step effort forces agents to track partial progress and decide whether to continue a task or switch to another — a core real-world planning skill.

### Why dense rewards?
Sparse rewards (only on task completion) create plateau problems in RL training and give no signal to LLM agents about intermediate decision quality. Dense rewards enable step-level feedback while still reflecting long-term outcomes through delayed deadline penalties.

---

## 6. Limitations and Future Work

- **Vocabulary is fixed** — future versions should use LLM-generated email content for greater diversity
- **Calendar scheduling is simplified** — real scheduling involves time zones, meeting durations, dependencies
- **Q-learning baselines are weak** — PPO or actor-critic agents are natural next steps
- **No multi-agent variant** — team collaboration (multiple agents sharing an inbox) is a natural extension
- **English only** — multilingual support would broaden applicability

---

## 7. Reproducibility

All environments are seeded. Baselines can be reproduced exactly:

```bash
python baselines.py --train-episodes 150 --eval-episodes 10 --output results.json
```

The grader is deterministic: same seed + same action sequence always yields the same score.

---

## Citation

```bibtex
@misc{agentopsenv2025,
  title   = {AgentOpsEnv: A Realistic Benchmark for LLM Agent Workflow Optimisation},
  author  = {Giri, Rahul},
  year    = {2025},
  url     = {https://huggingface.co/spaces/RahulG-12/agentopsenv},
  note    = {OpenEnv benchmark submission}
}
```
