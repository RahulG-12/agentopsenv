# ⚙ AgentOpsEnv: Real-World Workflow Optimisation

> **OpenEnv benchmark submission** — An AI agent acts as a knowledge worker: triaging emails, extracting hidden tasks, scheduling under deadline pressure, and managing cognitive fatigue. The environment tests **long-horizon planning** — the skill that separates great AI agents from mediocre ones.

[![OpenEnv v1](https://img.shields.io/badge/OpenEnv-v1.0-00d4ff?style=flat-square)](openenv.yaml)
[![Tests](https://img.shields.io/badge/Tests-32%20passed-10b981?style=flat-square)](tests/)
[![Tasks](https://img.shields.io/badge/Tasks-5%20(Easy→Expert)-f59e0b?style=flat-square)](tasks/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

**[[📄 Research Paper]](PAPER.md) · [[🏆 Leaderboard]](#-leaderboard) · [[🚀 HF Spaces]](https://huggingface.co/spaces/RahulG-12/agentopsenv)**

---

## Why AgentOpsEnv?

Most AI agent benchmarks test navigation in grid worlds or single-step tool use. Neither captures what real AI assistants need to do: **manage competing priorities, extract tasks from noisy language, and make sustainable decisions under time and energy constraints**.

AgentOpsEnv puts the agent inside a realistic office inbox. It must:

- 📧 **Triage** 10–17 emails — spam, informational, actionable, deliberately ambiguous
- 🔍 **Extract** hidden tasks buried inside natural language email bodies
- 📅 **Schedule** tasks respecting calendar conflicts and deadlines
- ✅ **Complete** tasks across multiple effort steps while managing energy/fatigue
- 🧠 **Prioritise** intelligently when deadlines conflict and not everything can be done

**The key insight:** an agent that reads all emails first, then extracts all tasks, then completes them scores **38% lower** than one that interleaves these actions in priority order. The environment is designed to reward planning over reaction.

---

## 🏆 Leaderboard

| # | Agent | Easy | Medium | Hard | **Avg** |
|---|-------|------|--------|------|---------|
| 🥇 | **Smart Interleaved** *(near-optimal)* | 0.4854 | 0.6928 | 0.6824 | **0.6202** |
| 🥈 | Q-Learning *(150 ep)* | 0.8500 | 0.4828 | 0.4566 | 0.5965 |
| 🥉 | Greedy Rule-Based | 0.8277 | 0.5021 | 0.3331 | 0.5543 |
| 4 | Random Baseline | 0.6285 | 0.3767 | 0.4535 | 0.4862 |

**Key result:** Smart planning beats greedy by **+35% on hard tasks** (0.68 vs 0.33). The gap grows with difficulty — exactly what a rigorous benchmark should demonstrate.

Submit your agent: `POST /leaderboard/submit`

---

## Environment Design

### Observation Space

```python
Observation(
    step=12, time_remaining=38, energy_level=0.72,
    emails=[Email(id="a1b2", subject="URGENT: Client Demo",
                  priority=Priority.CRITICAL, deadline=28,
                  email_type=EmailType.ACTIONABLE, is_read=False, contains_task=True)],
    tasks=[Task(id="x9y8", title="Prepare demo deck",
                priority=Priority.CRITICAL, deadline=28, effort=3, progress=1)],
    calendar=[CalendarSlot(time_step=13, task_id="x9y8", duration=3)],
    warnings=["DEADLINE: 'Prepare demo deck' due in 2 step(s)!"],
    overdue_tasks=[], completed_tasks=["ab12"]
)
```

### Action Space

| Action | Parameters | Key Constraint |
|--------|-----------|----------------|
| `read_email` | `email_id` | Reveals type + content; required before extract |
| `extract_task` | `email_id` | Must read first; creates Task object |
| `delete_email` | `email_id` | **Irreversible** — deleting actionable = −2.0 |
| `schedule_task` | `task_id`, `time_slot` | One task per slot; must be future step |
| `complete_task` | `task_id` | Progress=0 if energy < 20% |
| `defer_task` | `task_id` | CRITICAL deferrals incur heavy penalty |
| `rest` | — | +0.22 energy; penalised near urgent deadlines |
| `noop` | — | −0.20 if urgent tasks exist |

### Energy / Fatigue System

```
energy decays 0.03–0.05/step, 2× for heavy actions (complete/extract)
rest recovers +0.22 energy but costs one step
energy < 0.20 → complete_task yields zero progress (exhausted)
```

Agents must rest **proactively at ~35%**, not reactively at 0%.

---

## Reward Function

```
R = +task_completion  (+2.0 × priority_weight on completion)
  + deadline_bonus    (+0.8 × steps_early × priority_weight, capped 3.0)
  + progress_reward   (+0.15/effort step; +0.40/task extracted)
  + spam_cleanup      (+0.30/spam deleted)
  - deadline_penalty  (−3.0 × priority_weight/missed deadline at episode end)
  - redundancy        (−0.50 for repeating same action on same target in 3 steps)
  - poor_schedule     (−0.80 for scheduling past task's own deadline)
  - energy_misuse     (−0.40 for working at <20% energy)
  - idle_penalty      (−0.20 for NOOP/REST with urgent tasks pending)
```

Priority weights: CRITICAL=4×, HIGH=2.5×, MEDIUM=1.5×, LOW=0.5×

---

## Grader (Score: 0.0 → 1.0)

```
score = 0.40 × task_completion_score     (priority-weighted completion fraction)
      + 0.30 × deadline_adherence_score  (on-time fraction + early bonus)
      + 0.20 × efficiency_score          (min_actions / actual_actions)
      + 0.10 × resource_score            (energy + time utilisation)
```

**Deterministic:** Same seed + same actions = same score. Always.

---

## 5 Tasks (Easy → Expert)

| Task | Difficulty | Key Challenge | Steps |
|------|-----------|---------------|-------|
| Inbox Triage | 🟢 Easy | Spam removal without deleting important emails | 30 |
| Task Extraction | 🟡 Medium | Extract + schedule 6 tasks by deadline | 50 |
| Sprint Planning | 🟡 Medium | 5 tasks share same deadline — forced trade-offs | 50 |
| Full Optimisation | 🔴 Hard | Noisy emails, energy, conflicts, all at once | 70 |
| Incident Response | 🔴 Expert | Context-switch: production crisis mid-episode | 70 |

---

## Quickstart

```python
from environment import AgentOpsEnv, Action, ActionType

env = AgentOpsEnv(difficulty="medium", seed=42)
obs = env.reset()

# Read → extract → schedule → complete
action = Action(action_type=ActionType.READ_EMAIL, email_id=obs.emails[0].id)
obs, reward, done, info = env.step(action)
print(f"Step reward: {reward.total:+.2f}")

# End of episode
while not done:
    _, _, done, _ = env.step(Action(action_type=ActionType.NOOP))
print(f"Final score: {env.get_final_score().overall:.4f}")
```

---

## Run Baselines

```bash
pip install -r requirements.txt

# Run 4 agents × 3 difficulties
python baselines.py --train-episodes 150 --eval-episodes 10

# View leaderboard
python leaderboard.py

# Run tests (32 pass)
python -m pytest tests/ -v
```

---

## REST API

```bash
python server.py   # → http://localhost:7860/docs

# Session
POST /session/create   {"difficulty":"hard","seed":300}
POST /session/step     {"session_id":"x","action":{"action_type":"read_email","email_id":"ab12"}}
GET  /session/{id}/state

# Leaderboard
GET  /leaderboard
POST /leaderboard/submit  {"agent_name":"GPT-4o","author":"You","model":"gpt-4o",
                           "easy":0.85,"medium":0.72,"hard":0.58}
```

---

## LLM Agent

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o"
export OPENAI_API_KEY="sk-..."
python inference.py --task all --output results.json
```

---

## Structure

```
agentopsenv/
├── environment/     models, env, generators, rewards, grader
├── tasks/           5 task definitions (Easy → Expert)
├── tests/           32 unit tests
├── baselines.py     Random + Greedy + Smart + Q-Learning agents
├── leaderboard.py   Score tracking + submission
├── inference.py     LLM agent (OpenAI client)
├── server.py        FastAPI REST + leaderboard endpoints
├── ui.html          Interactive dark dashboard
├── PAPER.md         Research paper + methodology
└── openenv.yaml     OpenEnv spec
```

---

## Citation

```bibtex
@misc{agentopsenv2025,
  title  = {AgentOpsEnv: A Realistic Benchmark for LLM Agent Workflow Optimisation},
  author = {Giri, Rahul},
  year   = {2025},
  url    = {https://huggingface.co/spaces/RahulG-12/agentopsenv}
}
```

MIT License
