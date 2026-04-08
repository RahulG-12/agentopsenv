"""
AgentOpsEnv — Task definitions (easy / medium / hard).

Each task is a self-contained dict describing:
  - id, name, description, difficulty
  - env config (seed, max_steps)
  - success_criteria used by the grader
  - example_solution (optimal action sequence) for reference
"""
from __future__ import annotations
from typing import Any, Dict, List

TASKS: List[Dict[str, Any]] = [
    # ────────────────────────────────────────────────────────
    # TASK 1 — EASY
    # Goal: Clean the inbox. Delete all spam. Keep important emails.
    # ────────────────────────────────────────────────────────
    {
        "id": "task_easy_inbox_triage",
        "name": "Inbox Triage: Spam Removal",
        "description": (
            "You start with a cluttered inbox containing a mix of spam, informational, "
            "and actionable emails. Your goal is to identify and delete ALL spam emails "
            "without deleting any important (informational or actionable) emails. "
            "Read emails to identify their type before deleting. "
            "You do NOT need to extract tasks or schedule anything in this task — "
            "just get the inbox clean."
        ),
        "difficulty": "easy",
        "seed": 100,
        "max_steps": 30,
        "success_criteria": {
            "primary": "All spam emails deleted",
            "secondary": "No important emails deleted",
            "scoring_weights": {
                "task_completion": 0.1,   # not the focus
                "deadline_adherence": 0.1,
                "efficiency": 0.5,        # spam cleanup is the focus
                "resource": 0.3,
            }
        },
        "hints": [
            "Read each email before deciding to delete it.",
            "Spam emails come from suspicious external domains.",
            "Actionable emails often contain urgency words and internal senders.",
            "You can check email_type in the observation after reading.",
        ],
        "optimal_min_actions": 8,  # 4 reads + 4 deletes for typical easy set
        "example_solution": [
            "# Read each email first",
            "read_email(email_id=<id>) for each email",
            "# Delete only spam",
            "delete_email(email_id=<spam_id>) for each spam",
            "# Noop / rest for remaining steps",
        ],
    },

    # ────────────────────────────────────────────────────────
    # TASK 2 — MEDIUM
    # Goal: Extract tasks, schedule them by priority, complete before deadlines.
    # ────────────────────────────────────────────────────────
    {
        "id": "task_medium_task_extraction",
        "name": "Task Extraction & Deadline Scheduling",
        "description": (
            "Your inbox has 6 actionable emails, 2 ambiguous, 2 spam, and 3 informational emails. "
            "You must: (1) read all emails, (2) extract tasks from actionable emails, "
            "(3) schedule tasks in calendar slots respecting deadlines, "
            "(4) complete tasks before their deadlines. "
            "Tasks have varying effort (1–3 steps). Prioritise CRITICAL and HIGH tasks first. "
            "You have 30 steps and starting energy of 1.0 (decays 0.05/step, 0.10/completion). "
            "Resting recovers 0.22 energy but costs a step — plan wisely."
        ),
        "difficulty": "medium",
        "seed": 200,
        "max_steps": 50,
        "success_criteria": {
            "primary": "All CRITICAL and HIGH priority tasks completed before deadline",
            "secondary": "MEDIUM tasks completed; efficient action count",
            "scoring_weights": {
                "task_completion": 0.40,
                "deadline_adherence": 0.35,
                "efficiency": 0.15,
                "resource": 0.10,
            }
        },
        "hints": [
            "Read email → extract_task → schedule_task → complete_task (repeat per step of effort).",
            "Schedule high-priority tasks in early slots.",
            "If energy drops below 0.2, rest immediately — completing tasks at 0 energy yields no progress.",
            "Defer LOW priority tasks if you're running out of time.",
        ],
        "optimal_min_actions": 24,
        "example_solution": [
            "# Phase 1: Triage inbox (read + delete spam)",
            "# Phase 2: Extract tasks from actionable emails",
            "# Phase 3: Schedule critical/high tasks in early slots",
            "# Phase 4: Complete tasks (repeat complete_task for effort > 1)",
            "# Phase 5: Handle remaining medium tasks; defer low-priority ones",
        ],
    },

    # ────────────────────────────────────────────────────────
    # TASK 3 — MEDIUM-HARD
    # Goal: Sprint planning — deadline clustering, effort estimation
    # ────────────────────────────────────────────────────────
    {
        "id": "task_medium_sprint_planning",
        "name": "Sprint Planning Under Deadline Crunch",
        "description": (
            "It's the start of a sprint. Your inbox has 5 CRITICAL and HIGH tasks "
            "all due within a tight window, plus 3 MEDIUM and 2 LOW tasks. "
            "Several tasks have the SAME deadline — you cannot do all of them. "
            "You must triage, extract, and make hard prioritisation decisions. "
            "Deferred tasks incur a penalty proportional to their priority. "
            "Energy decays faster this sprint (fatigue from overwork). "
            "The optimal agent identifies the highest ROI tasks and defers the rest early."
        ),
        "difficulty": "medium",
        "seed": 777,
        "max_steps": 50,
        "success_criteria": {
            "primary": "Complete all CRITICAL tasks; complete at least 2 HIGH tasks",
            "secondary": "Defer LOW tasks early; maintain energy above 30% at episode end",
            "scoring_weights": {
                "task_completion": 0.45,
                "deadline_adherence": 0.35,
                "efficiency": 0.10,
                "resource": 0.10,
            }
        },
        "hints": [
            "Multiple tasks share the same deadline — pick the highest priority.",
            "Defer LOW tasks immediately after extracting them.",
            "CRITICAL tasks are worth 4× the score of LOW tasks.",
            "Resting at 35% energy is more efficient than working at 20%.",
        ],
        "optimal_min_actions": 28,
        "example_solution": [
            "# Read + extract CRITICAL emails first",
            "# Immediately complete CRITICAL tasks before reading rest",
            "# Defer LOW tasks right after extraction",
            "# Handle HIGH tasks next",
        ],
    },

    # ────────────────────────────────────────────────────────
    # TASK 4 — EXPERT
    # Goal: Production incident + normal work — context switching under pressure
    # ────────────────────────────────────────────────────────
    {
        "id": "task_expert_incident_response",
        "name": "Production Incident + Normal Workflow",
        "description": (
            "A production incident hits mid-week. Two CRITICAL emails arrive with "
            "2-step deadlines (fix the bug, notify stakeholders). Meanwhile, your "
            "normal inbox has 8 other tasks at various priorities. "
            "Your energy starts at 0.85 (already partially fatigued). "
            "You have 70 steps but the CRITICAL tasks MUST be done in the first 15 "
            "or they expire. This tests context-switching: drop everything for the "
            "incident, then return to normal work — a key real-world skill. "
            "Ambiguous emails from leadership may or may not contain urgent tasks — "
            "reading them costs a step but ignoring them risks missing a critical ask."
        ),
        "difficulty": "hard",
        "seed": 999,
        "max_steps": 70,
        "success_criteria": {
            "primary": "Both CRITICAL incident tasks completed within 15 steps",
            "secondary": "At least 50% of HIGH tasks completed; energy > 25% at end",
            "scoring_weights": {
                "task_completion": 0.40,
                "deadline_adherence": 0.40,
                "efficiency": 0.10,
                "resource": 0.10,
            }
        },
        "hints": [
            "The 2 CRITICAL tasks have deadline=15 — do them FIRST.",
            "Do NOT rest in the first 10 steps — energy is sufficient.",
            "After the incident, switch to normal workflow — sort by deadline.",
            "Some ambiguous emails contain HIGH priority hidden tasks — read them.",
            "You start at 85% energy — factor this into your rest schedule.",
        ],
        "optimal_min_actions": 45,
        "example_solution": [
            "# Steps 1-8:  Read CRITICAL emails → extract → complete (incident response)",
            "# Steps 9-15: Clean up incident aftermath, notify stakeholders",
            "# Steps 16-40: Normal triage — read remaining emails, extract tasks",
            "# Steps 41-65: Complete HIGH/MEDIUM tasks in deadline order",
            "# Steps 66-70: Defer LOW tasks, rest if needed",
        ],
    },

    # ────────────────────────────────────────────────────────
    # TASK 5 — HARD
    # Goal: Full workflow optimisation under conflict, noise, and fatigue.
    # ────────────────────────────────────────────────────────
    {
        "id": "task_hard_full_optimisation",
        "name": "Full Workflow Optimisation Under Constraints",
        "description": (
            "The most realistic and demanding scenario. You have 10 actionable emails "
            "(several with noisy, ambiguous bodies), 3 ambiguous emails (some contain hidden tasks), "
            "2 spam, and 2 informational. "
            "Multiple tasks share the same deadline, forcing hard prioritisation trade-offs. "
            "Energy decays faster (0.06/step, 0.12/completion). "
            "Some emails have intentionally misleading urgency cues. "
            "You have 40 steps to maximise weighted task completion. "
            "Key constraints:\n"
            "  • At most one task per calendar slot\n"
            "  • You cannot complete a task you haven't extracted\n"
            "  • Working at <20% energy yields zero progress\n"
            "  • Deleting actionable emails permanently loses those tasks\n"
            "  • Deferring CRITICAL tasks incurs a heavy penalty\n"
            "The optimal strategy requires triage → extraction → prioritised scheduling → "
            "energy-aware completion → selective deferral of low-value tasks."
        ),
        "difficulty": "hard",
        "seed": 300,
        "max_steps": 70,
        "success_criteria": {
            "primary": "Maximise priority-weighted task completion before deadlines",
            "secondary": "Efficient action use; healthy energy at episode end",
            "scoring_weights": {
                "task_completion": 0.40,
                "deadline_adherence": 0.30,
                "efficiency": 0.20,
                "resource": 0.10,
            }
        },
        "hints": [
            "Not all tasks can be completed — identify which to defer early.",
            "Read ambiguous emails carefully; some contain hidden tasks worth pursuing.",
            "CRITICAL + HIGH tasks have 4× and 2.5× the score weight of LOW tasks.",
            "Rest proactively at ~35% energy rather than waiting for exhaustion.",
            "Calendar conflicts: schedule different tasks in different slots.",
            "Use defer_task on LOW priority tasks when deadlines conflict.",
        ],
        "optimal_min_actions": 38,
        "example_solution": [
            "# Phase 1 (steps 1-8): Fast triage — read all emails, delete spam",
            "# Phase 2 (steps 9-15): Extract tasks from all actionable emails",
            "# Phase 3 (steps 16-20): Schedule tasks — CRITICAL first, then HIGH, MEDIUM, LOW",
            "# Phase 4 (steps 21-35): Complete tasks in priority order; rest at ~35% energy",
            "# Phase 5 (steps 36-40): Defer remaining LOW tasks; verify no missed deadlines",
        ],
    },
]


def get_task(task_id: str) -> Dict[str, Any]:
    for t in TASKS:
        if t["id"] == task_id:
            return t
    raise ValueError(f"Unknown task_id: {task_id}")


def list_tasks() -> List[Dict[str, Any]]:
    return [{"id": t["id"], "name": t["name"], "difficulty": t["difficulty"]} for t in TASKS]
