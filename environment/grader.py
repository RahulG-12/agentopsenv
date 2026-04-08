"""
AgentOpsEnv — Deterministic grader.

Score = weighted sum of four sub-scores, each in [0, 1]:
  1. Task Completion   (40%) — fraction of non-spam tasks completed
  2. Deadline Adherence(30%) — fraction completed on/before deadline, priority-weighted
  3. Efficiency        (20%) — how close agent was to minimum viable action count
  4. Resource Usage    (10%) — energy discipline and time usage
"""
from __future__ import annotations
from typing import Any, Dict, List
from environment.models import (
    Task, TaskStatus, Priority, ActionHistoryEntry, EpisodeScore,
)

PRIORITY_WEIGHTS = {
    Priority.CRITICAL: 4.0,
    Priority.HIGH:     2.5,
    Priority.MEDIUM:   1.5,
    Priority.LOW:      0.5,
}

W_COMPLETION  = 0.40
W_DEADLINE    = 0.30
W_EFFICIENCY  = 0.20
W_RESOURCE    = 0.10


def grade_episode(
    tasks: List[Task],
    action_history: List[ActionHistoryEntry],
    total_steps: int,
    max_steps: int,
    initial_energy: float,
    final_energy: float,
    spam_deleted: int,
    total_spam: int,
    important_deleted: int,
) -> EpisodeScore:

    # ── 1. Task Completion Score
    completable = [t for t in tasks if t.priority != Priority.LOW or t.effort > 0]
    if not completable:
        completion_score = 1.0
    else:
        pw_total = sum(PRIORITY_WEIGHTS[t.priority] for t in completable)
        pw_done  = sum(PRIORITY_WEIGHTS[t.priority] for t in completable
                       if t.status == TaskStatus.COMPLETED)
        completion_score = pw_done / pw_total if pw_total else 0.0

    # ── 2. Deadline Adherence Score
    completed_tasks = [t for t in tasks if t.status == TaskStatus.COMPLETED]
    if not completable:
        deadline_score = 1.0
    else:
        pw_total = sum(PRIORITY_WEIGHTS[t.priority] for t in completable)
        pw_on_time = 0.0
        for t in completed_tasks:
            if t.completed_at is not None and t.completed_at <= t.deadline:
                steps_early = max(0, t.deadline - t.completed_at)
                early_bonus = min(1.0, steps_early * 0.1)
                pw_on_time += PRIORITY_WEIGHTS[t.priority] * (1.0 + early_bonus)
            elif t.completed_at is not None:
                # Partial credit for late completion (50%)
                pw_on_time += PRIORITY_WEIGHTS[t.priority] * 0.5
        deadline_score = min(1.0, pw_on_time / pw_total) if pw_total else 0.0

    # ── 3. Efficiency Score
    # Minimum viable actions = (emails to read + tasks to extract + tasks to schedule + tasks to complete)
    min_actions = max(1, len(completable) * 3 + total_spam)
    actual_actions = len(action_history)
    efficiency_score = max(0.0, min(1.0, min_actions / actual_actions)) if actual_actions else 0.0

    # Penalty for deleting important emails
    if important_deleted > 0:
        efficiency_score = max(0.0, efficiency_score - 0.2 * important_deleted)

    # ── 4. Resource Usage Score
    # Energy discipline: good if energy > 20% at end, great if > 40%
    energy_score = min(1.0, final_energy / initial_energy + 0.2) if initial_energy > 0 else 0.5
    # Time discipline: penalise wasted time
    time_score = 1.0 - max(0.0, (total_steps - max_steps * 0.85) / max_steps)
    resource_score = min(1.0, (energy_score * 0.6 + time_score * 0.4))

    # ── Spam cleanup bonus (absorbed into efficiency)
    if total_spam > 0:
        spam_bonus = (spam_deleted / total_spam) * 0.05
        efficiency_score = min(1.0, efficiency_score + spam_bonus)

    overall = (
        W_COMPLETION * completion_score +
        W_DEADLINE   * deadline_score   +
        W_EFFICIENCY * efficiency_score +
        W_RESOURCE   * resource_score
    )

    breakdown: Dict[str, Any] = {
        "tasks_total":       len(completable),
        "tasks_completed":   len(completed_tasks),
        "tasks_overdue":     sum(1 for t in tasks if t.status == TaskStatus.OVERDUE),
        "tasks_deferred":    sum(1 for t in tasks if t.status == TaskStatus.DEFERRED),
        "spam_deleted":      spam_deleted,
        "total_spam":        total_spam,
        "important_deleted": important_deleted,
        "total_actions":     actual_actions,
        "min_actions_est":   min_actions,
        "final_energy":      round(final_energy, 2),
        "sub_scores": {
            "completion":  round(completion_score, 4),
            "deadline":    round(deadline_score, 4),
            "efficiency":  round(efficiency_score, 4),
            "resource":    round(resource_score, 4),
        }
    }

    return EpisodeScore(
        overall=round(overall, 4),
        task_completion_score=round(completion_score, 4),
        deadline_adherence_score=round(deadline_score, 4),
        efficiency_score=round(efficiency_score, 4),
        resource_score=round(resource_score, 4),
        breakdown=breakdown,
    )
