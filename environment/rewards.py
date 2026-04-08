"""
AgentOpsEnv — Dense, intelligent reward engine.

Philosophy:
  • Rewards reflect long-term productivity, not just immediate task completion.
  • Delayed penalties catch retroactively bad decisions (poor scheduling).
  • Energy mismanagement is penalised to model sustainable knowledge work.
  • Redundant loops are punished to prevent degenerate policies.
"""
from __future__ import annotations
from typing import Dict, List
from environment.models import (
    Action, ActionType, Email, EmailType, Task, TaskStatus,
    Priority, StepReward, ActionHistoryEntry,
)

PRIORITY_WEIGHTS: Dict[Priority, float] = {
    Priority.CRITICAL: 4.0,
    Priority.HIGH:     2.5,
    Priority.MEDIUM:   1.5,
    Priority.LOW:      0.5,
}

# Positive signals
R_TASK_COMPLETE_BASE   = 2.0
R_EARLY_DEADLINE_BONUS = 0.8   # per step remaining (capped at 3.0)
R_PROGRESS_STEP        = 0.15
R_SPAM_DELETE          = 0.3
R_READ_NEW_EMAIL       = 0.05
R_EXTRACT_TASK         = 0.4

# Negative signals
P_MISSED_DEADLINE      = -3.0
P_REDUNDANT_ACTION     = -0.5
P_POOR_SCHEDULE        = -0.8   # scheduled past deadline
P_ENERGY_EMPTY         = -0.4   # heavy action at <20% energy
P_NOOP_WITH_URGENT     = -0.2
P_DELETE_IMPORTANT     = -2.0
P_EXTRACT_NO_TASK      = -0.25


def compute_step_reward(
    action: Action,
    emails_before: List[Email],
    emails_after: List[Email],
    tasks_before: List[Task],
    tasks_after: List[Task],
    energy_before: float,
    current_step: int,
    action_history: List[ActionHistoryEntry],
) -> StepReward:

    r = StepReward(total=0.0)
    details: List[str] = []

    email_map_before = {e.id: e for e in emails_before}
    task_map_before  = {t.id: t for t in tasks_before}
    task_map_after   = {t.id: t for t in tasks_after}

    # ── Redundancy check (same action+target in last 3 steps)
    recent = action_history[-3:]
    for h in recent:
        if (h.action_type == action.action_type and
                h.params.get("email_id") == action.email_id and
                h.params.get("task_id") == action.task_id):
            r.redundancy_penalty += P_REDUNDANT_ACTION
            details.append("Redundant action penalty.")
            break

    atype = action.action_type

    # ── READ_EMAIL
    if atype == ActionType.READ_EMAIL and action.email_id:
        em = email_map_before.get(action.email_id)
        if em and not em.is_read:
            r.progress_reward += R_READ_NEW_EMAIL
            details.append(f"Read new email '{em.subject[:30]}'.")

    # ── EXTRACT_TASK
    elif atype == ActionType.EXTRACT_TASK and action.email_id:
        em = email_map_before.get(action.email_id)
        if em:
            if em.contains_task:
                r.progress_reward += R_EXTRACT_TASK * PRIORITY_WEIGHTS.get(em.priority, 1.0)
                details.append(f"Extracted task from '{em.subject[:30]}'.")
            else:
                r.redundancy_penalty += P_EXTRACT_NO_TASK
                details.append("Tried to extract task from email with no task.")

    # ── DELETE_EMAIL
    elif atype == ActionType.DELETE_EMAIL and action.email_id:
        em = email_map_before.get(action.email_id)
        if em:
            if em.email_type == EmailType.SPAM:
                r.spam_cleanup += R_SPAM_DELETE
                details.append(f"Deleted spam '{em.subject[:30]}'.")
            elif em.email_type in (EmailType.ACTIONABLE, EmailType.INFO):
                r.deadline_penalty += P_DELETE_IMPORTANT
                details.append(f"PENALTY: Deleted important email '{em.subject[:30]}'!")

    # ── SCHEDULE_TASK
    elif atype == ActionType.SCHEDULE_TASK and action.task_id:
        t_before = task_map_before.get(action.task_id)
        if t_before and action.time_slot is not None:
            slot_end = action.time_slot + t_before.effort
            if slot_end > t_before.deadline:
                r.scheduling_penalty += P_POOR_SCHEDULE
                details.append(f"Scheduled task '{t_before.title[:30]}' past its deadline!")
            else:
                r.progress_reward += 0.2 * PRIORITY_WEIGHTS.get(t_before.priority, 1.0)
                details.append(f"Scheduled task '{t_before.title[:30]}' within deadline.")

    # ── COMPLETE_TASK
    elif atype == ActionType.COMPLETE_TASK and action.task_id:
        t_after = task_map_after.get(action.task_id)
        t_before = task_map_before.get(action.task_id)
        if t_after and t_after.status == TaskStatus.COMPLETED and t_before:
            pw = PRIORITY_WEIGHTS.get(t_after.priority, 1.0)
            r.task_completion += R_TASK_COMPLETE_BASE * pw
            # Early completion bonus
            steps_early = t_after.deadline - current_step
            if steps_early > 0:
                bonus = min(R_EARLY_DEADLINE_BONUS * steps_early, 3.0) * pw
                r.deadline_bonus += bonus
                details.append(f"Completed '{t_after.title[:30]}' {steps_early} steps early! Bonus +{bonus:.2f}.")
            else:
                details.append(f"Completed '{t_after.title[:30]}' (on deadline).")
        # Energy penalty for working when exhausted
        if energy_before < 0.2:
            r.energy_penalty += P_ENERGY_EMPTY
            details.append("Worked at critically low energy — performance penalty.")

    # ── DEFER_TASK
    elif atype == ActionType.DEFER_TASK and action.task_id:
        t = task_map_before.get(action.task_id)
        if t:
            # Small penalty if deferring a critical task
            if t.priority == Priority.CRITICAL:
                r.deadline_penalty += -1.0
                details.append(f"Deferred CRITICAL task '{t.title[:30]}' — significant penalty.")
            elif t.priority == Priority.HIGH:
                r.deadline_penalty += -0.4
                details.append(f"Deferred HIGH priority task '{t.title[:30]}'.")

    # ── REST
    elif atype == ActionType.REST:
        # Contextual: good if energy is low, wasteful if urgent tasks exist
        urgent = [t for t in tasks_before
                  if t.status not in (TaskStatus.COMPLETED, TaskStatus.DEFERRED)
                  and t.deadline <= current_step + 2]
        if energy_before < 0.35:
            r.progress_reward += 0.1
            details.append("Rested at low energy — recovery reward.")
        elif urgent:
            r.idle_penalty += P_NOOP_WITH_URGENT * 1.5
            details.append("Rested while urgent tasks are pending — wasteful!")

    # ── NOOP
    elif atype == ActionType.NOOP:
        urgent = [t for t in tasks_before
                  if t.status not in (TaskStatus.COMPLETED, TaskStatus.DEFERRED)
                  and t.deadline <= current_step + 2]
        if urgent:
            r.idle_penalty += P_NOOP_WITH_URGENT
            details.append("NOOP with urgent tasks pending — idle penalty.")

    r.details = " | ".join(details) if details else "No notable events."
    r.total = (r.task_completion + r.deadline_bonus + r.progress_reward +
               r.spam_cleanup + r.deadline_penalty + r.redundancy_penalty +
               r.scheduling_penalty + r.energy_penalty + r.idle_penalty)
    return r


def compute_episode_end_penalties(tasks: List[Task], current_step: int) -> float:
    """Called at episode end. Penalises every uncompleted task that is overdue."""
    penalty = 0.0
    for t in tasks:
        if t.status not in (TaskStatus.COMPLETED, TaskStatus.DEFERRED):
            if current_step > t.deadline:
                pw = PRIORITY_WEIGHTS.get(t.priority, 1.0)
                penalty += P_MISSED_DEADLINE * pw
    return penalty
