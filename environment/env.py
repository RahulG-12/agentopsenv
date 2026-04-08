"""
AgentOpsEnv — Core environment implementing the full OpenEnv spec.

step(action)  → (Observation, StepReward, done: bool, info: dict)
reset()       → Observation
state()       → dict  (serialisable snapshot)
"""
from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional, Tuple

from environment.models import (
    Action, ActionHistoryEntry, ActionType, CalendarSlot, Email, EmailType,
    EpisodeScore, Observation, Priority, StepReward, Task, TaskStatus,
)
from environment.generators import generate_email_set, generate_task_from_email
from environment.rewards import compute_step_reward, compute_episode_end_penalties
from environment.grader import grade_episode


class AgentOpsEnv:
    """
    OpenEnv-compliant environment simulating a knowledge-worker inbox.

    Parameters
    ----------
    difficulty : "easy" | "medium" | "hard"
    seed       : reproducibility seed
    max_steps  : episode horizon (default varies by difficulty)
    """

    DIFFICULTY_CONFIGS = {
        "easy":   dict(max_steps=30,  init_energy=1.0, energy_decay=0.03, energy_rest=0.25),
        "medium": dict(max_steps=50,  init_energy=1.0, energy_decay=0.04, energy_rest=0.22),
        "hard":   dict(max_steps=70,  init_energy=1.0, energy_decay=0.05, energy_rest=0.20),
    }

    def __init__(self, difficulty: str = "medium", seed: int = 42, max_steps: Optional[int] = None):
        assert difficulty in self.DIFFICULTY_CONFIGS, f"difficulty must be easy/medium/hard, got {difficulty}"
        self.difficulty = difficulty
        self.seed       = seed
        cfg             = self.DIFFICULTY_CONFIGS[difficulty]
        self.max_steps  = max_steps or cfg["max_steps"]
        self._cfg       = cfg

        # Mutable episode state
        self._step:         int              = 0
        self._energy:       float            = cfg["init_energy"]
        self._emails:       List[Email]      = []
        self._tasks:        List[Task]       = []
        self._calendar:     List[CalendarSlot] = []
        self._history:      List[ActionHistoryEntry] = []
        self._overdue:      List[str]        = []
        self._completed:    List[str]        = []
        self._spam_deleted: int              = 0
        self._total_spam:   int              = 0
        self._imp_deleted:  int              = 0
        self._done:         bool             = False
        self._final_score:  Optional[EpisodeScore] = None

    # ─────────────────────────────────────────────
    # OpenEnv API
    # ─────────────────────────────────────────────

    def reset(self) -> Observation:
        """Reset to initial state. Returns first observation."""
        cfg             = self._cfg
        self._step      = 0
        self._energy    = cfg["init_energy"]
        self._emails    = generate_email_set(self.difficulty, self.seed, 0, self.max_steps)
        self._tasks     = []
        self._calendar  = []
        self._history   = []
        self._overdue   = []
        self._completed = []
        self._spam_deleted = 0
        self._total_spam   = sum(1 for e in self._emails if e.email_type == EmailType.SPAM)
        self._imp_deleted  = 0
        self._done      = False
        self._final_score = None
        return self._observe()

    def step(self, action: Action) -> Tuple[Observation, StepReward, bool, Dict[str, Any]]:
        """
        Execute one action and advance the environment by one step.

        Returns
        -------
        observation : Observation
        reward      : StepReward
        done        : bool
        info        : dict
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        emails_before = copy.deepcopy(self._emails)
        tasks_before  = copy.deepcopy(self._tasks)
        energy_before = self._energy

        success, message = self._apply_action(action)

        # Energy decay (every step; rest recovers instead)
        if action.action_type == ActionType.REST:
            self._energy = min(1.0, self._energy + self._cfg["energy_rest"])
        else:
            decay = self._cfg["energy_decay"]
            # Heavy actions cost more energy
            if action.action_type in (ActionType.COMPLETE_TASK, ActionType.EXTRACT_TASK):
                decay *= 2.0
            self._energy = max(0.0, self._energy - decay)

        self._step += 1
        self._tick_deadlines()

        reward = compute_step_reward(
            action=action,
            emails_before=emails_before,
            emails_after=self._emails,
            tasks_before=tasks_before,
            tasks_after=self._tasks,
            energy_before=energy_before,
            current_step=self._step,
            action_history=self._history,
        )

        # Record history
        params: Dict[str, Any] = {}
        if action.email_id: params["email_id"] = action.email_id
        if action.task_id:  params["task_id"]  = action.task_id
        if action.time_slot is not None: params["time_slot"] = action.time_slot
        self._history.append(ActionHistoryEntry(
            step=self._step,
            action_type=action.action_type,
            params=params,
            reward_delta=reward.total,
            success=success,
            message=message,
        ))

        done = self._step >= self.max_steps or self._is_finished()
        if done:
            self._done = True
            end_penalty = compute_episode_end_penalties(self._tasks, self._step)
            reward.deadline_penalty += end_penalty
            reward.total += end_penalty
            self._final_score = grade_episode(
                tasks=self._tasks,
                action_history=self._history,
                total_steps=self._step,
                max_steps=self.max_steps,
                initial_energy=self._cfg["init_energy"],
                final_energy=self._energy,
                spam_deleted=self._spam_deleted,
                total_spam=self._total_spam,
                important_deleted=self._imp_deleted,
            )

        info: Dict[str, Any] = {
            "success": success,
            "message": message,
            "reward_breakdown": reward.model_dump(),
        }
        if done and self._final_score:
            info["episode_score"] = self._final_score.model_dump()

        return self._observe(), reward, done, info

    def state(self) -> Dict[str, Any]:
        """Return a fully serialisable snapshot of current state."""
        return json.loads(self._observe().model_dump_json())

    # ─────────────────────────────────────────────
    # Action handlers
    # ─────────────────────────────────────────────

    def _apply_action(self, action: Action) -> Tuple[bool, str]:
        atype = action.action_type

        if atype == ActionType.READ_EMAIL:
            return self._read_email(action.email_id)
        elif atype == ActionType.EXTRACT_TASK:
            return self._extract_task(action.email_id)
        elif atype == ActionType.DELETE_EMAIL:
            return self._delete_email(action.email_id)
        elif atype == ActionType.SCHEDULE_TASK:
            return self._schedule_task(action.task_id, action.time_slot)
        elif atype == ActionType.COMPLETE_TASK:
            return self._complete_task(action.task_id)
        elif atype == ActionType.DEFER_TASK:
            return self._defer_task(action.task_id)
        elif atype == ActionType.REST:
            return True, "Agent rested. Energy recovering."
        elif atype == ActionType.NOOP:
            return True, "No operation performed."
        return False, f"Unknown action: {atype}"

    def _read_email(self, email_id: Optional[str]) -> Tuple[bool, str]:
        e = self._find_email(email_id)
        if not e:
            return False, f"Email '{email_id}' not found."
        if e.is_read:
            return False, f"Email '{email_id}' already read."
        e.is_read = True
        return True, f"Read email: '{e.subject[:50]}'."

    def _extract_task(self, email_id: Optional[str]) -> Tuple[bool, str]:
        e = self._find_email(email_id)
        if not e:
            return False, f"Email '{email_id}' not found."
        if not e.is_read:
            return False, "Must read the email before extracting tasks."
        if not e.contains_task:
            return False, f"No extractable task in email '{email_id}'."
        # Check not already extracted
        for t in self._tasks:
            if t.source_email_id == email_id:
                return False, "Task already extracted from this email."
        task = generate_task_from_email(e, self._step, self.seed)
        self._tasks.append(task)
        return True, f"Extracted task: '{task.title}'."

    def _delete_email(self, email_id: Optional[str]) -> Tuple[bool, str]:
        e = self._find_email(email_id)
        if not e:
            return False, f"Email '{email_id}' not found."
        if e.email_type in (EmailType.ACTIONABLE, EmailType.INFO):
            self._imp_deleted += 1
        elif e.email_type == EmailType.SPAM:
            self._spam_deleted += 1
        self._emails = [x for x in self._emails if x.id != email_id]
        return True, f"Deleted email: '{e.subject[:50]}'."

    def _schedule_task(self, task_id: Optional[str], time_slot: Optional[int]) -> Tuple[bool, str]:
        t = self._find_task(task_id)
        if not t:
            return False, f"Task '{task_id}' not found."
        if time_slot is None:
            return False, "time_slot is required for schedule_task."
        if t.status == TaskStatus.COMPLETED:
            return False, "Cannot schedule a completed task."
        # Check for calendar conflicts
        for slot in self._calendar:
            if slot.time_step == time_slot:
                return False, f"Time slot {time_slot} already occupied by task '{slot.task_id}'."
        t.status = TaskStatus.SCHEDULED
        t.scheduled_slot = time_slot
        self._calendar.append(CalendarSlot(time_step=time_slot, task_id=t.id, duration=t.effort))
        return True, f"Scheduled '{t.title[:40]}' at step {time_slot}."

    def _complete_task(self, task_id: Optional[str]) -> Tuple[bool, str]:
        t = self._find_task(task_id)
        if not t:
            return False, f"Task '{task_id}' not found."
        if t.status == TaskStatus.COMPLETED:
            return False, "Task already completed."
        if t.status == TaskStatus.DEFERRED:
            return False, "Cannot complete a deferred task — un-defer first."
        # Energy reduces effectiveness at low levels
        effective_progress = 1
        if self._energy < 0.2:
            effective_progress = 0  # exhausted — no progress
        t.progress += effective_progress
        if t.progress >= t.effort:
            t.status = TaskStatus.COMPLETED
            t.completed_at = self._step
            self._completed.append(t.id)
            return True, f"Completed task: '{t.title[:40]}'."
        return True, f"Made progress on '{t.title[:40]}' ({t.progress}/{t.effort})."

    def _defer_task(self, task_id: Optional[str]) -> Tuple[bool, str]:
        t = self._find_task(task_id)
        if not t:
            return False, f"Task '{task_id}' not found."
        if t.status == TaskStatus.COMPLETED:
            return False, "Cannot defer a completed task."
        t.status = TaskStatus.DEFERRED
        return True, f"Deferred task: '{t.title[:40]}'."

    # ─────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────

    def _tick_deadlines(self):
        for t in self._tasks:
            if t.status in (TaskStatus.COMPLETED, TaskStatus.DEFERRED, TaskStatus.OVERDUE):
                continue
            steps_left = t.deadline - self._step
            if steps_left <= 0:
                t.status = TaskStatus.OVERDUE
                if t.id not in self._overdue:
                    self._overdue.append(t.id)

    def _observe(self) -> Observation:
        warnings = []
        for t in self._tasks:
            if t.status not in (TaskStatus.COMPLETED, TaskStatus.DEFERRED, TaskStatus.OVERDUE):
                steps_left = t.deadline - self._step
                if 0 < steps_left <= 2:
                    warnings.append(f"DEADLINE: '{t.title[:30]}' due in {steps_left} step(s)!")
        unread_spam = sum(1 for e in self._emails if e.email_type == EmailType.SPAM and not e.is_read)
        if unread_spam > 0:
            warnings.append(f"{unread_spam} spam email(s) still in inbox.")

        return Observation(
            step=self._step,
            time_remaining=self.max_steps - self._step,
            energy_level=round(self._energy, 3),
            emails=self._emails,
            tasks=self._tasks,
            calendar=self._calendar,
            action_history=self._history[-10:],   # last 10 for context window
            overdue_tasks=self._overdue,
            completed_tasks=self._completed,
            warnings=warnings,
        )

    def _find_email(self, email_id: Optional[str]) -> Optional[Email]:
        if not email_id:
            return None
        return next((e for e in self._emails if e.id == email_id), None)

    def _find_task(self, task_id: Optional[str]) -> Optional[Task]:
        if not task_id:
            return None
        return next((t for t in self._tasks if t.id == task_id), None)

    def _is_finished(self) -> bool:
        """Early termination only when all tasks resolved AND actionable emails fully processed."""
        active_tasks = [t for t in self._tasks
                        if t.status not in (TaskStatus.COMPLETED, TaskStatus.DEFERRED, TaskStatus.OVERDUE)]
        extracted_ids = {t.source_email_id for t in self._tasks}
        unprocessed_actionable = [
            e for e in self._emails
            if e.email_type == EmailType.ACTIONABLE and e.contains_task
            and e.id not in extracted_ids
        ]
        remaining_spam = [e for e in self._emails if e.email_type == EmailType.SPAM]
        return (len(active_tasks) == 0
                and len(unprocessed_actionable) == 0
                and len(remaining_spam) == 0)

    def get_final_score(self) -> Optional[EpisodeScore]:
        return self._final_score

    def render(self) -> str:
        """Human-readable state summary."""
        obs = self._observe()
        lines = [
            f"╔══ AgentOpsEnv [{self.difficulty.upper()}] Step {obs.step}/{self.max_steps} ══╗",
            f"  Energy : {'█' * int(obs.energy_level * 20):<20} {obs.energy_level*100:.0f}%",
            f"  Emails : {len(obs.emails)} in inbox ({sum(1 for e in obs.emails if not e.is_read)} unread)",
            f"  Tasks  : {len(obs.tasks)} total | {len(obs.completed_tasks)} done | {len(obs.overdue_tasks)} overdue",
        ]
        if obs.warnings:
            for w in obs.warnings:
                lines.append(f"  ⚠  {w}")
        if obs.action_history:
            last = obs.action_history[-1]
            lines.append(f"  Last   : [{last.action_type}] {last.message[:60]}")
        lines.append("╚" + "═" * 48 + "╝")
        return "\n".join(lines)
