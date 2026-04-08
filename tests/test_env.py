"""
AgentOpsEnv — Test suite.
Tests environment correctness, reward logic, and grader determinism.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from environment import AgentOpsEnv, Action, ActionType
from environment.models import Priority, TaskStatus, EmailType
from environment.grader import grade_episode
from environment.rewards import compute_step_reward
from tasks import list_tasks, get_task


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def make_env(difficulty="easy", seed=42):
    env = AgentOpsEnv(difficulty=difficulty, seed=seed)
    env.reset()
    return env

def step(env, atype, **kwargs):
    action = Action(action_type=atype, **kwargs)
    return env.step(action)


# ──────────────────────────────────────────────────────────────
# 1. Reset & Observation
# ──────────────────────────────────────────────────────────────
class TestReset:
    def test_reset_returns_observation(self):
        env = AgentOpsEnv(difficulty="easy", seed=1)
        obs = env.reset()
        assert obs is not None
        assert obs.step == 0
        assert obs.energy_level == 1.0
        assert len(obs.emails) > 0

    def test_reset_idempotent(self):
        env = AgentOpsEnv(difficulty="medium", seed=99)
        obs1 = env.reset()
        obs2 = env.reset()
        assert len(obs1.emails) == len(obs2.emails)
        assert obs1.energy_level == obs2.energy_level

    def test_emails_have_required_fields(self):
        env = AgentOpsEnv(difficulty="hard", seed=7)
        obs = env.reset()
        for e in obs.emails:
            assert e.id
            assert e.subject
            assert e.body
            assert e.email_type in EmailType
            assert e.priority in Priority


# ──────────────────────────────────────────────────────────────
# 2. State
# ──────────────────────────────────────────────────────────────
class TestState:
    def test_state_is_serialisable(self):
        env = make_env()
        s = env.state()
        assert isinstance(s, dict)
        assert "emails" in s
        assert "tasks" in s
        assert "energy_level" in s

    def test_state_matches_observation(self):
        env = make_env()
        obs = env._observe()
        s = env.state()
        assert s["step"] == obs.step
        assert abs(s["energy_level"] - obs.energy_level) < 0.001


# ──────────────────────────────────────────────────────────────
# 3. Actions — determinism & effects
# ──────────────────────────────────────────────────────────────
class TestActions:
    def test_read_email_marks_as_read(self):
        env = make_env("easy", 10)
        obs0 = env.reset()
        email = obs0.emails[0]
        assert not email.is_read

        obs1, _, _, info = step(env, ActionType.READ_EMAIL, email_id=email.id)
        assert info["success"]
        read_email = next(e for e in obs1.emails if e.id == email.id)
        assert read_email.is_read

    def test_cannot_read_same_email_twice(self):
        env = make_env("easy", 10)
        obs0 = env.reset()
        eid = obs0.emails[0].id
        step(env, ActionType.READ_EMAIL, email_id=eid)
        _, _, _, info = step(env, ActionType.READ_EMAIL, email_id=eid)
        assert not info["success"]

    def test_delete_email_removes_it(self):
        env = make_env("easy", 10)
        obs0 = env.reset()
        eid = obs0.emails[0].id
        obs1, _, _, info = step(env, ActionType.DELETE_EMAIL, email_id=eid)
        assert info["success"]
        assert all(e.id != eid for e in obs1.emails)

    def test_extract_requires_read_first(self):
        env = make_env("medium", 20)
        obs0 = env.reset()
        actionable = next(e for e in obs0.emails if e.email_type == EmailType.ACTIONABLE)
        _, _, _, info = step(env, ActionType.EXTRACT_TASK, email_id=actionable.id)
        assert not info["success"]
        assert "read" in info["message"].lower()

    def test_extract_task_adds_to_tasks(self):
        env = make_env("medium", 20)
        obs0 = env.reset()
        actionable = next(e for e in obs0.emails if e.email_type == EmailType.ACTIONABLE)
        step(env, ActionType.READ_EMAIL, email_id=actionable.id)
        obs2, _, _, info = step(env, ActionType.EXTRACT_TASK, email_id=actionable.id)
        assert info["success"]
        assert len(obs2.tasks) == 1

    def test_schedule_task_populates_calendar(self):
        env = make_env("medium", 20)
        obs0 = env.reset()
        actionable = next(e for e in obs0.emails if e.email_type == EmailType.ACTIONABLE)
        step(env, ActionType.READ_EMAIL, email_id=actionable.id)
        obs2, _, _, _ = step(env, ActionType.EXTRACT_TASK, email_id=actionable.id)
        task = obs2.tasks[0]
        obs3, _, _, info = step(env, ActionType.SCHEDULE_TASK, task_id=task.id, time_slot=5)
        assert info["success"]
        assert any(s.task_id == task.id for s in obs3.calendar)

    def test_complete_task_tracks_progress(self):
        env = make_env("medium", 20)
        obs0 = env.reset()
        actionable = next(e for e in obs0.emails if e.email_type == EmailType.ACTIONABLE)
        step(env, ActionType.READ_EMAIL, email_id=actionable.id)
        obs2, _, _, _ = step(env, ActionType.EXTRACT_TASK, email_id=actionable.id)
        task = obs2.tasks[0]
        # Complete effort steps
        for _ in range(task.effort):
            obs_n, _, done, info = step(env, ActionType.COMPLETE_TASK, task_id=task.id)
            if done: break
        final_task = next((t for t in obs_n.tasks if t.id == task.id), None)
        assert final_task is not None
        assert final_task.status in (TaskStatus.COMPLETED, TaskStatus.PENDING)

    def test_defer_task_changes_status(self):
        env = make_env("medium", 20)
        obs0 = env.reset()
        actionable = next(e for e in obs0.emails if e.email_type == EmailType.ACTIONABLE)
        step(env, ActionType.READ_EMAIL, email_id=actionable.id)
        obs2, _, _, _ = step(env, ActionType.EXTRACT_TASK, email_id=actionable.id)
        task = obs2.tasks[0]
        obs3, _, _, info = step(env, ActionType.DEFER_TASK, task_id=task.id)
        assert info["success"]
        deferred = next(t for t in obs3.tasks if t.id == task.id)
        assert deferred.status == TaskStatus.DEFERRED

    def test_rest_recovers_energy(self):
        env = make_env("medium", 20)
        env.reset()
        # drain energy manually
        env._energy = 0.1
        obs1, _, _, _ = step(env, ActionType.REST)
        assert obs1.energy_level > 0.1

    def test_noop_steps_time(self):
        env = make_env()
        env.reset()
        obs1, _, _, _ = step(env, ActionType.NOOP)
        assert obs1.step == 1

    def test_step_after_done_raises(self):
        env = AgentOpsEnv(difficulty="easy", seed=42, max_steps=1)
        env.reset()
        step(env, ActionType.NOOP)
        with pytest.raises(RuntimeError):
            step(env, ActionType.NOOP)

    def test_invalid_email_id_fails_gracefully(self):
        env = make_env()
        env.reset()
        _, _, _, info = step(env, ActionType.READ_EMAIL, email_id="nonexistent")
        assert not info["success"]

    def test_calendar_conflict_prevented(self):
        env = make_env("medium", 20)
        obs0 = env.reset()
        # Extract two tasks
        count = 0
        for e in obs0.emails:
            if e.email_type == EmailType.ACTIONABLE and count < 2:
                step(env, ActionType.READ_EMAIL, email_id=e.id)
                step(env, ActionType.EXTRACT_TASK, email_id=e.id)
                count += 1
        obs_now = env._observe()
        t1, t2 = obs_now.tasks[0], obs_now.tasks[1]
        step(env, ActionType.SCHEDULE_TASK, task_id=t1.id, time_slot=10)
        _, _, _, info = step(env, ActionType.SCHEDULE_TASK, task_id=t2.id, time_slot=10)
        assert not info["success"]
        assert "occupied" in info["message"].lower()


# ──────────────────────────────────────────────────────────────
# 4. Reward correctness
# ──────────────────────────────────────────────────────────────
class TestRewards:
    def test_spam_delete_positive_reward(self):
        env = make_env("easy", 10)
        obs0 = env.reset()
        spam = next(e for e in obs0.emails if e.email_type == EmailType.SPAM)
        step(env, ActionType.READ_EMAIL, email_id=spam.id)
        _, reward, _, _ = step(env, ActionType.DELETE_EMAIL, email_id=spam.id)
        assert reward.spam_cleanup > 0

    def test_delete_important_negative_reward(self):
        env = make_env("medium", 20)
        obs0 = env.reset()
        important = next(e for e in obs0.emails
                         if e.email_type in (EmailType.ACTIONABLE, EmailType.INFO))
        _, reward, _, _ = step(env, ActionType.DELETE_EMAIL, email_id=important.id)
        assert reward.deadline_penalty < 0

    def test_redundant_action_penalty(self):
        env = make_env("easy", 10)
        obs0 = env.reset()
        eid = next(e for e in obs0.emails).id
        # Read once (success)
        step(env, ActionType.READ_EMAIL, email_id=eid)
        # Try to read again x3 to trigger redundancy
        for _ in range(3):
            _, reward, _, _ = step(env, ActionType.READ_EMAIL, email_id=eid)
        assert reward.redundancy_penalty < 0

    def test_reward_total_equals_sum(self):
        env = make_env("medium", 50)
        obs0 = env.reset()
        eid = obs0.emails[0].id
        _, reward, _, _ = step(env, ActionType.READ_EMAIL, email_id=eid)
        expected_total = (reward.task_completion + reward.deadline_bonus +
                          reward.progress_reward + reward.spam_cleanup +
                          reward.deadline_penalty + reward.redundancy_penalty +
                          reward.scheduling_penalty + reward.energy_penalty +
                          reward.idle_penalty)
        assert abs(reward.total - expected_total) < 1e-6


# ──────────────────────────────────────────────────────────────
# 5. Grader
# ──────────────────────────────────────────────────────────────
class TestGrader:
    def test_score_between_0_and_1(self):
        env = make_env("medium", 200)
        env.reset()
        done = False
        while not done:
            _, _, done, _ = step(env, ActionType.NOOP)
        score = env.get_final_score()
        assert score is not None
        assert 0.0 <= score.overall <= 1.0

    def test_score_is_deterministic(self):
        def run(seed):
            e = AgentOpsEnv(difficulty="easy", seed=seed, max_steps=5)
            e.reset()
            done = False
            while not done:
                _, _, done, _ = e.step(Action(action_type=ActionType.NOOP))
            return e.get_final_score().overall

        s1 = run(42)
        s2 = run(42)
        assert s1 == s2

    def test_completing_tasks_improves_score(self):
        """A run that completes tasks should outscore a do-nothing run."""
        def do_nothing(seed=42):
            e = AgentOpsEnv(difficulty="easy", seed=seed, max_steps=20)
            e.reset()
            done = False
            while not done:
                _, _, done, _ = e.step(Action(action_type=ActionType.NOOP))
            return e.get_final_score().overall

        def do_work(seed=42):
            e = AgentOpsEnv(difficulty="easy", seed=seed, max_steps=20)
            obs = e.reset()
            done = False
            spam_done = False
            for email in obs.emails[:4]:
                e.step(Action(action_type=ActionType.READ_EMAIL, email_id=email.id))
                if email.email_type == EmailType.SPAM:
                    e.step(Action(action_type=ActionType.DELETE_EMAIL, email_id=email.id))
                    spam_done = True
            while not done:
                _, _, done, _ = e.step(Action(action_type=ActionType.NOOP))
            return e.get_final_score().overall

        assert do_work() > do_nothing()


# ──────────────────────────────────────────────────────────────
# 6. Tasks
# ──────────────────────────────────────────────────────────────
class TestTasks:
    def test_list_tasks(self):
        tasks = list_tasks()
        assert len(tasks) >= 3
        diffs = {t["difficulty"] for t in tasks}
        assert "easy" in diffs and "medium" in diffs and "hard" in diffs

    def test_get_task(self):
        t = get_task("task_easy_inbox_triage")
        assert t["difficulty"] == "easy"
        assert "seed" in t
        assert "max_steps" in t

    def test_get_task_not_found(self):
        with pytest.raises(ValueError):
            get_task("nonexistent_task_id")


# ──────────────────────────────────────────────────────────────
# 7. Full episode
# ──────────────────────────────────────────────────────────────
class TestFullEpisode:
    def test_easy_episode_completes(self):
        env = AgentOpsEnv(difficulty="easy", seed=100, max_steps=20)
        obs = env.reset()
        done = False
        steps = 0
        while not done and steps < 30:
            obs, _, done, _ = env.step(Action(action_type=ActionType.NOOP))
            steps += 1
        assert done
        assert env.get_final_score() is not None

    def test_medium_episode_completes(self):
        env = AgentOpsEnv(difficulty="medium", seed=200, max_steps=30)
        env.reset()
        done = False
        while not done:
            _, _, done, _ = env.step(Action(action_type=ActionType.REST))
        assert env.get_final_score() is not None

    def test_hard_episode_completes(self):
        env = AgentOpsEnv(difficulty="hard", seed=300, max_steps=40)
        env.reset()
        done = False
        while not done:
            _, _, done, _ = env.step(Action(action_type=ActionType.NOOP))
        assert env.get_final_score() is not None

    def test_render_returns_string(self):
        env = make_env()
        env.reset()
        rendered = env.render()
        assert isinstance(rendered, str)
        assert "AgentOpsEnv" in rendered
