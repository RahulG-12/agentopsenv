"""
AgentOpsEnv — Baseline Agents Suite.

Implements 4 agents of increasing sophistication:
  1. RandomAgent      — picks valid actions uniformly at random
  2. GreedyAgent      — always acts on the most urgent visible item
  3. SmartAgent       — interleaved priority-aware planning (hand-crafted optimal)
  4. QLearningAgent   — tabular Q-learning trained online across episodes

Run:
    python baselines.py [--episodes 200] [--output baseline_results.json]
"""
from __future__ import annotations
import argparse, json, random, collections
from typing import Any, Dict, List, Optional, Tuple
from environment import AgentOpsEnv, Action, ActionType
from environment.models import EmailType, Priority, TaskStatus, Observation

PW = {Priority.CRITICAL: 4.0, Priority.HIGH: 2.5, Priority.MEDIUM: 1.5, Priority.LOW: 0.5}

# ─────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────
class BaseAgent:
    name: str = "base"

    def act(self, obs: Observation) -> Action:
        raise NotImplementedError

    def on_episode_start(self, obs: Observation): pass
    def on_step(self, obs: Observation, action: Action, reward: float, done: bool): pass


# ─────────────────────────────────────────────
# 1. Random Agent
# ─────────────────────────────────────────────
class RandomAgent(BaseAgent):
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def act(self, obs: Observation) -> Action:
        choices = [Action(action_type=ActionType.NOOP),
                   Action(action_type=ActionType.REST)]
        if obs.emails:
            e = self.rng.choice(obs.emails)
            choices += [
                Action(action_type=ActionType.READ_EMAIL, email_id=e.id),
                Action(action_type=ActionType.DELETE_EMAIL, email_id=e.id),
            ]
            if e.is_read and e.contains_task:
                choices.append(Action(action_type=ActionType.EXTRACT_TASK, email_id=e.id))
        if obs.tasks:
            t = self.rng.choice(obs.tasks)
            choices += [
                Action(action_type=ActionType.COMPLETE_TASK, task_id=t.id),
                Action(action_type=ActionType.DEFER_TASK, task_id=t.id),
                Action(action_type=ActionType.SCHEDULE_TASK, task_id=t.id,
                       time_slot=self.rng.randint(obs.step + 1, obs.step + 10)),
            ]
        return self.rng.choice(choices)


# ─────────────────────────────────────────────
# 2. Greedy Agent
# ─────────────────────────────────────────────
class GreedyAgent(BaseAgent):
    """Always acts on the single most urgent visible item."""
    name = "greedy"

    def act(self, obs: Observation) -> Action:
        # 1. Delete spam if seen
        for e in obs.emails:
            if e.email_type == EmailType.SPAM and e.is_read:
                return Action(action_type=ActionType.DELETE_EMAIL, email_id=e.id)

        # 2. Complete most-urgent pending task
        pending = [t for t in obs.tasks
                   if t.status not in (TaskStatus.COMPLETED, TaskStatus.DEFERRED, TaskStatus.OVERDUE)]
        if pending:
            t = min(pending, key=lambda x: (x.deadline, -PW.get(x.priority, 1)))
            return Action(action_type=ActionType.COMPLETE_TASK, task_id=t.id)

        # 3. Extract task from read actionable email
        for e in obs.emails:
            if e.is_read and e.contains_task and e.email_type == EmailType.ACTIONABLE:
                already = any(tk.source_email_id == e.id for tk in obs.tasks)
                if not already:
                    return Action(action_type=ActionType.EXTRACT_TASK, email_id=e.id)

        # 4. Read unread emails
        for e in obs.emails:
            if not e.is_read:
                return Action(action_type=ActionType.READ_EMAIL, email_id=e.id)

        # 5. Rest if energy low
        if obs.energy_level < 0.4:
            return Action(action_type=ActionType.REST)

        return Action(action_type=ActionType.NOOP)


# ─────────────────────────────────────────────
# 3. Smart Interleaved Agent (near-optimal)
# ─────────────────────────────────────────────
class SmartAgent(BaseAgent):
    """
    Interleaved priority-aware planner:
    - Sorts emails by priority (CRITICAL/HIGH actionable first)
    - Read → extract → immediately schedule+complete CRITICAL/HIGH
    - Then handles MEDIUM tasks
    - Defers LOW tasks when time is tight
    - Rests proactively at 35% energy
    """
    name = "smart"

    def __init__(self):
        self._phase = "triage"       # triage | execute | cleanup
        self._email_queue: List[str] = []
        self._task_queue:  List[str] = []
        self._processed_emails: set  = set()

    def on_episode_start(self, obs: Observation):
        self._phase = "triage"
        self._processed_emails = set()
        # Sort: CRITICAL actionable first, then HIGH, then others, spam last
        def email_rank(e):
            if e.email_type == EmailType.SPAM: return 99
            if e.email_type == EmailType.ACTIONABLE:
                return {Priority.CRITICAL:0, Priority.HIGH:1,
                        Priority.MEDIUM:2, Priority.LOW:3}.get(e.priority, 4)
            return 10
        self._email_queue = [e.id for e in sorted(obs.emails, key=email_rank)]
        self._task_queue = []

    def act(self, obs: Observation) -> Action:
        # Energy guard
        if obs.energy_level < 0.28:
            return Action(action_type=ActionType.REST)

        email_map = {e.id: e for e in obs.emails}
        task_map  = {t.id: t for t in obs.tasks}

        # ── Triage phase: work through email queue
        for eid in list(self._email_queue):
            if eid not in email_map:
                self._email_queue.remove(eid)
                continue
            e = email_map[eid]
            if not e.is_read:
                return Action(action_type=ActionType.READ_EMAIL, email_id=eid)
            if e.email_type == EmailType.SPAM:
                self._email_queue.remove(eid)
                return Action(action_type=ActionType.DELETE_EMAIL, email_id=eid)
            if e.contains_task and e.email_type == EmailType.ACTIONABLE:
                already = any(t.source_email_id == eid for t in obs.tasks)
                if not already:
                    return Action(action_type=ActionType.EXTRACT_TASK, email_id=eid)
            self._email_queue.remove(eid)

        # ── Execute phase: complete tasks in priority+deadline order
        pending = [t for t in obs.tasks
                   if t.status not in (TaskStatus.COMPLETED, TaskStatus.DEFERRED, TaskStatus.OVERDUE)]
        pending_sorted = sorted(pending, key=lambda t: (-PW.get(t.priority, 1), t.deadline))

        for task in pending_sorted:
            if task.priority == Priority.LOW and obs.time_remaining < 10:
                return Action(action_type=ActionType.DEFER_TASK, task_id=task.id)
            if task.status == TaskStatus.PENDING:
                slot = obs.step + 1
                occupied = {s.time_step for s in obs.calendar}
                while slot in occupied:
                    slot += 1
                return Action(action_type=ActionType.SCHEDULE_TASK,
                              task_id=task.id, time_slot=slot)
            return Action(action_type=ActionType.COMPLETE_TASK, task_id=task.id)

        # ── Defer remaining low tasks
        for t in obs.tasks:
            if t.status == TaskStatus.PENDING and t.priority == Priority.LOW:
                return Action(action_type=ActionType.DEFER_TASK, task_id=t.id)

        return Action(action_type=ActionType.NOOP)


# ─────────────────────────────────────────────
# 4. Q-Learning Agent
# ─────────────────────────────────────────────
class QLearningAgent(BaseAgent):
    """
    Tabular Q-learning agent.
    State = discretised (unread_emails, pending_tasks, energy_bucket, time_bucket)
    Action = one of 8 macro-action types (parameterised greedily)
    """
    name = "qlearning"

    ACTION_TYPES = list(ActionType)   # 8 actions

    def __init__(self, lr: float = 0.15, gamma: float = 0.92,
                 epsilon: float = 1.0, epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.995, seed: int = 0):
        self.lr            = lr
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng           = random.Random(seed)
        self.q: Dict[Tuple, Dict[str, float]] = collections.defaultdict(
            lambda: {a.value: 0.0 for a in self.ACTION_TYPES}
        )
        self._prev_state: Optional[Tuple] = None
        self._prev_action: Optional[str]  = None
        self._greedy = GreedyAgent()   # fallback for parameterisation

    def _state(self, obs: Observation) -> Tuple:
        unread      = min(sum(1 for e in obs.emails if not e.is_read), 10)
        pending     = min(sum(1 for t in obs.tasks
                              if t.status not in (TaskStatus.COMPLETED,
                                                  TaskStatus.DEFERRED,
                                                  TaskStatus.OVERDUE)), 10)
        spam        = min(sum(1 for e in obs.emails if e.email_type == EmailType.SPAM), 5)
        energy_b    = int(obs.energy_level * 4)          # 0-4
        time_b      = int((obs.time_remaining / max(1, obs.time_remaining + obs.step)) * 4)
        critical    = min(sum(1 for t in obs.tasks
                              if t.priority == Priority.CRITICAL
                              and t.status == TaskStatus.PENDING), 3)
        return (unread, pending, spam, energy_b, time_b, critical)

    def _parameterise(self, action_type: ActionType, obs: Observation) -> Action:
        """Convert abstract action type into a concrete parameterised action."""
        email_map = {e.id: e for e in obs.emails}
        task_map  = {t.id: t for t in obs.tasks}

        if action_type == ActionType.READ_EMAIL:
            unread = [e for e in obs.emails if not e.is_read]
            if not unread: return Action(action_type=ActionType.NOOP)
            # Pick highest priority unread
            e = min(unread, key=lambda x: ({Priority.CRITICAL:0, Priority.HIGH:1,
                                             Priority.MEDIUM:2, Priority.LOW:3,
                                             None:4}.get(x.priority, 4)))
            return Action(action_type=ActionType.READ_EMAIL, email_id=e.id)

        elif action_type == ActionType.EXTRACT_TASK:
            extractable = [e for e in obs.emails
                           if e.is_read and e.contains_task
                           and e.email_type == EmailType.ACTIONABLE
                           and not any(t.source_email_id == e.id for t in obs.tasks)]
            if not extractable: return Action(action_type=ActionType.NOOP)
            e = min(extractable, key=lambda x: PW.get(x.priority, 1))
            return Action(action_type=ActionType.EXTRACT_TASK, email_id=e.id)

        elif action_type == ActionType.DELETE_EMAIL:
            spam = [e for e in obs.emails if e.email_type == EmailType.SPAM and e.is_read]
            if not spam: return Action(action_type=ActionType.NOOP)
            return Action(action_type=ActionType.DELETE_EMAIL, email_id=spam[0].id)

        elif action_type == ActionType.SCHEDULE_TASK:
            pending = [t for t in obs.tasks if t.status == TaskStatus.PENDING]
            if not pending: return Action(action_type=ActionType.NOOP)
            t = min(pending, key=lambda x: (-PW.get(x.priority, 1), x.deadline))
            occupied = {s.time_step for s in obs.calendar}
            slot = obs.step + 1
            while slot in occupied: slot += 1
            return Action(action_type=ActionType.SCHEDULE_TASK,
                          task_id=t.id, time_slot=slot)

        elif action_type == ActionType.COMPLETE_TASK:
            active = [t for t in obs.tasks
                      if t.status == TaskStatus.SCHEDULED
                      and t.status not in (TaskStatus.COMPLETED, TaskStatus.OVERDUE)]
            if not active:
                active = [t for t in obs.tasks
                          if t.status not in (TaskStatus.COMPLETED,
                                              TaskStatus.DEFERRED,
                                              TaskStatus.OVERDUE)]
            if not active: return Action(action_type=ActionType.NOOP)
            t = min(active, key=lambda x: (-PW.get(x.priority, 1), x.deadline))
            return Action(action_type=ActionType.COMPLETE_TASK, task_id=t.id)

        elif action_type == ActionType.DEFER_TASK:
            low = [t for t in obs.tasks
                   if t.priority == Priority.LOW
                   and t.status not in (TaskStatus.COMPLETED, TaskStatus.DEFERRED,
                                        TaskStatus.OVERDUE)]
            if not low: return Action(action_type=ActionType.NOOP)
            return Action(action_type=ActionType.DEFER_TASK, task_id=low[0].id)

        elif action_type == ActionType.REST:
            return Action(action_type=ActionType.REST)

        return Action(action_type=ActionType.NOOP)

    def act(self, obs: Observation) -> Action:
        state = self._state(obs)
        if self.rng.random() < self.epsilon:
            atype = self.rng.choice(self.ACTION_TYPES)
        else:
            q_vals = self.q[state]
            atype = ActionType(max(q_vals, key=q_vals.get))
        self._prev_state  = state
        self._prev_action = atype.value
        return self._parameterise(atype, obs)

    def on_step(self, obs: Observation, action: Action, reward: float, done: bool):
        if self._prev_state is None: return
        next_state  = self._state(obs)
        q_old       = self.q[self._prev_state][self._prev_action]
        future_best = max(self.q[next_state].values()) if not done else 0.0
        td_target   = reward + self.gamma * future_best
        self.q[self._prev_state][self._prev_action] += self.lr * (td_target - q_old)
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def on_episode_start(self, obs: Observation):
        self._prev_state  = None
        self._prev_action = None


# ─────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────
def run_agent(agent: BaseAgent, difficulty: str, seed: int,
              n_episodes: int = 1, train: bool = False) -> Dict[str, Any]:
    scores, rewards = [], []
    for ep in range(n_episodes):
        env = AgentOpsEnv(difficulty=difficulty, seed=seed + ep if train else seed)
        obs = env.reset()
        agent.on_episode_start(obs)
        total_r = 0.0; done = False
        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            agent.on_step(obs, action, reward.total, done)
            total_r += reward.total
        score = env.get_final_score()
        scores.append(score.overall if score else 0.0)
        rewards.append(total_r)

    return {
        "agent":      agent.name,
        "difficulty": difficulty,
        "episodes":   n_episodes,
        "score_mean": round(sum(scores)  / len(scores), 4),
        "score_max":  round(max(scores), 4),
        "score_min":  round(min(scores), 4),
        "reward_mean":round(sum(rewards) / len(rewards), 2),
        "scores":     [round(s, 4) for s in scores],
    }


def run_all_baselines(n_train: int = 150, n_eval: int = 10,
                      output: str = "baseline_results.json") -> Dict:
    print("\n" + "="*65)
    print("  AgentOpsEnv — Baseline Evaluation Suite")
    print("="*65)

    difficulties = ["easy", "medium", "hard"]
    agents_eval  = [RandomAgent(seed=0), GreedyAgent(), SmartAgent()]
    ql_agent     = QLearningAgent(seed=0)

    results = {}

    # ── Train Q-learning agent first
    print(f"\n[QLearning] Training for {n_train} episodes per difficulty…")
    for diff in difficulties:
        for ep in range(n_train):
            env = AgentOpsEnv(difficulty=diff, seed=ep % 50)
            obs = env.reset(); ql_agent.on_episode_start(obs)
            done = False
            while not done:
                a = ql_agent.act(obs)
                obs, r, done, _ = env.step(a)
                ql_agent.on_step(obs, a, r.total, done)
        print(f"  {diff}: trained. ε={ql_agent.epsilon:.3f}")

    # ── Evaluate all agents
    for agent in agents_eval + [ql_agent]:
        results[agent.name] = {}
        for diff in difficulties:
            seed = {"easy":100,"medium":200,"hard":300}[diff]
            r = run_agent(agent, diff, seed, n_episodes=n_eval)
            results[agent.name][diff] = r
            print(f"  [{agent.name:10}] {diff:6} → score={r['score_mean']:.4f}  "
                  f"max={r['score_max']:.4f}  reward={r['reward_mean']:+.1f}")

    # ── Summary table
    print("\n" + "="*65)
    print(f"  {'Agent':<12} {'Easy':>8} {'Medium':>8} {'Hard':>8} {'Avg':>8}")
    print("  " + "-"*52)
    for name, res in results.items():
        e = res["easy"]["score_mean"]
        m = res["medium"]["score_mean"]
        h = res["hard"]["score_mean"]
        avg = (e + m + h) / 3
        print(f"  {name:<12} {e:>8.4f} {m:>8.4f} {h:>8.4f} {avg:>8.4f}")
    print("="*65)

    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to: {output}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-episodes", type=int, default=150)
    parser.add_argument("--eval-episodes",  type=int, default=10)
    parser.add_argument("--output", default="baseline_results.json")
    args = parser.parse_args()
    run_all_baselines(args.train_episodes, args.eval_episodes, args.output)
