"""
AgentOpsEnv — Baseline Inference Agent.

Uses an OpenAI-compatible API to run an LLM agent across all three tasks.
Produces reproducible scores and saves full trajectories.

Usage:
    export API_BASE_URL="https://api.openai.com/v1"
    export MODEL_NAME="gpt-4o"
    export HF_TOKEN="hf_..."   # optional, for HF Inference Endpoints
    python inference.py [--task all|easy|medium|hard] [--seed 42] [--output results.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai")
    sys.exit(1)

from environment import AgentOpsEnv, Action, ActionType
from tasks import TASKS, get_task

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
API_KEY      = os.environ.get("OPENAI_API_KEY", HF_TOKEN or "sk-placeholder")

SYSTEM_PROMPT = """You are an AI agent acting as a highly efficient knowledge worker.
You manage a professional inbox: reading emails, extracting tasks, scheduling work,
and completing tasks before deadlines — all while managing your limited energy.

Your goal is to MAXIMISE LONG-TERM PRODUCTIVITY, not just immediate rewards.

== OBSERVATION FORMAT ==
You receive a JSON observation with:
  - step / time_remaining / energy_level (0-1)
  - emails: list of {id, subject, body, sender, priority, email_type, is_read, deadline}
  - tasks: list of {id, title, priority, deadline, effort, status, progress}
  - calendar: scheduled tasks
  - warnings: urgent alerts
  - action_history: last actions taken

== ACTION FORMAT ==
Respond with ONLY a JSON object (no markdown, no explanation):
{
  "action_type": "<action>",
  "email_id": "<id>",     // if applicable
  "task_id": "<id>",      // if applicable
  "time_slot": <int>      // if applicable (for schedule_task)
}

Valid action_types:
  read_email     → params: email_id
  extract_task   → params: email_id  (must have read the email first)
  delete_email   → params: email_id
  schedule_task  → params: task_id, time_slot (integer step number)
  complete_task  → params: task_id  (call multiple times for effort > 1)
  defer_task     → params: task_id
  rest           → no params (recover energy)
  noop           → no params (avoid this)

== STRATEGY GUIDE ==
1. TRIAGE first: read all emails, delete spam, identify actionable emails.
2. EXTRACT tasks from actionable emails immediately after reading.
3. SCHEDULE tasks — CRITICAL first, then HIGH, MEDIUM, LOW.
4. COMPLETE tasks — call complete_task once per effort point.
5. MANAGE ENERGY — rest when energy < 0.30; working at <0.20 yields no progress.
6. DEFER low-priority tasks if time is running out.
7. NEVER repeat the same action on the same target — it incurs a penalty.

== PRIORITY WEIGHTS ==
  CRITICAL: 4.0×   HIGH: 2.5×   MEDIUM: 1.5×   LOW: 0.5×
"""


def build_user_message(obs_dict: Dict[str, Any], task_desc: str) -> str:
    # Trim action history for brevity
    obs_trimmed = dict(obs_dict)
    obs_trimmed["action_history"] = obs_dict.get("action_history", [])[-5:]
    return (
        f"TASK: {task_desc}\n\n"
        f"CURRENT OBSERVATION:\n{json.dumps(obs_trimmed, indent=2)}\n\n"
        "What is your next action? Respond with ONLY a JSON object."
    )


def parse_action(raw: str) -> Optional[Action]:
    try:
        # Strip markdown fences if present
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        data = json.loads(clean.strip())
        return Action(**data)
    except Exception as e:
        print(f"  [WARN] Failed to parse action: {e} | raw: {raw[:80]}")
        return None


def run_task(
    task_cfg: Dict[str, Any],
    client: OpenAI,
    verbose: bool = True,
) -> Dict[str, Any]:
    env = AgentOpsEnv(
        difficulty=task_cfg["difficulty"],
        seed=task_cfg["seed"],
        max_steps=task_cfg["max_steps"],
    )
    obs = env.reset()
    task_desc = task_cfg["description"][:300]

    trajectory: List[Dict[str, Any]] = []
    conversation: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    total_reward = 0.0
    step = 0

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Task: {task_cfg['name']} [{task_cfg['difficulty'].upper()}]")
        print(f"  Seed: {task_cfg['seed']} | Max Steps: {task_cfg['max_steps']}")
        print(f"{'='*60}")

    while True:
        obs_dict = json.loads(obs.model_dump_json())
        user_msg = build_user_message(obs_dict, task_desc)
        conversation.append({"role": "user", "content": user_msg})

        # Call LLM
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=conversation,
                max_tokens=200,
                temperature=0.1,
            )
            raw_action = response.choices[0].message.content or ""
        except Exception as e:
            print(f"  [ERROR] API call failed: {e}")
            raw_action = '{"action_type": "noop"}'

        conversation.append({"role": "assistant", "content": raw_action})

        action = parse_action(raw_action) or Action(action_type=ActionType.NOOP)

        obs, reward, done, info = env.step(action)
        total_reward += reward.total
        step += 1

        if verbose:
            print(f"  Step {step:02d} | {action.action_type:<15} | "
                  f"r={reward.total:+.2f} | energy={obs.energy_level:.2f} | "
                  f"{info['message'][:45]}")

        trajectory.append({
            "step": step,
            "action": action.to_dict(),
            "reward": reward.model_dump(),
            "success": info["success"],
            "message": info["message"],
        })

        if done:
            break

        time.sleep(0.1)  # rate limit courtesy

    score = env.get_final_score()
    if verbose and score:
        print(f"\n  ── FINAL SCORE ──────────────────────────────")
        print(f"  Overall         : {score.overall:.4f}")
        print(f"  Task Completion : {score.task_completion_score:.4f}")
        print(f"  Deadline Adhere : {score.deadline_adherence_score:.4f}")
        print(f"  Efficiency      : {score.efficiency_score:.4f}")
        print(f"  Resource Usage  : {score.resource_score:.4f}")
        print(f"  Total Reward    : {total_reward:+.2f}")
        print(f"  Breakdown       : {json.dumps(score.breakdown, indent=4)}")

    return {
        "task_id":      task_cfg["id"],
        "difficulty":   task_cfg["difficulty"],
        "seed":         task_cfg["seed"],
        "total_steps":  step,
        "total_reward": round(total_reward, 4),
        "score":        score.model_dump() if score else None,
        "trajectory":   trajectory,
    }


def main():
    parser = argparse.ArgumentParser(description="AgentOpsEnv Baseline Inference")
    parser.add_argument("--task", default="all",
                        choices=["all", "easy", "medium", "hard",
                                 "task_easy_inbox_triage",
                                 "task_medium_task_extraction",
                                 "task_hard_full_optimisation"])
    parser.add_argument("--seed", type=int, default=None,
                        help="Override task seed for reproducibility")
    parser.add_argument("--output", default="results.json",
                        help="Output file for results")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-step output")
    args = parser.parse_args()

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )

    # Select tasks
    if args.task == "all":
        selected_tasks = TASKS
    elif args.task in ("easy", "medium", "hard"):
        selected_tasks = [t for t in TASKS if t["difficulty"] == args.task]
    else:
        selected_tasks = [get_task(args.task)]

    print(f"\nAgentOpsEnv Baseline Inference")
    print(f"  Model       : {MODEL_NAME}")
    print(f"  API Base    : {API_BASE_URL}")
    print(f"  Tasks       : {[t['id'] for t in selected_tasks]}")

    all_results = []
    for task_cfg in selected_tasks:
        cfg = dict(task_cfg)
        if args.seed is not None:
            cfg["seed"] = args.seed
        result = run_task(cfg, client, verbose=not args.quiet)
        all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        sc = r["score"]["overall"] if r["score"] else 0.0
        print(f"  {r['task_id']:<40} score={sc:.4f}  reward={r['total_reward']:+.2f}")

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
