"""
AgentOpsEnv — Baseline Inference Agent.

Uses an OpenAI-compatible API to run an LLM agent across all three tasks.
Produces reproducible scores and saves full trajectories.
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
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
API_KEY = os.environ.get("OPENAI_API_KEY", HF_TOKEN or "sk-placeholder")


SYSTEM_PROMPT = """
You are an AI agent acting as a highly efficient knowledge worker.
You manage a professional inbox: reading emails, extracting tasks,
scheduling work, and completing tasks before deadlines.
Respond ONLY in JSON format.
"""


def build_user_message(obs_dict: Dict[str, Any], task_desc: str) -> str:
    obs_trimmed = dict(obs_dict)
    obs_trimmed["action_history"] = obs_dict.get("action_history", [])[-5:]

    return (
        f"TASK: {task_desc}\n\n"
        f"CURRENT OBSERVATION:\n{json.dumps(obs_trimmed, indent=2)}\n\n"
        "Respond ONLY with JSON action."
    )


def parse_action(raw: str) -> Optional[Action]:
    try:
        clean = raw.strip()

        if clean.startswith("```"):
            clean = clean.split("```")[1]

            if clean.startswith("json"):
                clean = clean[4:]

        data = json.loads(clean.strip())

        return Action(**data)

    except Exception as e:
        print(f"[WARN] Failed parsing action: {e}", flush=True)
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

    conversation: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    total_reward = 0.0
    step = 0

    # REQUIRED START BLOCK
    print(
        f"[START] task={task_cfg['name']} "
        f"difficulty={task_cfg['difficulty']} "
        f"seed={task_cfg['seed']}",
        flush=True
    )

    while True:

        obs_dict = json.loads(obs.model_dump_json())

        user_msg = build_user_message(obs_dict, task_desc)

        conversation.append({
            "role": "user",
            "content": user_msg
        })

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=conversation,
                max_tokens=200,
                temperature=0.1,
            )

            raw_action = response.choices[0].message.content or ""

        except Exception as e:
            print(f"[ERROR] API failed: {e}", flush=True)

            raw_action = '{"action_type":"noop"}'

        conversation.append({
            "role": "assistant",
            "content": raw_action
        })

        action = parse_action(raw_action) or Action(
            action_type=ActionType.NOOP
        )

        obs, reward, done, info = env.step(action)

        total_reward += reward.total

        step += 1

        # REQUIRED STEP BLOCK
        print(
            f"[STEP] step={step} "
            f"action={action.action_type} "
            f"reward={reward.total:.2f}",
            flush=True
        )

        trajectory.append({
            "step": step,
            "action": action.to_dict(),
            "reward": reward.model_dump(),
            "success": info["success"],
            "message": info["message"],
        })

        if done:
            break

        time.sleep(0.1)

    score = env.get_final_score()

    # REQUIRED END BLOCK
    if score:
        print(
            f"[END] task={task_cfg['name']} "
            f"score={score.overall:.4f} "
            f"steps={step}",
            flush=True
        )

    return {
        "task_id": task_cfg["id"],
        "difficulty": task_cfg["difficulty"],
        "seed": task_cfg["seed"],
        "total_steps": step,
        "total_reward": round(total_reward, 4),
        "score": score.model_dump() if score else None,
        "trajectory": trajectory,
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        default="all",
        choices=[
            "all",
            "easy",
            "medium",
            "hard",
            "task_easy_inbox_triage",
            "task_medium_task_extraction",
            "task_hard_full_optimisation"
        ]
    )

    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument(
        "--output",
        default="results.json"
    )

    args = parser.parse_args()

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )

    if args.task == "all":
        selected_tasks = TASKS

    elif args.task in ("easy", "medium", "hard"):
        selected_tasks = [
            t for t in TASKS
            if t["difficulty"] == args.task
        ]

    else:
        selected_tasks = [
            get_task(args.task)
        ]

    all_results = []

    for task_cfg in selected_tasks:

        cfg = dict(task_cfg)

        if args.seed is not None:
            cfg["seed"] = args.seed

        result = run_task(cfg, client)

        all_results.append(result)

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    print(
        f"[END] all_tasks_completed total={len(all_results)}",
        flush=True
    )


if __name__ == "__main__":
    main()
