"""
AgentOpsEnv — Leaderboard tracker.

Tracks agent scores across all tasks, supports submission and display.
Saved to leaderboard.json — can be served via the REST API.
"""
from __future__ import annotations
import json, os, time
from typing import Any, Dict, List, Optional
from datetime import datetime

LEADERBOARD_FILE = "leaderboard.json"

KNOWN_BASELINES = [
    {
        "rank": None,
        "agent_name": "Random Agent",
        "author": "AgentOpsEnv Team",
        "model": "random",
        "scores": {"easy": 0.6285, "medium": 0.3767, "hard": 0.4535},
        "avg_score": 0.4862,
        "submitted_at": "2025-01-01T00:00:00",
        "notes": "Uniform random action selection baseline"
    },
    {
        "rank": None,
        "agent_name": "Greedy Agent",
        "author": "AgentOpsEnv Team",
        "model": "greedy-rules",
        "scores": {"easy": 0.8277, "medium": 0.5021, "hard": 0.3331},
        "avg_score": 0.5543,
        "submitted_at": "2025-01-01T00:00:00",
        "notes": "Rule-based greedy: always acts on most urgent visible item"
    },
    {
        "rank": None,
        "agent_name": "Q-Learning Agent",
        "author": "AgentOpsEnv Team",
        "model": "tabular-qlearning-150ep",
        "scores": {"easy": 0.8500, "medium": 0.4828, "hard": 0.4566},
        "avg_score": 0.5965,
        "submitted_at": "2025-01-01T00:00:00",
        "notes": "Tabular Q-learning, 150 training episodes per difficulty"
    },
    {
        "rank": None,
        "agent_name": "Smart Interleaved Agent",
        "author": "AgentOpsEnv Team",
        "model": "smart-handcrafted",
        "scores": {"easy": 0.4854, "medium": 0.6928, "hard": 0.6824},
        "avg_score": 0.6202,
        "submitted_at": "2025-01-01T00:00:00",
        "notes": "Near-optimal hand-crafted planner — interleaved priority-aware execution"
    },
]


def load_leaderboard() -> List[Dict]:
    if not os.path.exists(LEADERBOARD_FILE):
        return list(KNOWN_BASELINES)
    with open(LEADERBOARD_FILE) as f:
        return json.load(f)


def save_leaderboard(board: List[Dict]):
    ranked = sorted(board, key=lambda x: -x.get("avg_score", 0))
    for i, entry in enumerate(ranked):
        entry["rank"] = i + 1
    with open(LEADERBOARD_FILE, "w") as f:
        json.dump(ranked, f, indent=2)
    return ranked


def submit(agent_name: str, author: str, model: str,
           easy: float, medium: float, hard: float,
           notes: str = "") -> Dict:
    board = load_leaderboard()
    avg = round((easy + medium + hard) / 3, 4)
    entry = {
        "rank": None,
        "agent_name": agent_name,
        "author": author,
        "model": model,
        "scores": {"easy": round(easy,4), "medium": round(medium,4), "hard": round(hard,4)},
        "avg_score": avg,
        "submitted_at": datetime.utcnow().isoformat(),
        "notes": notes,
    }
    board.append(entry)
    ranked = save_leaderboard(board)
    rank = next(e["rank"] for e in ranked if e["agent_name"] == agent_name and e["model"] == model)
    print(f"\n✅ Submitted! {agent_name} ranked #{rank} with avg score {avg:.4f}")
    return entry


def display(top_n: int = 20):
    board = load_leaderboard()
    ranked = sorted(board, key=lambda x: -x.get("avg_score", 0))
    for i, e in enumerate(ranked[:top_n]):
        e["rank"] = i + 1

    print("\n" + "="*80)
    print("  🏆  AgentOpsEnv Leaderboard")
    print("="*80)
    print(f"  {'#':<4} {'Agent':<25} {'Model':<22} {'Easy':>6} {'Med':>6} {'Hard':>6} {'Avg':>6}")
    print("  " + "-"*74)
    for e in ranked[:top_n]:
        s = e["scores"]
        print(f"  {e['rank']:<4} {e['agent_name'][:24]:<25} {e['model'][:21]:<22} "
              f"{s['easy']:>6.4f} {s['medium']:>6.4f} {s['hard']:>6.4f} "
              f"{e['avg_score']:>6.4f}")
    print("="*80)
    print(f"  Total entries: {len(ranked)}")


def init_with_baselines():
    """Initialise leaderboard with known baselines if file doesn't exist."""
    if not os.path.exists(LEADERBOARD_FILE):
        save_leaderboard(list(KNOWN_BASELINES))
        print(f"Initialised leaderboard with {len(KNOWN_BASELINES)} baselines.")


if __name__ == "__main__":
    init_with_baselines()
    display()
