"""
AgentOpsEnv — FastAPI server.

Exposes the environment as a REST API for HuggingFace Spaces deployment.
Also serves the interactive UI at the root path.
"""
from __future__ import annotations

import json
import os
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from environment import AgentOpsEnv, Action, ActionType
from tasks import TASKS, list_tasks, get_task
from leaderboard import load_leaderboard, submit as lb_submit, init_with_baselines

# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
app = FastAPI(
    title="AgentOpsEnv",
    description="Real-World Workflow Optimisation Environment (OpenEnv spec)",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store (production would use Redis)
_sessions: Dict[str, AgentOpsEnv] = {}


# ─────────────────────────────────────────────
# Request/Response schemas
# ─────────────────────────────────────────────
class CreateSessionRequest(BaseModel):
    difficulty: str = "medium"
    seed: int = 42
    max_steps: Optional[int] = None

class StepRequest(BaseModel):
    session_id: str
    action: Dict[str, Any]

class ResetRequest(BaseModel):
    session_id: str


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    """Interactive HTML UI."""
    with open("ui.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.get("/health")
async def health():
    return {"status": "ok", "environment": "AgentOpsEnv", "version": "1.0.0"}


@app.get("/tasks")
async def get_tasks():
    """List all available tasks."""
    return {"tasks": list_tasks()}


@app.get("/tasks/{task_id}")
async def get_task_detail(task_id: str):
    try:
        return get_task(task_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")


@app.post("/session/create")
async def create_session(req: CreateSessionRequest):
    """Create a new environment session."""
    if req.difficulty not in ("easy", "medium", "hard"):
        raise HTTPException(status_code=400, detail="difficulty must be easy/medium/hard")
    session_id = str(uuid.uuid4())[:12]
    env = AgentOpsEnv(difficulty=req.difficulty, seed=req.seed, max_steps=req.max_steps)
    obs = env.reset()
    _sessions[session_id] = env
    return {
        "session_id": session_id,
        "observation": json.loads(obs.model_dump_json()),
        "config": {
            "difficulty": req.difficulty,
            "seed": req.seed,
            "max_steps": env.max_steps,
        }
    }


@app.post("/session/reset")
async def reset_session(req: ResetRequest):
    env = _get_session(req.session_id)
    obs = env.reset()
    return {
        "session_id": req.session_id,
        "observation": json.loads(obs.model_dump_json()),
    }


@app.post("/session/step")
async def step_session(req: StepRequest):
    env = _get_session(req.session_id)
    try:
        action = Action(**req.action)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid action: {e}")
    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "session_id": req.session_id,
        "observation": json.loads(obs.model_dump_json()),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/session/{session_id}/state")
async def get_state(session_id: str):
    env = _get_session(session_id)
    return env.state()


@app.get("/session/{session_id}/render")
async def render(session_id: str):
    env = _get_session(session_id)
    return {"render": env.render()}


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    if session_id in _sessions:
        del _sessions[session_id]
    return {"deleted": session_id}


@app.get("/openenv.yaml")
async def openenv_yaml():
    with open("openenv.yaml", "r") as f:
        return JSONResponse(content={"yaml": f.read()})


def _get_session(session_id: str) -> AgentOpsEnv:
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found. Create one first.")
    return _sessions[session_id]


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)

# ── Leaderboard endpoints
from leaderboard import load_leaderboard, submit as lb_submit, init_with_baselines

@app.on_event("startup")
async def _startup():
    init_with_baselines()

class SubmitRequest(BaseModel):
    agent_name: str
    author: str
    model: str
    easy: float
    medium: float
    hard: float
    notes: str = ""

@app.get("/leaderboard")
async def get_leaderboard():
    board = load_leaderboard()
    ranked = sorted(board, key=lambda x: -x.get("avg_score", 0))
    for i, e in enumerate(ranked): e["rank"] = i + 1
    return {"leaderboard": ranked, "total": len(ranked)}

@app.post("/leaderboard/submit")
async def submit_score(req: SubmitRequest):
    entry = lb_submit(req.agent_name, req.author, req.model,
                      req.easy, req.medium, req.hard, req.notes)
    return {"status": "submitted", "entry": entry}
