"""
AgentOpsEnv — Typed Pydantic models (OpenEnv spec).
"""
from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import uuid


class EmailType(str, Enum):
    SPAM       = "spam"
    INFO       = "informational"
    ACTIONABLE = "actionable"
    AMBIGUOUS  = "ambiguous"

class Priority(str, Enum):
    CRITICAL = "critical"
    HIGH     = "high"
    MEDIUM   = "medium"
    LOW      = "low"

class TaskStatus(str, Enum):
    PENDING    = "pending"
    SCHEDULED  = "scheduled"
    COMPLETED  = "completed"
    DEFERRED   = "deferred"
    OVERDUE    = "overdue"

class ActionType(str, Enum):
    READ_EMAIL    = "read_email"
    EXTRACT_TASK  = "extract_task"
    DELETE_EMAIL  = "delete_email"
    SCHEDULE_TASK = "schedule_task"
    COMPLETE_TASK = "complete_task"
    DEFER_TASK    = "defer_task"
    REST          = "rest"
    NOOP          = "noop"


class Email(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    subject: str
    body: str
    sender: str
    priority: Priority
    deadline: Optional[int] = None
    email_type: EmailType
    is_read: bool = False
    contains_task: bool = False
    task_keywords: List[str] = Field(default_factory=list)
    noise_level: float = 0.0


class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str
    description: str
    priority: Priority
    deadline: int
    effort: int
    status: TaskStatus = TaskStatus.PENDING
    source_email_id: Optional[str] = None
    scheduled_slot: Optional[int] = None
    completed_at: Optional[int] = None
    progress: int = 0


class CalendarSlot(BaseModel):
    time_step: int
    task_id: str
    duration: int


class ActionHistoryEntry(BaseModel):
    step: int
    action_type: ActionType
    params: Dict[str, Any]
    reward_delta: float
    success: bool
    message: str


class Observation(BaseModel):
    step: int
    time_remaining: int
    energy_level: float
    emails: List[Email]
    tasks: List[Task]
    calendar: List[CalendarSlot]
    action_history: List[ActionHistoryEntry]
    overdue_tasks: List[str]
    completed_tasks: List[str]
    warnings: List[str]


class Action(BaseModel):
    action_type: ActionType
    email_id: Optional[str] = None
    task_id: Optional[str] = None
    time_slot: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)


class StepReward(BaseModel):
    total: float
    task_completion: float  = 0.0
    deadline_bonus: float   = 0.0
    progress_reward: float  = 0.0
    spam_cleanup: float     = 0.0
    deadline_penalty: float = 0.0
    redundancy_penalty: float = 0.0
    scheduling_penalty: float = 0.0
    energy_penalty: float   = 0.0
    idle_penalty: float     = 0.0
    details: str            = ""


class EpisodeScore(BaseModel):
    overall: float
    task_completion_score: float
    deadline_adherence_score: float
    efficiency_score: float
    resource_score: float
    breakdown: Dict[str, Any]
