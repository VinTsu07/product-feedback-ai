from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    FAILED = "failed"

class AnalysisRequest(BaseModel):
    product_name: str
    sources: Optional[List[str]] = ["reddit", "youtube"]

class TaskResponse(BaseModel):
    task_id: str
    status: TaskStatus
    message: str
    created_at: datetime

# TODO: Add more models as needed
