from dataclasses import dataclass
from typing import Optional

@dataclass
class Result:
    node_id: str
    name: str
    file: str
    line: int
    summary: str
    score: float
    snippet: Optional[str] = None
    explanation: Optional[str] = None
