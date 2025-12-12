from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any

Phase = Literal[
    "batch_start",
    "llm_call_start",
    "llm_call_success",
    "llm_call_error",
    "node_validated",
    "node_invalid",
    "batch_done",
    "disabled"
]

@dataclass
class SummarizationProgressEvent:
    phase: Phase
    batch_index: Optional[int] = None
    total_batches: Optional[int] = None
    batch_size: Optional[int] = None
    completed_nodes: Optional[int] = None
    total_nodes: Optional[int] = None
    node_id: Optional[str] = None
    message: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None
