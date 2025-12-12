"""
Summarization progress event types.

This module defines the data structures for tracking progress during
LLM summarization. Events are emitted at key stages and can be consumed
by progress bars, loggers, or other monitoring systems.

Phase Types:
    - batch_start: Beginning of a new batch
    - llm_call_start: LLM API call initiated
    - llm_call_success: LLM API call completed
    - llm_call_error: LLM API call failed
    - node_validated: Node summary passed validation
    - node_invalid: Node summary failed validation
    - batch_done: Batch processing completed
    - disabled: Summarizer has been disabled
"""

from dataclasses import dataclass, field
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
    """
    Event representing progress during summarization.

    Instances of this class are emitted via callbacks to track the
    state of batch summarization. Consumers can use these events
    for progress bars, logging, or telemetry.

    Attributes:
        phase: The current stage of processing.
        batch_index: Zero-based index of the current batch.
        total_batches: Total number of batches to process.
        batch_size: Number of nodes in the current batch.
        completed_nodes: Count of successfully processed nodes.
        total_nodes: Total nodes to process across all batches.
        node_id: ID of the specific node (for node-level events).
        message: Human-readable status message.
        extra: Additional metadata (e.g., timing, scores).
    """

    phase: Phase
    batch_index: Optional[int] = None
    total_batches: Optional[int] = None
    batch_size: Optional[int] = None
    completed_nodes: Optional[int] = None
    total_nodes: Optional[int] = None
    node_id: Optional[str] = None
    message: Optional[str] = None
    extra: Optional[Dict[str, Any]] = field(default_factory=dict)

