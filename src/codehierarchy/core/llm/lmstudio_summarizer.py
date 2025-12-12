import time
import logging
from typing import List, Dict, Optional, Callable

from openai import OpenAI, APIConnectionError, BadRequestError

from codehierarchy.config.schema import LLMConfig
from codehierarchy.analysis.graph.graph_builder import InMemoryGraphBuilder
from .validator import validate_summary
from .backends import create_backend, BaseLLMBackend
from .progress import SummarizationProgressEvent

# A callback that receives progress events emitted during summarization
ProgressCallback = Callable[[SummarizationProgressEvent], None]


class LMStudioSummarizer:
    """
    Summarizer using configurable LLM backends.

    Despite the name (kept for backward compatibility), this class
    now supports both LM Studio and llama.cpp backends via the
    backends.create_backend() factory.
    """

    def __init__(self, config: LLMConfig, prompt_template: str):
        self.config = config
        self.prompt_template = prompt_template
        self.enabled = True  # Flag to track if LLM is available

        # Create and setup backend based on config
        self.backend: BaseLLMBackend = create_backend(config)
        try:
            self.backend.setup()
        except RuntimeError as e:
            logging.error(f"Backend setup failed: {e}")
            self.enabled = False
            self.model = config.model_name
            self.client = OpenAI(base_url=config.base_url, api_key=config.api_key)
            return

        # Load model
        loaded_id = self.backend.load_model()
        if loaded_id:
            self.model = loaded_id
            logging.info(f"Using model identifier: {self.model}")
        else:
            logging.warning("Model failed to load. Summarization disabled.")
            self.enabled = False
            self.model = config.model_name

        # Create OpenAI client using backend's base_url
        self.client = OpenAI(
            base_url=self.backend.base_url,
            api_key=config.api_key
        )

    def summarize_batch(
        self,
        node_ids: List[str],
        builder: InMemoryGraphBuilder,
        *,
        progress_cb: Optional[ProgressCallback] = None,
        batch_index: Optional[int] = None,
        total_batches: Optional[int] = None,
        completed_so_far: int = 0,
        total_nodes: Optional[int] = None,
    ) -> Dict[str, str]:
        """
        Summarize a batch of nodes using LM Studio.
        Returns a dictionary mapping node_id to summary.

        Progress visibility:
        - Emits SummarizationProgressEvent instances via progress_cb at key stages:
          - "batch_start"
          - "llm_call_start"
          - "llm_call_success"
          - "llm_call_error"
          - "node_validated"
          - "node_invalid"
          - "batch_done"
          - "disabled"
        """

        # Summarization globally disabled (model not loaded / prior failure)
        if not self.enabled:
            if progress_cb:
                progress_cb(
                    SummarizationProgressEvent(
                        phase="disabled",
                        batch_index=batch_index,
                        total_batches=total_batches,
                        completed_nodes=completed_so_far,
                        total_nodes=total_nodes,
                        message=(
                            "LLM summarization disabled "
                            "(model not loaded or previous failure)."
                        ),
                    )
                )
            return {}

        if not node_ids:
            # Nothing to do; no need to emit an event here
            return {}

        # Emit batch_start as early as possible so the orchestrator/CLI knows
        # that this batch has begun, even before we build LLM context.
        if progress_cb:
            msg_batch_idx = (
                f" {batch_index}" if batch_index is not None else ""
            )
            progress_cb(
                SummarizationProgressEvent(
                    phase="batch_start",
                    batch_index=batch_index,
                    total_batches=total_batches,
                    batch_size=len(node_ids),
                    completed_nodes=completed_so_far,
                    total_nodes=total_nodes,
                    message=f"Starting batch{msg_batch_idx} with {len(node_ids)} node(s).",
                )
            )

        # 1. Build context for each node
        contexts = []
        valid_node_ids: List[str] = []

        for nid in node_ids:
            ctx = builder.get_node_with_context(nid)
            if ctx:
                contexts.append(ctx)
                valid_node_ids.append(nid)

        if not valid_node_ids:
            # No valid contexts; effectively an empty batch
            if progress_cb:
                progress_cb(
                    SummarizationProgressEvent(
                        phase="batch_done",
                        batch_index=batch_index,
                        total_batches=total_batches,
                        batch_size=0,
                        completed_nodes=completed_so_far,
                        total_nodes=total_nodes,
                        message="No valid nodes to summarize in this batch.",
                    )
                )
            return {}

        prompt = self._build_batch_prompt(valid_node_ids, contexts)

        try:
            logging.info(
                f"Summarizing batch of {len(node_ids)} nodes via LM Studio."
            )

            # Notify that the LLM call is about to start
            if progress_cb:
                progress_cb(
                    SummarizationProgressEvent(
                        phase="llm_call_start",
                        batch_index=batch_index,
                        total_batches=total_batches,
                        batch_size=len(valid_node_ids),
                        completed_nodes=completed_so_far,
                        total_nodes=total_nodes,
                        message="Sending summarization batch to LLM.",
                    )
                )

            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.prompt_template},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.context_window,
                top_p=self.config.top_p,
                extra_body={
                    "top_k": self.config.top_k,
                    "repeat_penalty": self.config.repeat_penalty,
                    "min_p": self.config.min_p,
                    "context_overflow_policy": self.config.context_overflow_policy,
                },
            )
            elapsed = time.time() - start_time
            logging.debug(f"LLM call took {elapsed:.2f}s")

            # Signal success of the LLM call (even if we still need to validate)
            if progress_cb:
                progress_cb(
                    SummarizationProgressEvent(
                        phase="llm_call_success",
                        batch_index=batch_index,
                        total_batches=total_batches,
                        batch_size=len(valid_node_ids),
                        completed_nodes=completed_so_far,
                        total_nodes=total_nodes,
                        extra={"elapsed": elapsed},
                        message=f"LLM call succeeded in {elapsed:.2f}s.",
                    )
                )

            if not response.choices:
                logging.warning("LLM returned no choices.")
                return {}

            content = response.choices[0].message.content
            if not content:
                logging.warning("LLM returned empty content.")
                return {}

            # 4. Parse response
            summaries = self._parse_batch_response(content, valid_node_ids)

            # 5. Validate summaries
            validated_summaries: Dict[str, str] = {}
            locally_completed = completed_so_far

            for nid, summary in summaries.items():
                is_valid, score = validate_summary(summary, nid.split(":")[-2])
                if is_valid:
                    validated_summaries[nid] = summary
                    locally_completed += 1

                    if progress_cb:
                        progress_cb(
                            SummarizationProgressEvent(
                                phase="node_validated",
                                batch_index=batch_index,
                                total_batches=total_batches,
                                node_id=nid,
                                completed_nodes=locally_completed,
                                total_nodes=total_nodes,
                                extra={"score": score},
                                message=f"Validated summary for {nid}.",
                            )
                        )
                else:
                    logging.warning(f"Invalid summary for {nid}")
                    if progress_cb:
                        progress_cb(
                            SummarizationProgressEvent(
                                phase="node_invalid",
                                batch_index=batch_index,
                                total_batches=total_batches,
                                node_id=nid,
                                completed_nodes=locally_completed,
                                total_nodes=total_nodes,
                                extra={"score": score},
                                message=f"Invalid summary for {nid}.",
                            )
                        )

            # Notify batch completion
            if progress_cb:
                progress_cb(
                    SummarizationProgressEvent(
                        phase="batch_done",
                        batch_index=batch_index,
                        total_batches=total_batches,
                        batch_size=len(valid_node_ids),
                        completed_nodes=locally_completed,
                        total_nodes=total_nodes,
                        message=f"Finished batch {batch_index}.",
                    )
                )

            return validated_summaries

        except APIConnectionError:
            logging.warning(
                f"Could not connect to LM Studio at {self.config.base_url}. "
                "Is the server running? "
                "Disabling LLM summarization for the remainder of this run."
            )
            self.enabled = False
            if progress_cb:
                progress_cb(
                    SummarizationProgressEvent(
                        phase="llm_call_error",
                        batch_index=batch_index,
                        total_batches=total_batches,
                        completed_nodes=completed_so_far,
                        total_nodes=total_nodes,
                        message=(
                            "APIConnectionError: LM Studio unreachable. "
                            "Disabling summarizer."
                        ),
                    )
                )
                progress_cb(
                    SummarizationProgressEvent(
                        phase="disabled",
                        batch_index=batch_index,
                        total_batches=total_batches,
                        completed_nodes=completed_so_far,
                        total_nodes=total_nodes,
                        message="Summarizer disabled due to connection failure.",
                    )
                )
            return {}

        except BadRequestError as e:
            logging.error(f"LLM Request failed: {e}")
            logging.warning(
                "This usually means the model is not loaded or "
                "the request context is too long."
            )
            if progress_cb:
                progress_cb(
                    SummarizationProgressEvent(
                        phase="llm_call_error",
                        batch_index=batch_index,
                        total_batches=total_batches,
                        completed_nodes=completed_so_far,
                        total_nodes=total_nodes,
                        message=f"BadRequestError: {e}",
                    )
                )
            return {}

        except Exception as e:
            logging.error(f"LLM call failed: {e}", exc_info=True)
            if progress_cb:
                progress_cb(
                    SummarizationProgressEvent(
                        phase="llm_call_error",
                        batch_index=batch_index,
                        total_batches=total_batches,
                        completed_nodes=completed_so_far,
                        total_nodes=total_nodes,
                        message=f"Unexpected error: {e}",
                    )
                )
            return {}

    def _build_batch_prompt(
        self,
        node_ids: List[str],
        contexts: List[dict],
    ) -> str:
        parts = [
            "Please analyze the following code components and provide clear, concise summaries.",
            "Follow the format: [COMPONENT_ID] <summary>",
            "\n---\n",
        ]

        for nid, ctx in zip(node_ids, contexts):
            node = ctx["node"]
            source = ctx["source"]

            parts.append(f"Component ID: [{nid}]")
            parts.append(f"Type: {node.get('type', 'unknown')}")
            parts.append(f"Name: {node.get('name', 'unknown')}")
            parts.append(f"File: {node.get('file', 'unknown')}")

            if source.get("docstring"):
                parts.append(f"Docstring: {source['docstring']}")

            parts.append(f"Source Code:\n{source.get('source_code', '')}")

            parents = [p for p in ctx["parents"]]
            if parents:
                parts.append(f"Called by: {', '.join(parents[:5])}")

            children = [c for c in ctx["children"]]
            if children:
                parts.append(f"Calls: {', '.join(children[:5])}")

            parts.append("\n---\n")

        return "\n".join(parts)

    def _parse_batch_response(
        self,
        response: str,
        node_ids: List[str],
    ) -> Dict[str, str]:
        summaries: Dict[str, str] = {}
        current_id: Optional[str] = None
        current_lines: List[str] = []

        for line in response.splitlines():
            line = line.strip()
            if not line:
                continue

            # Check for ID marker
            found_id = None
            for nid in node_ids:
                if f"[{nid}]" in line:
                    found_id = nid
                    break

            if found_id:
                # Save previous
                if current_id and current_lines:
                    summaries[current_id] = " ".join(current_lines).strip()

                current_id = found_id
                current_lines = []
                # Remove the marker from the line
                content = line.replace(f"[{found_id}]", "").strip()
                if content:
                    current_lines.append(content)
            elif current_id:
                current_lines.append(line)

        # Save last
        if current_id and current_lines:
            summaries[current_id] = " ".join(current_lines).strip()

        return summaries
