import logging
import time
import json
from typing import List, Dict, Any
from openai import OpenAI
from codehierarchy.config.schema import LLMConfig
from codehierarchy.analysis.graph.graph_builder import InMemoryGraphBuilder
from .validator import validate_summary
from .model_manager import ModelManager

class LMStudioSummarizer:
    def __init__(self, config: LLMConfig, prompt_template: str):
        self.config = config
        self.prompt_template = prompt_template
        self.client = OpenAI(base_url=config.base_url, api_key=config.api_key)
        self.model = config.model_name
        
        # Ensure model is loaded
        self.model_manager = ModelManager(config)
        self.model_manager.load_model()
        
    def summarize_batch(self, node_ids: List[str], builder: InMemoryGraphBuilder) -> Dict[str, str]:
        """
        Summarize a batch of nodes using LM Studio.
        Returns a dictionary mapping node_id to summary.
        """
        if not node_ids:
            return {}
            
        # 1. Build context for each node
        contexts = []
        valid_node_ids = []
        
        for nid in node_ids:
            ctx = builder.get_node_with_context(nid)
            if ctx:
                contexts.append(ctx)
                valid_node_ids.append(nid)
        
        if not valid_node_ids:
            return {}
            
        # 2. Construct prompt
        prompt = self._build_batch_prompt(valid_node_ids, contexts)
        
        # 3. Call LLM
        try:
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': self.prompt_template},
                    {'role': 'user', 'content': prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.context_window, 
                top_p=self.config.top_p,
                extra_body={
                    "top_k": self.config.top_k,
                    "repeat_penalty": self.config.repeat_penalty,
                    "min_p": self.config.min_p,
                    "context_overflow_policy": self.config.context_overflow_policy
                }
            )
            duration = time.time() - start_time
            logging.info(f"LLM call took {duration:.2f}s for {len(valid_node_ids)} nodes")
            
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
            validated_summaries = {}
            for nid, summary in summaries.items():
                is_valid, score = validate_summary(summary, nid.split(':')[-2])
                if is_valid:
                    validated_summaries[nid] = summary
                else:
                    logging.warning(f"Invalid summary for {nid}")
                    
            return validated_summaries
            
        except Exception as e:
            logging.error(f"LLM call failed: {e}", exc_info=True)
            return {}

    def _build_batch_prompt(self, node_ids: List[str], contexts: List[dict]) -> str:
        parts = [
            "Please analyze the following code components and provide clear, concise summaries.",
            "Follow the format: [COMPONENT_ID] <summary>",
            "\n---\n"
        ]
        
        for nid, ctx in zip(node_ids, contexts):
            node = ctx['node']
            source = ctx['source']
            
            parts.append(f"Component ID: [{nid}]")
            parts.append(f"Type: {node.get('type', 'unknown')}")
            parts.append(f"Name: {node.get('name', 'unknown')}")
            parts.append(f"File: {node.get('file', 'unknown')}")
            
            if source.get('docstring'):
                parts.append(f"Docstring: {source['docstring']}")
                
            parts.append(f"Source Code:\n{source.get('source_code', '')}")
            
            parents = [p for p in ctx['parents']]
            if parents:
                parts.append(f"Called by: {', '.join(parents[:5])}")
                
            children = [c for c in ctx['children']]
            if children:
                parts.append(f"Calls: {', '.join(children[:5])}")
                
            parts.append("\n---\n")
            
        return "\n".join(parts)

    def _parse_batch_response(self, response: str, node_ids: List[str]) -> Dict[str, str]:
        summaries = {}
        current_id = None
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
