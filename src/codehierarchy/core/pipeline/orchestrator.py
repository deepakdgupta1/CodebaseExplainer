"""
Pipeline orchestrator for codebase analysis.

This module provides the Orchestrator class which coordinates the full
analysis pipeline from file scanning through LLM summarization and
search indexing. It manages the execution of all pipeline phases:

1. **Scan**: Discover source files in the repository
2. **Parse**: Extract code structure (classes, functions, etc.)
3. **Graph**: Build an in-memory code graph
4. **Summarize**: Generate LLM summaries for code elements
5. **Index**: Build semantic and keyword search indices

The orchestrator also handles checkpointing for resumable runs and
performance profiling for each phase.
"""

from codehierarchy.core.llm.progress import SummarizationProgressEvent
from typing import Any, Dict
from pathlib import Path
import logging
from rich.progress import Progress

from codehierarchy.config.schema import Config
from codehierarchy.analysis.scanner.file_scanner import FileScanner
from codehierarchy.analysis.parser.parallel_parser import ParallelParser
from codehierarchy.analysis.graph.graph_builder import InMemoryGraphBuilder
from codehierarchy.core.llm.lmstudio_summarizer import LMStudioSummarizer
from codehierarchy.core.search.embedder import HighQualityEmbedder
from codehierarchy.core.search.keyword_search import KeywordSearch
from codehierarchy.utils.profiler import Profiler
from codehierarchy.config.loader import load_prompt_template
from codehierarchy.core.llm.checkpoint import save_checkpoint, load_checkpoint


class Orchestrator:
    """
    Coordinates the full codebase analysis pipeline.

    The orchestrator is the main entry point for running analysis.
    It instantiates and coordinates all pipeline components, handles
    progress reporting, and manages checkpointing.

    Attributes:
        config: Application configuration.
        profiler: Performance profiler for timing phases.
        output_dir: Directory for output files and indices.

    Example:
        >>> from codehierarchy.config.loader import load_config
        >>> config = load_config()
        >>> orchestrator = Orchestrator(config)
        >>> results = orchestrator.run_pipeline(Path("/path/to/repo"))
    """

    def __init__(self, config: Config) -> None:
        """
        Initialize the orchestrator.

        Args:
            config: Application configuration containing system,
                   parsing, LLM, and embedding settings.
        """
        self.config = config
        self.profiler = Profiler()
        self.output_dir = config.system.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_pipeline(self, repo_path: Path) -> Dict[str, Any]:
        """
        Run the full analysis pipeline on a repository.

        Executes all pipeline phases in sequence: scan, parse, graph,
        summarize, and index. Supports checkpointing for resumable runs.

        Args:
            repo_path: Path to the repository to analyze.

        Returns:
            Dictionary containing:
            - graph: The built code graph (NetworkX DiGraph)
            - summaries: Dict mapping node IDs to summary text
            - builder: The graph builder instance

        Note:
            Returns empty dict if no files are found to process.
        """
        logging.info(f"Starting analysis of {repo_path}")

        # Phase 1: Scan
        logging.info(
            f"Starting phase: Scan | Input: {repo_path}"
        )
        self.profiler.start_phase("scan")
        scanner = FileScanner(self.config.parsing)
        files = scanner.scan_directory(repo_path)
        self.profiler.end_phase("scan", {'files_found': len(files)})
        logging.info(f"Completed phase: Scan | Output: {len(files)} files found")

        if not files:
            logging.error("No files found to process.")
            return {}

        # Phase 2: Parse
        logging.info(f"Starting phase: Parse | Input: {len(files)} files")
        self.profiler.start_phase("parse")
        parser = ParallelParser(self.config.parsing.num_workers)
        parse_results = parser.parse_repository(files)
        self.profiler.end_phase("parse", {'files_parsed': len(parse_results)})
        logging.info(
            f"Completed phase: Parse | "
            f"Output: {len(parse_results)} successful parses"
        )

        # Phase 3: Graph
        logging.info(
            f"Starting phase: Graph | "
            f"Input: {len(parse_results)} parse results"
        )
        self.profiler.start_phase("graph")
        builder = InMemoryGraphBuilder()
        graph = builder.build_from_results(parse_results)
        node_count = graph.number_of_nodes()
        edge_count = graph.number_of_edges()
        self.profiler.end_phase("graph",
                                {'nodes': node_count,
                                 'edges': edge_count})
        logging.info(
            f"Completed phase: Graph | "
            f"Output: {node_count} nodes, {edge_count} edges"
        )

        # Phase 4: Summarize
        logging.info("Starting phase: Summarize | Input: Graph nodes")
        self.profiler.start_phase("summarize")
        prompt = load_prompt_template("deepseek-optimized")
        summarizer = LMStudioSummarizer(self.config.llm, prompt)

        node_ids = list(graph.nodes())
        summaries = {}

        checkpoint_file = self.output_dir / "checkpoints" / "summaries.json"
        if self.config.system.checkpointing_enabled and checkpoint_file.exists():
            summaries = load_checkpoint(checkpoint_file)
            node_ids = [nid for nid in node_ids if nid not in summaries]

        batch_size = self.config.llm.batch_size
        total_nodes = len(node_ids)
        total_batches = (total_nodes + batch_size - 1) // batch_size

        if total_nodes > 0:
            from time import time
            last_event = time()

        def on_progress(event: SummarizationProgressEvent):
            nonlocal last_event
            last_event = time()
            if event.phase == "node_validated":
                progress.update(task, advance=1)
            elif event.phase == "llm_call_start":
                progress.console.print(f"[yellow]Batch {event.batch_index+1}/{total_batches}: LLM started")
            elif event.phase == "llm_call_success":
                progress.console.print(f"[green]Batch {event.batch_index+1} done ({event.extra.get('elapsed', 0):.1f}s)")
            elif event.phase == "llm_call_error":
                progress.console.print(f"[red]LLM error: {event.message}")
            elif event.phase == "disabled":
                progress.console.print("[bold red]Summarizer disabled")

        with Progress() as progress:
            task = progress.add_task("[cyan]Summarizing...", total=total_nodes)

            for i in range(0, total_nodes, batch_size):
                batch = node_ids[i:i + batch_size]
                batch_summaries = summarizer.summarize_batch(
                    batch,
                    builder,
                    progress_cb=on_progress,
                    batch_index=i // batch_size,
                    total_batches=total_batches,
                    completed_so_far=len(summaries),
                    total_nodes=total_nodes,
                )
                summaries.update(batch_summaries)

                # Save checkpoint
                should_checkpoint = (
                    (i // batch_size) %
                    self.config.system.checkpoint_interval == 0
                )
                if self.config.system.checkpointing_enabled and should_checkpoint:
                    save_checkpoint(summaries, checkpoint_file)

            if self.config.system.checkpointing_enabled:
                save_checkpoint(summaries, checkpoint_file)
        self.profiler.end_phase("summarize",
                                {'summaries_generated': len(summaries)})
        logging.info(
            f"Completed phase: Summarize | "
            f"Output: {len(summaries)} summaries generated"
        )

        # Phase 5: Index
        logging.info(
            f"Starting phase: Index | Input: {len(summaries)} summaries"
        )
        self.profiler.start_phase("index")
        index_dir = self.output_dir / "index"

        # Vector Index
        embedder = HighQualityEmbedder(self.config.embeddings.model_name)
        index, mapping = embedder.build_index(summaries)
        embedder.save_index(index, mapping, index_dir)

        # Keyword Index
        keyword_search = KeywordSearch(index_dir / "keyword.db")
        nodes_data = {nid: graph.nodes[nid]
                      for nid in summaries if nid in graph.nodes}
        keyword_search.index_data(summaries, nodes_data)

        self.profiler.end_phase("index")
        logging.info(
            f"Completed phase: Index | Output: Index saved to {index_dir}"
        )

        # Save metrics
        self.profiler.save_metrics(
            self.output_dir /
            "performance-metrics.json")

        return {
            'graph': graph,
            'summaries': summaries,
            'builder': builder
        }
