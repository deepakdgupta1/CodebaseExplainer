from pathlib import Path
from typing import Dict, Any, List
import logging
from rich.progress import Progress

from codehierarchy.config.schema import Config
from codehierarchy.analysis.scanner.file_scanner import FileScanner
from codehierarchy.analysis.parser.parallel_parser import ParallelParser
from codehierarchy.analysis.graph.graph_builder import InMemoryGraphBuilder
from codehierarchy.core.llm.deepseek_summarizer import DeepSeekSummarizer
from codehierarchy.core.search.embedder import HighQualityEmbedder
from codehierarchy.core.search.keyword_search import KeywordSearch
from codehierarchy.utils.profiler import Profiler
from codehierarchy.config.loader import load_prompt_template
from codehierarchy.core.llm.checkpoint import save_checkpoint, load_checkpoint

class Orchestrator:
    def __init__(self, config: Config):
        self.config = config
        self.profiler = Profiler()
        self.output_dir = config.system.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def run_pipeline(self, repo_path: Path) -> Dict[str, Any]:
        """
        Run the full analysis pipeline.
        """
        logging.info(f"Starting analysis of {repo_path}")
        
        # Phase 1: Scan
        self.profiler.start_phase("scan")
        scanner = FileScanner(self.config.parsing)
        files = scanner.scan_directory(repo_path)
        self.profiler.end_phase("scan", {'files_found': len(files)})
        
        if not files:
            logging.error("No files found to process.")
            return {}
            
        # Phase 2: Parse
        self.profiler.start_phase("parse")
        parser = ParallelParser(self.config.parsing.num_workers)
        parse_results = parser.parse_repository(files)
        self.profiler.end_phase("parse", {'files_parsed': len(parse_results)})
        
        # Phase 3: Graph
        self.profiler.start_phase("graph")
        builder = InMemoryGraphBuilder()
        graph = builder.build_from_results(parse_results)
        self.profiler.end_phase("graph", {'nodes': graph.number_of_nodes(), 'edges': graph.number_of_edges()})
        
        # Phase 4: Summarize
        self.profiler.start_phase("summarize")
        prompt = load_prompt_template("deepseek-optimized")
        summarizer = DeepSeekSummarizer(self.config.llm, prompt)
        
        node_ids = list(graph.nodes())
        summaries = {}
        
        # Check for checkpoint
        checkpoint_file = self.output_dir / "checkpoints" / "summaries.json"
        if self.config.system.checkpointing_enabled and checkpoint_file.exists():
            summaries = load_checkpoint(checkpoint_file)
            logging.info(f"Loaded {len(summaries)} summaries from checkpoint")
            # Filter out nodes already summarized
            node_ids = [nid for nid in node_ids if nid not in summaries]
            
        batch_size = self.config.llm.batch_size
        total_nodes = len(node_ids)
        
        if total_nodes > 0:
            with Progress() as progress:
                task = progress.add_task("[cyan]Summarizing...", total=total_nodes)
                
                for i in range(0, total_nodes, batch_size):
                    batch = node_ids[i:i+batch_size]
                    batch_summaries = summarizer.summarize_batch(batch, builder)
                    summaries.update(batch_summaries)
                    progress.update(task, advance=len(batch))
                    
                    # Save checkpoint
                    if self.config.system.checkpointing_enabled and (i // batch_size) % self.config.system.checkpoint_interval == 0:
                        save_checkpoint(summaries, checkpoint_file)
                        
            # Final checkpoint
            if self.config.system.checkpointing_enabled:
                save_checkpoint(summaries, checkpoint_file)
                
        self.profiler.end_phase("summarize", {'summaries_generated': len(summaries)})
        
        # Phase 5: Index
        self.profiler.start_phase("index")
        index_dir = self.output_dir / "index"
        
        # Vector Index
        embedder = HighQualityEmbedder(self.config.embeddings.model_name)
        index, mapping = embedder.build_index(summaries)
        embedder.save_index(index, mapping, index_dir)
        
        # Keyword Index
        keyword_search = KeywordSearch(index_dir / "keyword.db")
        nodes_data = {nid: graph.nodes[nid] for nid in summaries if nid in graph.nodes}
        keyword_search.index_data(summaries, nodes_data)
        
        self.profiler.end_phase("index")
        
        # Save metrics
        self.profiler.save_metrics(self.output_dir / "performance-metrics.json")
        
        return {
            'graph': graph,
            'summaries': summaries,
            'builder': builder
        }
