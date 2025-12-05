from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import logging

from .tree_sitter_parser import TreeSitterParser
from .node_extractor import NodeExtractor, NodeInfo
from .call_graph_analyzer import CallGraphAnalyzer, Edge
from codehierarchy.utils.language_detector import detect_language

@dataclass
class ParseResult:
    nodes: List[NodeInfo]
    edges: List[Edge]
    complexity: int = 0
    error: Optional[str] = None
    skipped: bool = False

def _parse_single_file(file_path: Path) -> ParseResult:
    """
    Worker function to parse a single file.
    Instantiates parser/extractor locally to avoid pickling issues.
    """
    try:
        lang = detect_language(file_path)
        if not lang:
            return ParseResult(nodes=[], edges=[], skipped=True)
            
        # Initialize components
        parser = TreeSitterParser(lang)
        extractor = NodeExtractor()
        analyzer = CallGraphAnalyzer(lang)
        
        # Read file
        try:
            content = file_path.read_bytes()
        except Exception as e:
            return ParseResult(nodes=[], edges=[], error=f"Read error: {e}")
            
        # Parse
        tree = parser.parse_bytes(content)
        
        # Extract nodes
        nodes = extractor.extract_all_nodes(tree, lang, content)
        
        # Analyze edges
        edges = analyzer.analyze(file_path, tree)
        
        # Compute total complexity
        total_complexity = sum(n.complexity for n in nodes)
        
        return ParseResult(
            nodes=nodes,
            edges=edges,
            complexity=total_complexity
        )
        
    except Exception as e:
        return ParseResult(nodes=[], edges=[], error=str(e))

class ParallelParser:
    def __init__(self, num_workers: int = 6):
        self.num_workers = num_workers
        
    def parse_repository(self, files: List[Path]) -> Dict[Path, ParseResult]:
        """
        Parse a list of files in parallel.
        Returns a dictionary mapping file path to ParseResult.
        """
        results = {}
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(_parse_single_file, f): f 
                for f in files
            }
            
            # Process results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    # 30s timeout per file
                    result = future.result(timeout=30)
                    results[file_path] = result
                except TimeoutError:
                    results[file_path] = ParseResult(nodes=[], edges=[], error="Timeout")
                except Exception as e:
                    results[file_path] = ParseResult(nodes=[], edges=[], error=f"Worker error: {e}")
                    
        return results
