import networkx as nx
from typing import Dict, List
from pathlib import Path
import logging

class MarkdownGenerator:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_documentation(self, graph: nx.DiGraph, summaries: Dict[str, str]) -> None:
        """
        Generate markdown documentation from graph and summaries.
        """
        logging.info("Generating markdown documentation...")
        
        content = ["# Codebase Explanation\n"]
        
        # 1. Overview / Stats
        content.append("## Overview\n")
        content.append(f"- **Total Components**: {graph.number_of_nodes()}")
        content.append(f"- **Total Dependencies**: {graph.number_of_edges()}")
        content.append("\n")
        
        # 2. Component Details by File
        content.append("## Component Details\n")
        
        # Group by file
        files = sorted(list(set(data.get('file', 'unknown') for _, data in graph.nodes(data=True))))
        
        for file_path in files:
            content.append(f"### File: `{file_path}`\n")
            
            # Get nodes in this file
            file_nodes = [n for n, d in graph.nodes(data=True) if d.get('file') == file_path]
            # Sort by line number
            file_nodes.sort(key=lambda n: graph.nodes[n].get('line', 0))
            
            for nid in file_nodes:
                node = graph.nodes[nid]
                summary = summaries.get(nid, "No summary available.")
                
                content.append(f"#### {node.get('type', 'component')} `{node.get('name', 'unnamed')}`\n")
                content.append(f"- **Location**: Line {node.get('line', '?')}")
                # Add complexity if available
                # We need to access metadata which is not in graph node data directly in my implementation?
                # Wait, graph_builder puts metadata in self.metadata, not node attributes?
                # Let's check graph_builder.py
                # It adds attributes: type, name, file, line, end_line.
                # Metadata is separate.
                # So I can't access complexity here unless I pass metadata or put it in graph.
                # I should have put it in graph attributes too.
                # But for now, I'll skip complexity in MD or rely on summary.
                
                content.append(f"\n{summary}\n")
                
                content.append("---\n")
                
        output_file = self.output_dir / "CODEBASE_EXPLAINER.md"
        with open(output_file, 'w') as f:
            f.write("\n".join(content))
            
        logging.info(f"Documentation saved to {output_file}")
