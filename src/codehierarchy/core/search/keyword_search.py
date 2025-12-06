import sqlite3
from pathlib import Path
from typing import List, Dict, Any
from .result import Result

class KeywordSearch:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            # Create FTS5 table
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS search_index 
                USING fts5(node_id, name, file, content, tokenize='porter')
            """)

    def index_data(self, summaries: Dict[str, str], nodes: Dict[str, Any]) -> None:
        """
        Index nodes and summaries into SQLite FTS5.
        """
        data = []
        for nid, summary in summaries.items():
            node = nodes.get(nid, {})
            name = node.get('name', '')
            file = node.get('file', '')
            # Combine name and summary for content
            content = f"{name} {summary}"
            data.append((nid, name, file, content))
            
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                "INSERT INTO search_index(node_id, name, file, content) VALUES (?, ?, ?, ?)",
                data
            )

    def search(self, query: str, top_k: int = 20) -> List[Result]:
        """
        Perform keyword search using FTS5 (BM25).
        """
        results = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT node_id, name, file, content, rank 
                FROM search_index 
                WHERE search_index MATCH ? 
                ORDER BY rank 
                LIMIT ?
            """, (query, top_k))
            
            for row in cursor:
                nid, name, file, content, rank = row
                # FTS5 rank is lower is better, but we want score (higher is better)
                # Simple inversion or just use rank as is (negative)
                score = -rank 
                
                # Extract summary from content (simplified)
                summary = content.replace(f"{name} ", "", 1)
                
                results.append(Result(
                    node_id=nid,
                    name=name,
                    file=file,
                    line=0, # Need to look up if needed
                    summary=summary,
                    score=score
                ))
        return results
