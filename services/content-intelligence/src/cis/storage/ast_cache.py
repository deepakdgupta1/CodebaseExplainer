"""
SQLite AST Cache for fast AST lookups.

Stores parsed AST trees, symbols, and fingerprints to avoid
re-parsing unchanged files.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


@dataclass
class ASTCacheEntry:
    """An entry in the AST cache."""
    file_path: str
    ast_json: str
    symbols: List[Dict[str, Any]]
    imports: List[Dict[str, Any]]
    classes: List[Dict[str, Any]]
    functions: List[Dict[str, Any]]
    fingerprint: str
    last_updated: str
    file_size_bytes: int = 0


class ASTCache:
    """
    SQLite-based AST cache for fast lookups.

    Stores parsed AST data, symbols, and fingerprints to enable
    incremental updates without re-parsing unchanged files.

    Schema:
        file_path TEXT PRIMARY KEY
        ast_json TEXT
        symbols JSON
        imports JSON
        classes JSON
        functions JSON
        fingerprint TEXT
        last_updated TIMESTAMP
        file_size_bytes INTEGER
    """

    def __init__(self, db_path: Path) -> None:
        """
        Initialize the AST cache.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ast_cache (
                    file_path TEXT PRIMARY KEY,
                    ast_json TEXT,
                    symbols TEXT,
                    imports TEXT,
                    classes TEXT,
                    functions TEXT,
                    fingerprint TEXT,
                    last_updated TEXT,
                    file_size_bytes INTEGER
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_fingerprint 
                ON ast_cache(fingerprint)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_last_updated 
                ON ast_cache(last_updated)
            """)
            conn.commit()

        logging.info(f"AST cache initialized at {self.db_path}")

    def get(self, file_path: str) -> Optional[ASTCacheEntry]:
        """
        Get cached AST entry for a file.

        Args:
            file_path: Path to source file.

        Returns:
            ASTCacheEntry if cached, None otherwise.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM ast_cache WHERE file_path = ?",
                (file_path,)
            )
            row = cursor.fetchone()

            if row:
                return ASTCacheEntry(
                    file_path=row['file_path'],
                    ast_json=row['ast_json'],
                    symbols=json.loads(row['symbols'] or '[]'),
                    imports=json.loads(row['imports'] or '[]'),
                    classes=json.loads(row['classes'] or '[]'),
                    functions=json.loads(row['functions'] or '[]'),
                    fingerprint=row['fingerprint'],
                    last_updated=row['last_updated'],
                    file_size_bytes=row['file_size_bytes'] or 0
                )

        return None

    def put(self, entry: ASTCacheEntry) -> None:
        """
        Store or update AST entry.

        Args:
            entry: ASTCacheEntry to store.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO ast_cache 
                (file_path, ast_json, symbols, imports, classes, 
                 functions, fingerprint, last_updated, file_size_bytes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.file_path,
                entry.ast_json,
                json.dumps(entry.symbols),
                json.dumps(entry.imports),
                json.dumps(entry.classes),
                json.dumps(entry.functions),
                entry.fingerprint,
                entry.last_updated,
                entry.file_size_bytes
            ))
            conn.commit()

    def delete(self, file_path: str) -> None:
        """
        Delete cached entry for a file.

        Args:
            file_path: Path to source file.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM ast_cache WHERE file_path = ?",
                (file_path,)
            )
            conn.commit()

    def get_by_fingerprint(self, fingerprint: str) -> List[ASTCacheEntry]:
        """
        Find entries with matching fingerprint.

        Args:
            fingerprint: SHA-256 hash to search for.

        Returns:
            List of matching entries.
        """
        entries = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM ast_cache WHERE fingerprint = ?",
                (fingerprint,)
            )
            for row in cursor.fetchall():
                entries.append(ASTCacheEntry(
                    file_path=row['file_path'],
                    ast_json=row['ast_json'],
                    symbols=json.loads(row['symbols'] or '[]'),
                    imports=json.loads(row['imports'] or '[]'),
                    classes=json.loads(row['classes'] or '[]'),
                    functions=json.loads(row['functions'] or '[]'),
                    fingerprint=row['fingerprint'],
                    last_updated=row['last_updated'],
                    file_size_bytes=row['file_size_bytes'] or 0
                ))

        return entries

    def get_all_symbols(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all symbols indexed by file.

        Returns:
            Dict mapping file_path to list of symbols.
        """
        symbols_by_file = {}
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT file_path, symbols FROM ast_cache"
            )
            for row in cursor.fetchall():
                symbols_by_file[row[0]] = json.loads(row[1] or '[]')

        return symbols_by_file

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT COUNT(*), SUM(file_size_bytes) FROM ast_cache"
            )
            row = cursor.fetchone()
            total_files = row[0] or 0
            total_bytes = row[1] or 0

        return {
            "total_files": total_files,
            "total_bytes": total_bytes,
            "db_path": str(self.db_path)
        }

    def clear(self) -> None:
        """Clear all cached entries."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM ast_cache")
            conn.commit()
        logging.info("AST cache cleared")
