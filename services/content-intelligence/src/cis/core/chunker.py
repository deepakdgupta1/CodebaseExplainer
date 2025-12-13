"""
AST-Aware Chunker using tree-sitter.

Chunks code by logical boundaries (functions, classes, modules)
preserving semantic meaning and code structure.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import tree_sitter_python as tspython
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logging.warning("tree-sitter not available, falling back to regex chunking")


@dataclass
class CodeChunk:
    """A chunk of code extracted from source file."""
    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    chunk_type: str  # function, class, module, import_block
    symbol_name: str
    signature: str
    content: str
    docstring: Optional[str] = None
    
    # For dependency tracking
    imports: List[str] = field(default_factory=list)
    calls: List[str] = field(default_factory=list)
    inherits: List[str] = field(default_factory=list)
    
    # Metadata
    complexity: int = 0
    fingerprint: str = ""


class ASTChunker:
    """
    AST-aware code chunker using tree-sitter.

    Extracts logical code units (functions, classes) as chunks,
    preserving semantic boundaries and structure.

    Attributes:
        parser: Tree-sitter parser instance.
        include_docstrings: Whether to include docstrings in chunks.
        include_decorators: Whether to include decorators.
    """

    def __init__(
        self,
        include_docstrings: bool = True,
        include_decorators: bool = True,
        min_chunk_lines: int = 5
    ) -> None:
        """
        Initialize the AST chunker.

        Args:
            include_docstrings: Include docstrings in chunks.
            include_decorators: Include decorators with functions/classes.
            min_chunk_lines: Minimum lines for a chunk.
        """
        self.include_docstrings = include_docstrings
        self.include_decorators = include_decorators
        self.min_chunk_lines = min_chunk_lines

        self.parser: Optional[Parser] = None
        if TREE_SITTER_AVAILABLE:
            self._init_parser()

    def _init_parser(self) -> None:
        """Initialize tree-sitter parser for Python."""
        try:
            self.parser = Parser(Language(tspython.language()))
            logging.info("Tree-sitter Python parser initialized")
        except Exception as e:
            logging.error(f"Failed to init tree-sitter: {e}")
            self.parser = None

    def chunk_file(self, file_path: str, source: str) -> List[CodeChunk]:
        """
        Chunk a source file into logical code units.

        Args:
            file_path: Path to the source file.
            source: Source code content.

        Returns:
            List of CodeChunk objects.
        """
        if self.parser:
            return self._chunk_with_treesitter(file_path, source)
        else:
            return self._chunk_with_regex(file_path, source)

    def _chunk_with_treesitter(self, file_path: str, source: str) -> List[CodeChunk]:
        """Chunk using tree-sitter AST."""
        chunks = []
        tree = self.parser.parse(source.encode('utf-8'))

        # Extract imports block
        imports_chunk = self._extract_imports(file_path, source, tree)
        if imports_chunk:
            chunks.append(imports_chunk)

        # Extract functions and classes
        for node in self._traverse(tree.root_node):
            if node.type == 'function_definition':
                chunk = self._extract_function(file_path, source, node)
                if chunk:
                    chunks.append(chunk)
            elif node.type == 'class_definition':
                chunk = self._extract_class(file_path, source, node)
                if chunk:
                    chunks.append(chunk)

        return chunks

    def _traverse(self, node):
        """Traverse AST nodes depth-first."""
        yield node
        for child in node.children:
            # Don't recurse into function/class bodies for top-level extraction
            if node.type not in ('function_definition', 'class_definition'):
                yield from self._traverse(child)

    def _extract_function(self, file_path: str, source: str, node) -> Optional[CodeChunk]:
        """Extract a function definition as a chunk."""
        name_node = node.child_by_field_name('name')
        if not name_node:
            return None

        name = source[name_node.start_byte:name_node.end_byte]
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        if end_line - start_line < self.min_chunk_lines:
            return None

        content = source[node.start_byte:node.end_byte]

        # Extract signature (first line)
        lines = content.split('\n')
        signature = lines[0] if lines else ""

        # Extract docstring
        docstring = self._extract_docstring(node, source)

        return CodeChunk(
            chunk_id=f"{file_path}:{name}:{start_line}",
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type="function",
            symbol_name=name,
            signature=signature,
            content=content,
            docstring=docstring,
            calls=self._extract_calls(node, source),
        )

    def _extract_class(self, file_path: str, source: str, node) -> Optional[CodeChunk]:
        """Extract a class definition as a chunk."""
        name_node = node.child_by_field_name('name')
        if not name_node:
            return None

        name = source[name_node.start_byte:name_node.end_byte]
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        content = source[node.start_byte:node.end_byte]

        # Extract signature
        lines = content.split('\n')
        signature = lines[0] if lines else ""

        # Extract base classes
        bases = self._extract_bases(node, source)

        # Extract docstring
        docstring = self._extract_docstring(node, source)

        return CodeChunk(
            chunk_id=f"{file_path}:{name}:{start_line}",
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type="class",
            symbol_name=name,
            signature=signature,
            content=content,
            docstring=docstring,
            inherits=bases,
        )

    def _extract_imports(self, file_path: str, source: str, tree) -> Optional[CodeChunk]:
        """Extract all imports as a single chunk."""
        import_nodes = []

        for node in tree.root_node.children:
            if node.type in ('import_statement', 'import_from_statement'):
                import_nodes.append(node)

        if not import_nodes:
            return None

        start_line = import_nodes[0].start_point[0] + 1
        end_line = import_nodes[-1].end_point[0] + 1

        content_parts = []
        imports = []
        for node in import_nodes:
            text = source[node.start_byte:node.end_byte]
            content_parts.append(text)
            imports.append(text)

        return CodeChunk(
            chunk_id=f"{file_path}:imports:{start_line}",
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type="import_block",
            symbol_name="imports",
            signature="",
            content='\n'.join(content_parts),
            imports=imports,
        )

    def _extract_docstring(self, node, source: str) -> Optional[str]:
        """Extract docstring from a function or class."""
        if not self.include_docstrings:
            return None

        body = node.child_by_field_name('body')
        if body and body.children:
            first_stmt = body.children[0]
            if first_stmt.type == 'expression_statement':
                expr = first_stmt.children[0] if first_stmt.children else None
                if expr and expr.type == 'string':
                    return source[expr.start_byte:expr.end_byte]
        return None

    def _extract_bases(self, node, source: str) -> List[str]:
        """Extract base classes from a class definition."""
        bases = []
        args = node.child_by_field_name('superclasses')
        if args:
            for child in args.children:
                if child.type == 'identifier':
                    bases.append(source[child.start_byte:child.end_byte])
        return bases

    def _extract_calls(self, node, source: str) -> List[str]:
        """Extract function calls from a node."""
        calls = []

        def find_calls(n):
            if n.type == 'call':
                func = n.child_by_field_name('function')
                if func:
                    calls.append(source[func.start_byte:func.end_byte])
            for child in n.children:
                find_calls(child)

        find_calls(node)
        return calls

    def _chunk_with_regex(self, file_path: str, source: str) -> List[CodeChunk]:
        """Fallback chunking using regex when tree-sitter unavailable."""
        import re

        chunks = []
        lines = source.split('\n')

        # Simple pattern matching for functions and classes
        func_pattern = re.compile(r'^(\s*)def\s+(\w+)\s*\(')
        class_pattern = re.compile(r'^(\s*)class\s+(\w+)\s*[:\(]')

        i = 0
        while i < len(lines):
            line = lines[i]

            func_match = func_pattern.match(line)
            class_match = class_pattern.match(line)

            if func_match or class_match:
                match = func_match or class_match
                indent = len(match.group(1))
                name = match.group(2)
                chunk_type = "function" if func_match else "class"

                start_line = i + 1
                end_line = i + 1

                # Find end of block
                j = i + 1
                while j < len(lines):
                    next_line = lines[j]
                    if next_line.strip() and not next_line.startswith(' ' * (indent + 1)):
                        if not next_line.startswith(' ' * indent) or next_line.strip():
                            break
                    end_line = j + 1
                    j += 1

                content = '\n'.join(lines[i:end_line])

                chunks.append(CodeChunk(
                    chunk_id=f"{file_path}:{name}:{start_line}",
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    chunk_type=chunk_type,
                    symbol_name=name,
                    signature=line.strip(),
                    content=content,
                ))

                i = end_line
            else:
                i += 1

        return chunks
