from dataclasses import dataclass
from typing import List, Optional
from tree_sitter import Tree, Node, Language, Query, QueryCursor
import tree_sitter_python
import tree_sitter_typescript
from .complexity import compute_cyclomatic_complexity, compute_loc

@dataclass
class NodeInfo:
    type: str
    name: str
    line: int
    end_line: int
    source_code: str
    docstring: Optional[str] = None
    signature: Optional[str] = None
    complexity: int = 0
    loc: int = 0

class NodeExtractor:
    def __init__(self):
        # Initialize languages for queries
        self.langs = {
            'python': Language(tree_sitter_python.language()),
            'typescript': Language(tree_sitter_typescript.language_typescript())
        }
        
        # Define queries
        self.queries = {
            'python': """
                (function_definition
                    name: (identifier) @name
                    body: (block) @body) @function
                
                (class_definition
                    name: (identifier) @name
                    body: (block) @body) @class
            """,
            'typescript': """
                (function_declaration
                    name: (identifier) @name
                    body: (statement_block) @body) @function
                
                (class_declaration
                    name: (type_identifier) @name
                    body: (class_body) @body) @class
                    
                (method_definition
                    name: (property_identifier) @name
                    body: (statement_block) @body) @method
            """
        }

    def extract_all_nodes(self, tree: Tree, language: str, source_bytes: bytes) -> List[NodeInfo]:
        if language not in self.langs:
            return []
            
        lang_obj = self.langs[language]
        query_str = self.queries.get(language)
        if not query_str:
            return []
            
        query = Query(lang_obj, query_str)
        cursor = QueryCursor(query)
        captures_dict = cursor.captures(tree.root_node)
        
        nodes = []
        # Captures is now a dict: {capture_name: [nodes]}
        # We process main captures (function, class, method)
        
        # Helper to process a captured node
        processed_ids = set()
        
        # Iterate through the relevant capture names
        for capture_name in ['function', 'class', 'method']:
            if capture_name not in captures_dict:
                continue
            for node in captures_dict[capture_name]:
                if node.id in processed_ids:
                    continue
                processed_ids.add(node.id)
                
                # Extract details
                name_node = node.child_by_field_name('name')
                name = source_bytes[name_node.start_byte:name_node.end_byte].decode('utf-8') if name_node else "anonymous"
                
                source = source_bytes[node.start_byte:node.end_byte].decode('utf-8')
                
                # Extract docstring (simplified)
                docstring = self._extract_docstring(node, language, source_bytes)
                
                # Extract signature (first line or up to body)
                signature = self._extract_signature(node, source_bytes)
                
                # Metrics
                complexity = compute_cyclomatic_complexity(node, language)
                loc = compute_loc(source)
                
                nodes.append(NodeInfo(
                    type=capture_name,
                    name=name,
                    line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    source_code=source,
                    docstring=docstring,
                    signature=signature,
                    complexity=complexity,
                    loc=loc
                ))
                
        return nodes

    def _extract_docstring(self, node: Node, language: str, source_bytes: bytes) -> Optional[str]:
        if language == 'python':
            body = node.child_by_field_name('body')
            if body and body.child_count > 0:
                first_stmt = body.child(0)
                if first_stmt.type == 'expression_statement':
                    expr = first_stmt.child(0)
                    if expr.type == 'string':
                        return source_bytes[expr.start_byte:expr.end_byte].decode('utf-8').strip('"""\'\'\'')
        # TypeScript JSDoc usually precedes the node, tree-sitter might capture it if we look at previous sibling
        # For now, simplified: return None for TS or implement complex logic later
        return None

    def _extract_signature(self, node: Node, source_bytes: bytes) -> str:
        # Heuristic: take code from start up to the body start
        body = node.child_by_field_name('body')
        if body:
            sig_bytes = source_bytes[node.start_byte:body.start_byte]
            return sig_bytes.decode('utf-8').strip().rstrip(':').rstrip('{')
        return "unknown"
