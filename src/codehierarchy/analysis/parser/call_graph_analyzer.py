from typing import List, Optional
from dataclasses import dataclass
from pathlib import Path
from tree_sitter import Tree, Node, Language, Query, QueryCursor
import tree_sitter_python
import tree_sitter_typescript


@dataclass
class Edge:
    source: str
    target: str
    type: str  # 'call', 'import', 'inheritance'
    confidence: float


class CallGraphAnalyzer:
    def __init__(self, language: str):
        self.language = language
        self.lang: Optional[Language] = None

        # Initialize language object
        if language == 'python':
            self.lang = Language(tree_sitter_python.language())
        elif language == 'typescript':
            self.lang = Language(tree_sitter_typescript.language_typescript())

        # Define queries
        self.queries = {
            'python': {
                'call': """
                    (call function: (identifier) @callee)
                    (call function: (attribute attribute: (identifier) @callee))
                """,
                'import': """
                    (import_statement name: (dotted_name) @module)
                    (import_from_statement module_name: (dotted_name) @module)
                """,
                'inheritance': """
                    (class_definition superclasses: (argument_list (identifier) @parent))
                """
            },
            'typescript': {
                'call': """
                    (call_expression function: (identifier) @callee)
                    (call_expression function: (member_expression property: (property_identifier) @callee))
                """,
                'import': """
                    (import_statement source: (string) @module)
                """,
                'inheritance': """
                    (class_declaration class_heritage: (class_heritage (extends_clause value: (identifier) @parent)))
                """
            }
        }

    def analyze(self, file: Path, tree: Tree) -> List[Edge]:
        if not self.lang or self.language not in self.queries:
            return []

        edges = []
        queries = self.queries[self.language]

        # Analyze calls
        call_query = Query(self.lang, queries['call'])
        call_cursor = QueryCursor(call_query)
        call_captures = call_cursor.captures(tree.root_node)
        # Captures is now a dict: {capture_name: [nodes]}
        for capture_name, nodes_list in call_captures.items():
            for node in nodes_list:
                if not node.text:
                    continue
                target_name = node.text.decode('utf-8')
                # Source is the enclosing function/method/class
                source_name = self._find_enclosing_scope(node)
                edges.append(Edge(
                    source=source_name,
                    target=target_name,
                    type='call',
                    confidence=1.0
                ))

        # Analyze imports
        import_query = Query(self.lang, queries['import'])
        import_cursor = QueryCursor(import_query)
        import_captures = import_cursor.captures(tree.root_node)
        for capture_name, nodes_list in import_captures.items():
            for node in nodes_list:
                if not node.text:
                    continue
                module_name = node.text.decode('utf-8').strip("'\"")
                edges.append(Edge(
                    source=str(file),
                    target=module_name,
                    type='import',
                    confidence=0.8
                ))

        # Analyze inheritance
        inherit_query = Query(self.lang, queries['inheritance'])
        inherit_cursor = QueryCursor(inherit_query)
        inherit_captures = inherit_cursor.captures(tree.root_node)
        for capture_name, nodes_list in inherit_captures.items():
            for node in nodes_list:
                if not node.text:
                    continue
                parent_name = node.text.decode('utf-8')
                # Source is the class being defined
                # The capture is the parent identifier, so we need to find the
                # class def
                class_node = self._find_parent_of_type(
                    node, ['class_definition', 'class_declaration'])
                if class_node:
                    class_name_node = class_node.child_by_field_name('name')
                    if class_name_node and class_name_node.text:
                        class_name = class_name_node.text.decode('utf-8')
                        edges.append(Edge(
                            source=class_name,
                            target=parent_name,
                            type='inheritance',
                            confidence=0.9
                        ))

        return edges

    def _find_enclosing_scope(self, node: Node) -> str:
        """Find the name of the function/class containing this node."""
        curr: Optional[Node] = node
        while curr:
            if curr.type in [
                'function_definition',
                'function_declaration',
                    'method_definition']:
                name_node = curr.child_by_field_name('name')
                if name_node and name_node.text:
                    return name_node.text.decode('utf-8')
            elif curr.type in ['class_definition', 'class_declaration']:
                name_node = curr.child_by_field_name('name')
                if name_node and name_node.text:
                    return name_node.text.decode('utf-8')
            curr = curr.parent
        return "global"

    def _find_parent_of_type(
            self,
            node: Node,
            types: List[str]) -> Optional[Node]:
        curr: Optional[Node] = node
        while curr:
            if curr.type in types:
                return curr
            curr = curr.parent
        return None
