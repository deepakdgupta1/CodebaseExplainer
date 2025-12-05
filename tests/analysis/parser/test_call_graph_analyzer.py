import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from codehierarchy.analysis.parser.call_graph_analyzer import CallGraphAnalyzer

@pytest.fixture
def mock_dependencies():
    with patch('codehierarchy.analysis.parser.call_graph_analyzer.tree_sitter_python') as mock_py, \
         patch('codehierarchy.analysis.parser.call_graph_analyzer.tree_sitter_typescript') as mock_ts, \
         patch('codehierarchy.analysis.parser.call_graph_analyzer.Language') as mock_lang, \
         patch('codehierarchy.analysis.parser.call_graph_analyzer.Query') as mock_query_class, \
         patch('codehierarchy.analysis.parser.call_graph_analyzer.QueryCursor') as mock_cursor_class:
        
        mock_py.language.return_value = 1
        mock_ts.language_typescript.return_value = 2
        
        # Setup mock language object
        mock_language_instance = MagicMock()
        mock_lang.return_value = mock_language_instance
        
        # Yield the mock QueryCursor class so tests can configure side_effect
        yield mock_cursor_class

def test_analyze_python_calls(mock_dependencies):
    code = b"def caller(): callee()"
    mock_cursor_class = mock_dependencies
    
    mock_tree = MagicMock()
    analyzer = CallGraphAnalyzer('python')
    
    # Mock captures for calls
    call_node = MagicMock()
    call_node.text = b"callee"
    
    # Mock enclosing scope (caller)
    func_def = MagicMock()
    func_def.type = 'function_definition'
    func_def.child_by_field_name.return_value.text = b"caller"
    
    call_node.parent = func_def
    func_def.parent = None
    
    # Configure mock QueryCursor to return different captures based on query
    # The code creates 3 cursors, one for each query type
    call_cursors = []
    
    def cursor_side_effect(query):
        cursor = MagicMock()
        # Return call captures for first cursor (calls query)
        if len(call_cursors) == 0:
            cursor.captures.return_value = {'callee': [call_node]}
        else:
            cursor.captures.return_value = {}
        call_cursors.append(cursor)
        return cursor
        
    mock_cursor_class.side_effect = cursor_side_effect
    
    edges = analyzer.analyze(Path('test.py'), mock_tree)
    
    call_edges = [e for e in edges if e.type == 'call']
    assert len(call_edges) == 1
    assert call_edges[0].source == 'caller'
    assert call_edges[0].target == 'callee'

def test_analyze_python_imports(mock_dependencies):
    code = b"import os"
    mock_cursor_class = mock_dependencies
    
    mock_tree = MagicMock()
    analyzer = CallGraphAnalyzer('python')
    
    import_node = MagicMock()
    import_node.text = b"os"
    
    import_cursors = []
    
    def cursor_side_effect(query):
        cursor = MagicMock()
        # Return import captures for second cursor (imports query)
        if len(import_cursors) == 1:
            cursor.captures.return_value = {'module': [import_node]}
        else:
            cursor.captures.return_value = {}
        import_cursors.append(cursor)
        return cursor
        
    mock_cursor_class.side_effect = cursor_side_effect
    
    edges = analyzer.analyze(Path('test.py'), mock_tree)
    
    import_edges = [e for e in edges if e.type == 'import']
    assert len(import_edges) == 1
    assert import_edges[0].target == 'os'

def test_analyze_typescript_inheritance(mock_dependencies):
    code = b"class Child extends Parent {}"
    mock_cursor_class = mock_dependencies
    
    mock_tree = MagicMock()
    analyzer = CallGraphAnalyzer('typescript')
    
    parent_node = MagicMock()
    parent_node.text = b"Parent"
    
    # Mock class definition parent
    class_node = MagicMock()
    class_node.type = 'class_declaration'
    class_node.child_by_field_name.return_value.text = b"Child"
    
    # parent_node is the identifier 'Parent' inside extends clause
    # We need to link it to class_node via parents
    # The analyzer uses _find_parent_of_type
    parent_node.parent = MagicMock() # extends_clause
    parent_node.parent.parent = MagicMock() # class_heritage
    parent_node.parent.parent.parent = class_node
    class_node.parent = None
    
    inherit_cursors = []
    
    def cursor_side_effect(query):
        cursor = MagicMock()
        # Return inheritance captures for third cursor (inheritance query)
        if len(inherit_cursors) == 2:
            cursor.captures.return_value = {'parent': [parent_node]}
        else:
            cursor.captures.return_value = {}
        inherit_cursors.append(cursor)
        return cursor
        
    mock_cursor_class.side_effect = cursor_side_effect
    
    edges = analyzer.analyze(Path('test.ts'), mock_tree)
    
    inherit_edges = [e for e in edges if e.type == 'inheritance']
    assert len(inherit_edges) == 1
    assert inherit_edges[0].source == 'Child'
    assert inherit_edges[0].target == 'Parent'
