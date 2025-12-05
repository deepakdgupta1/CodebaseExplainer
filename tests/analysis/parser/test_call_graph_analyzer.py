import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from codehierarchy.analysis.parser.call_graph_analyzer import CallGraphAnalyzer

@pytest.fixture
def mock_dependencies():
    with patch('codehierarchy.analysis.parser.call_graph_analyzer.tree_sitter_python') as mock_py, \
         patch('codehierarchy.analysis.parser.call_graph_analyzer.tree_sitter_typescript') as mock_ts, \
         patch('codehierarchy.analysis.parser.call_graph_analyzer.Language') as mock_lang:
        
        mock_py.language.return_value = 1
        mock_ts.language_typescript.return_value = 2
        
        # Setup mock language object to return mock query
        mock_language_instance = MagicMock()
        mock_lang.return_value = mock_language_instance
        
        mock_query = MagicMock()
        mock_language_instance.query.return_value = mock_query
        
        yield mock_query

def test_analyze_python_calls(mock_dependencies):
    code = b"def caller(): callee()"
    mock_query = mock_dependencies
    
    mock_tree = MagicMock()
    analyzer = CallGraphAnalyzer('python')
    
    # Mock captures for calls
    call_node = MagicMock()
    call_node.text = b"callee"
    
    # Mock enclosing scope (caller)
    # The analyzer traverses up from call_node to find function definition
    # We need to mock parent chain
    func_def = MagicMock()
    func_def.type = 'function_definition'
    func_def.child_by_field_name.return_value.text = b"caller"
    
    call_node.parent = func_def
    func_def.parent = None
    
    # Configure mock query to return this capture when called for 'call' query
    # The analyzer calls query() multiple times (call, import, inheritance)
    # We need to ensure it returns captures only for the call query
    # But since we mock the query object returned by lang.query(), and lang.query() is called 3 times,
    # we can make it return different mocks or the same mock with side effects.
    # Simpler: make captures return list for first call, empty for others?
    # Or just return all captures and let the loop handle it?
    # Wait, analyzer creates NEW query object for each query type.
    # self.lang.query(queries['call']) -> query_obj
    # query_obj.captures(...)
    
    # So we need mock_language_instance.query to return different mocks based on input?
    # Or just return a generic mock that returns captures for all?
    # If we return captures for all, the loop for 'import' will try to process 'call' nodes if we are not careful.
    # But the loop iterates over captures.
    # If we return [(call_node, 'callee')] for ALL queries, then:
    # - call loop: processes it.
    # - import loop: processes it? No, import loop expects 'module' capture name?
    # Let's check analyzer code:
    # for node, _ in call_query.captures(...): ... (it ignores capture name?)
    # Yes: `for node, _ in call_query.captures(tree.root_node):`
    # It ignores capture name!
    # So if we return the same list for all queries, it will process call_node as import and inheritance too!
    # This will crash or produce wrong edges.
    
    # We must distinguish queries.
    # We can use side_effect on lang.query.
    
    def query_side_effect(query_str):
        q = MagicMock()
        if '(call' in query_str:
            q.captures.return_value = [(call_node, 'callee')]
        else:
            q.captures.return_value = []
        return q
        
    analyzer.lang.query.side_effect = query_side_effect
    
    edges = analyzer.analyze(Path('test.py'), mock_tree)
    
    call_edges = [e for e in edges if e.type == 'call']
    assert len(call_edges) == 1
    assert call_edges[0].source == 'caller'
    assert call_edges[0].target == 'callee'

def test_analyze_python_imports(mock_dependencies):
    code = b"import os"
    mock_query = mock_dependencies # This is the mock returned by fixture, but we override side_effect on instance
    
    mock_tree = MagicMock()
    analyzer = CallGraphAnalyzer('python')
    
    import_node = MagicMock()
    import_node.text = b"os"
    
    def query_side_effect(query_str):
        q = MagicMock()
        if '(import' in query_str:
            q.captures.return_value = [(import_node, 'module')]
        else:
            q.captures.return_value = []
        return q
        
    analyzer.lang.query.side_effect = query_side_effect
    
    edges = analyzer.analyze(Path('test.py'), mock_tree)
    
    import_edges = [e for e in edges if e.type == 'import']
    assert len(import_edges) == 1
    assert import_edges[0].target == 'os'

def test_analyze_typescript_inheritance(mock_dependencies):
    code = b"class Child extends Parent {}"
    
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
    
    def query_side_effect(query_str):
        q = MagicMock()
        if '(class_declaration' in query_str: # Inheritance query
            q.captures.return_value = [(parent_node, 'parent')]
        else:
            q.captures.return_value = []
        return q
        
    analyzer.lang.query.side_effect = query_side_effect
    
    edges = analyzer.analyze(Path('test.ts'), mock_tree)
    
    inherit_edges = [e for e in edges if e.type == 'inheritance']
    assert len(inherit_edges) == 1
    assert inherit_edges[0].source == 'Child'
    assert inherit_edges[0].target == 'Parent'
