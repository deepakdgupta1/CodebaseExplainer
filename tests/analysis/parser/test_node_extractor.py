import pytest
from unittest.mock import MagicMock, patch
from codehierarchy.analysis.parser.node_extractor import NodeExtractor

@pytest.fixture
def mock_dependencies():
    with patch('codehierarchy.analysis.parser.node_extractor.tree_sitter_python') as mock_py, \
         patch('codehierarchy.analysis.parser.node_extractor.tree_sitter_typescript') as mock_ts, \
         patch('codehierarchy.analysis.parser.node_extractor.Language') as mock_lang, \
         patch('codehierarchy.analysis.parser.node_extractor.Query') as mock_query_class, \
         patch('codehierarchy.analysis.parser.node_extractor.QueryCursor') as mock_cursor_class:
        
        mock_py.language.return_value = 1
        mock_ts.language_typescript.return_value = 2
        
        # Setup mock language object
        mock_language_instance = MagicMock()
        mock_lang.return_value = mock_language_instance
        
        # Setup mock Query class to return mock query instance
        mock_query = MagicMock()
        mock_query_class.return_value = mock_query
        
        # Setup mock QueryCursor to return mock cursor instance
        mock_cursor = MagicMock()
        mock_cursor_class.return_value = mock_cursor
        
        # Yield the mock cursor so tests can configure captures on it
        yield mock_cursor

@pytest.fixture
def extractor(mock_dependencies):
    return NodeExtractor()

def test_extract_python_function(extractor, mock_dependencies):
    code = b"def my_func(): pass"
    mock_cursor = mock_dependencies
    
    mock_tree = MagicMock()
    
    # Mock captures
    mock_node = MagicMock()
    mock_node.type = 'function_definition'
    mock_node.start_point = (1, 0)
    mock_node.end_point = (1, 19)
    mock_node.start_byte = 0
    mock_node.end_byte = 19
    mock_node.children = [] # No children for complexity
    
    # Mock name node and body node
    name_node = MagicMock()
    name_node.start_byte = 4
    name_node.end_byte = 11
    name_node.text = b"my_func"
    
    body_node = MagicMock()
    body_node.child_count = 0 # No docstring
    
    def child_by_field_side_effect(name):
        if name == 'name':
            return name_node
        if name == 'body':
            return body_node
        return None
        
    mock_node.child_by_field_name.side_effect = child_by_field_side_effect
    
    mock_node.id = 1
    
    # Mock cursor.captures return value (dict format: {capture_name: [nodes]})
    mock_cursor.captures.return_value = {'function': [mock_node]}
    
    nodes = extractor.extract_all_nodes(mock_tree, 'python', code)
    
    assert len(nodes) == 1
    assert nodes[0].name == 'my_func'
    assert nodes[0].type == 'function'

def test_extract_python_class(extractor, mock_dependencies):
    code = b"class MyClass: pass"
    mock_cursor = mock_dependencies
    
    mock_tree = MagicMock()
    
    class_node = MagicMock()
    class_node.type = 'class_definition'
    class_node.start_point = (1, 0)
    class_node.end_point = (1, 19)
    class_node.start_byte = 0
    class_node.end_byte = 19
    class_node.children = []
    
    name_node = MagicMock()
    name_node.start_byte = 6
    name_node.end_byte = 13
    name_node.text = b"MyClass"
    
    body_node = MagicMock()
    body_node.child_count = 0
    
    def child_by_field_side_effect(name):
        if name == 'name':
            return name_node
        if name == 'body':
            return body_node
        return None
        
    class_node.child_by_field_name.side_effect = child_by_field_side_effect
    
    class_node.id = 2
    
    mock_cursor.captures.return_value = {'class': [class_node]}
    
    nodes = extractor.extract_all_nodes(mock_tree, 'python', code)
    
    assert len(nodes) == 1
    assert nodes[0].name == 'MyClass'
    assert nodes[0].type == 'class'

def test_extract_typescript_function(extractor, mock_dependencies):
    code = b"function tsFunc() {}"
    mock_cursor = mock_dependencies
    
    mock_tree = MagicMock()
    
    func_node = MagicMock()
    func_node.type = 'function_declaration'
    func_node.start_point = (1, 0)
    func_node.end_point = (1, 20)
    func_node.start_byte = 0
    func_node.end_byte = 20
    func_node.children = []
    
    name_node = MagicMock()
    name_node.start_byte = 9
    name_node.end_byte = 15
    name_node.text = b"tsFunc"
    
    body_node = MagicMock()
    body_node.child_count = 0
    
    def child_by_field_side_effect(name):
        if name == 'name':
            return name_node
        if name == 'body':
            return body_node
        return None
        
    func_node.child_by_field_name.side_effect = child_by_field_side_effect
    
    func_node.id = 3
    
    mock_cursor.captures.return_value = {'function': [func_node]}
    
    nodes = extractor.extract_all_nodes(mock_tree, 'typescript', code)
    
    assert len(nodes) == 1
    assert nodes[0].name == 'tsFunc'

def test_extract_typescript_class_method(extractor, mock_dependencies):
    code = b"class TSClass { m() {} }"
    mock_cursor = mock_dependencies
    
    mock_tree = MagicMock()
    
    class_node = MagicMock()
    class_node.type = 'class_declaration'
    class_node.start_point = (0,0)
    class_node.end_point = (0,10)
    class_node.start_byte = 0
    class_node.end_byte = 10
    class_node.children = []
    
    class_name_node = MagicMock()
    class_name_node.start_byte = 6
    class_name_node.end_byte = 13
    class_name_node.text = b"TSClass"
    
    class_body = MagicMock()
    class_body.child_count = 0
    
    def class_child_side_effect(name):
        if name == 'name':
            return class_name_node
        if name == 'body':
            return class_body
        return None
        
    class_node.child_by_field_name.side_effect = class_child_side_effect
    
    class_node.id = 4
    
    method_node = MagicMock()
    method_node.type = 'method_definition'
    method_node.start_point = (0,12)
    method_node.end_point = (0,20)
    method_node.start_byte = 12
    method_node.end_byte = 20
    method_node.children = []
    
    method_name_node = MagicMock()
    method_name_node.start_byte = 16
    method_name_node.end_byte = 17
    method_name_node.text = b"m"
    
    method_body = MagicMock()
    method_body.child_count = 0
    
    def method_child_side_effect(name):
        if name == 'name':
            return method_name_node
        if name == 'body':
            return method_body
        return None
        
    method_node.child_by_field_name.side_effect = method_child_side_effect
    
    method_node.id = 5
    
    mock_cursor.captures.return_value = {
        'class': [class_node],
        'method': [method_node]
    }
    
    nodes = extractor.extract_all_nodes(mock_tree, 'typescript', code)
    
    assert len(nodes) == 2
    names = {n.name for n in nodes}
    assert 'TSClass' in names
    assert 'm' in names
