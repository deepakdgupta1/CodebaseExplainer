import pytest
from codehierarchy.parser.node_extractor import NodeExtractor
from codehierarchy.parser.tree_sitter_parser import TreeSitterParser

@pytest.fixture
def extractor():
    return NodeExtractor()

def test_extract_python_function(extractor):
    code = b"""
def my_func(a, b):
    '''Docstring'''
    return a + b
"""
    parser = TreeSitterParser('python')
    tree = parser.parse_bytes(code)
    nodes = extractor.extract_all_nodes(tree, 'python', code)
    
    assert len(nodes) == 1
    node = nodes[0]
    assert node.type == 'function'
    assert node.name == 'my_func'
    assert node.docstring == 'Docstring'
    assert node.line == 2

def test_extract_python_class(extractor):
    code = b"""
class MyClass:
    def method(self):
        pass
"""
    parser = TreeSitterParser('python')
    tree = parser.parse_bytes(code)
    nodes = extractor.extract_all_nodes(tree, 'python', code)
    
    # Should find class and method
    assert len(nodes) == 2
    types = [n.type for n in nodes]
    assert 'class' in types
    assert 'function' in types # Python methods are function_definition in tree-sitter python grammar usually, but my query labeled them? 
    # Wait, my query for python had:
    # (function_definition ... ) @function
    # (class_definition ... ) @class
    # It didn't distinguish methods specifically in the query capture name for Python, 
    # but they are function_definitions inside class.
    # Let's check the query in node_extractor.py
    # Python query:
    # (function_definition ...) @function
    # (class_definition ...) @class
    # So methods will be captured as 'function'.
    
    class_node = next(n for n in nodes if n.type == 'class')
    assert class_node.name == 'MyClass'

def test_extract_typescript_function(extractor):
    code = b"""
function tsFunc(x: number): number {
    return x * 2;
}
"""
    parser = TreeSitterParser('typescript')
    tree = parser.parse_bytes(code)
    nodes = extractor.extract_all_nodes(tree, 'typescript', code)
    
    assert len(nodes) == 1
    assert nodes[0].name == 'tsFunc'
    assert nodes[0].type == 'function'

def test_extract_typescript_class_method(extractor):
    code = b"""
class TSClass {
    myMethod() {}
}
"""
    parser = TreeSitterParser('typescript')
    tree = parser.parse_bytes(code)
    nodes = extractor.extract_all_nodes(tree, 'typescript', code)
    
    assert len(nodes) == 2
    types = [n.type for n in nodes]
    assert 'class' in types
    assert 'method' in types # TS query has @method capture
