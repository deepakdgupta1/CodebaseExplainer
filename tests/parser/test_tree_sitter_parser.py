import pytest
from codehierarchy.parser.tree_sitter_parser import TreeSitterParser

def test_python_parser_initialization():
    parser = TreeSitterParser('python')
    assert parser.language_name == 'python'
    assert parser.lang is not None

def test_typescript_parser_initialization():
    parser = TreeSitterParser('typescript')
    assert parser.language_name == 'typescript'
    assert parser.lang is not None

def test_unsupported_language():
    with pytest.raises(ValueError):
        TreeSitterParser('ruby')

def test_parse_python_code():
    parser = TreeSitterParser('python')
    code = b"def hello(): pass"
    tree = parser.parse_bytes(code)
    assert tree.root_node.type == 'module'
    assert tree.root_node.child_count > 0

def test_parse_typescript_code():
    parser = TreeSitterParser('typescript')
    code = b"function hello() { return 1; }"
    tree = parser.parse_bytes(code)
    assert tree.root_node.type == 'program'
    assert tree.root_node.child_count > 0
