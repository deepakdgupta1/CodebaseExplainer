import pytest
from pathlib import Path
from codehierarchy.parser.call_graph_analyzer import CallGraphAnalyzer
from codehierarchy.parser.tree_sitter_parser import TreeSitterParser

def test_analyze_python_calls():
    code = b"""
def caller():
    callee()
    
def callee():
    pass
"""
    parser = TreeSitterParser('python')
    tree = parser.parse_bytes(code)
    analyzer = CallGraphAnalyzer('python')
    edges = analyzer.analyze(Path('test.py'), tree)
    
    # Should find call from caller to callee
    call_edges = [e for e in edges if e.type == 'call']
    assert len(call_edges) == 1
    assert call_edges[0].source == 'caller'
    assert call_edges[0].target == 'callee'

def test_analyze_python_imports():
    code = b"""
import os
from sys import path
"""
    parser = TreeSitterParser('python')
    tree = parser.parse_bytes(code)
    analyzer = CallGraphAnalyzer('python')
    edges = analyzer.analyze(Path('test.py'), tree)
    
    import_edges = [e for e in edges if e.type == 'import']
    assert len(import_edges) == 2
    targets = {e.target for e in import_edges}
    assert 'os' in targets
    assert 'sys' in targets # or 'path'? The query captures dotted_name as module. 
    # "from sys import path" -> module_name is 'sys'.
    
def test_analyze_typescript_inheritance():
    code = b"""
class Child extends Parent {}
"""
    parser = TreeSitterParser('typescript')
    tree = parser.parse_bytes(code)
    analyzer = CallGraphAnalyzer('typescript')
    edges = analyzer.analyze(Path('test.ts'), tree)
    
    inherit_edges = [e for e in edges if e.type == 'inheritance']
    assert len(inherit_edges) == 1
    assert inherit_edges[0].source == 'Child'
    assert inherit_edges[0].target == 'Parent'
