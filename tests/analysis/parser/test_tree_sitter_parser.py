import pytest
from unittest.mock import MagicMock, patch
from codehierarchy.analysis.parser.tree_sitter_parser import TreeSitterParser

@pytest.fixture
def mock_tree_sitter_langs():
    with patch('codehierarchy.analysis.parser.tree_sitter_parser.tree_sitter_python') as mock_py, \
         patch('codehierarchy.analysis.parser.tree_sitter_parser.tree_sitter_typescript') as mock_ts, \
         patch('codehierarchy.analysis.parser.tree_sitter_parser.Language') as mock_lang, \
         patch('codehierarchy.analysis.parser.tree_sitter_parser.Parser') as mock_parser:
        
        mock_py.language.return_value = 1
        mock_ts.language_typescript.return_value = 2
        mock_lang.return_value = MagicMock()
        mock_parser.return_value = MagicMock()
        yield mock_parser

def test_python_parser_initialization(mock_tree_sitter_langs):
    """Test Python parser initialization."""
    parser = TreeSitterParser('python')
    assert parser.language_name == 'python'
    assert parser.parser is not None

def test_typescript_parser_initialization(mock_tree_sitter_langs):
    """Test TypeScript parser initialization."""
    parser = TreeSitterParser('typescript')
    assert parser.language_name == 'typescript'

def test_unsupported_language(mock_tree_sitter_langs):
    """Test error for unsupported language."""
    # The constructor wraps exceptions in RuntimeError
    with pytest.raises(RuntimeError):
        TreeSitterParser('java')

def test_parse_python_code(mock_tree_sitter_langs):
    """Test parsing Python code."""
    parser = TreeSitterParser('python')
    code = b"def foo(): pass"
    
    mock_tree = MagicMock()
    parser.parser.parse.return_value = mock_tree
    
    tree = parser.parse_bytes(code)
    assert tree == mock_tree
    parser.parser.parse.assert_called_with(code)

def test_parse_typescript_code(mock_tree_sitter_langs):
    """Test parsing TypeScript code."""
    parser = TreeSitterParser('typescript')
    code = b"function foo() {}"
    
    mock_tree = MagicMock()
    parser.parser.parse.return_value = mock_tree
    
    tree = parser.parse_bytes(code)
    assert tree == mock_tree
