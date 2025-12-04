from typing import Optional
import tree_sitter
from tree_sitter import Language, Parser
import tree_sitter_python
import tree_sitter_typescript

class TreeSitterParser:
    """
    Wrapper around tree-sitter parser for specific languages.
    Handles grammar loading and initialization.
    """
    def __init__(self, language: str):
        self.language_name = language
        self.parser = Parser()
        
        try:
            if language == 'python':
                # Load Python grammar
                self.lang = Language(tree_sitter_python.language())
            elif language == 'typescript':
                # Load TypeScript grammar
                # Note: We default to the standard typescript grammar. 
                # For TSX/JSX, we might need to switch to language_tsx() if parsing fails or if explicitly requested.
                self.lang = Language(tree_sitter_typescript.language_typescript())
            elif language == 'tsx':
                # Explicit support for TSX if needed
                self.lang = Language(tree_sitter_typescript.language_tsx())
            else:
                raise ValueError(f"Unsupported language: {language}")
                
            self.parser.set_language(self.lang)
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize tree-sitter for {language}: {e}")

    def parse_bytes(self, content: bytes) -> tree_sitter.Tree:
        """
        Parse source code bytes into a tree-sitter Tree.
        """
        return self.parser.parse(content)
