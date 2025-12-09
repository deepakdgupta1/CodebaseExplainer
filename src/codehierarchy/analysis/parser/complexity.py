from typing import Set
from tree_sitter import Node


def compute_loc(source: str) -> int:
    """
    Count Lines of Code (LOC), excluding empty lines and comments.
    """
    lines = source.splitlines()
    # Simple heuristic: ignore empty lines and lines starting with # or //
    # This is language-agnostic but approximate
    count = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('#') or stripped.startswith('//'):
            continue
        count += 1
    return count


def compute_comment_ratio(source: str) -> float:
    """
    Compute ratio of comment lines to total lines.
    """
    lines = source.splitlines()
    if not lines:
        return 0.0

    comment_lines = 0
    total_lines = len(lines)

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#') or stripped.startswith('//'):
            comment_lines += 1

    return comment_lines / total_lines if total_lines > 0 else 0.0


def compute_cyclomatic_complexity(node: Node, language: str) -> int:
    """
    Compute Cyclomatic Complexity by counting branching nodes.
    Base complexity is 1.
    """
    complexity = 1

    # Define branching node types for supported languages
    branching_types: Set[str] = set()

    if language == 'python':
        branching_types = {
            'if_statement', 'elif_clause', 'for_statement', 'while_statement',
            'except_clause', 'with_statement', 'assert_statement',
            'boolean_operator'  # and, or
        }
    elif language in ['typescript', 'tsx', 'javascript']:
        branching_types = {
            'if_statement', 'for_statement', 'for_in_statement', 'for_of_statement',
            'while_statement', 'do_statement', 'switch_case', 'catch_clause',
            'ternary_expression', 'binary_expression'  # need to check for && and ||
        }

    # Recursive traversal
    def traverse(n: Node) -> None:
        nonlocal complexity
        if n.type in branching_types:
            # For binary expressions in TS/JS, checks if operator is && or ||
            if n.type == 'binary_expression' or (
                    language == 'python' and n.type == 'boolean_operator'):
                # In Python boolean_operator is 'and'/'or'
                # In TS binary_expression can be +, -, etc. Need to check
                # operator child.
                pass  # Simplified: count all for now or refine
                # Refinement for TS:
                if language in [
                    'typescript',
                    'tsx',
                        'javascript'] and n.type == 'binary_expression':
                    # Check operator
                    operator = n.child_by_field_name('operator')
                    if operator and operator.type in ['&&', '||']:
                        complexity += 1
                    return  # Don't double count if we handled it, or continue?
                    # Actually, let's just count it if it matches

            # General case
            if n.type != 'binary_expression':
                complexity += 1
            elif language == 'python' and n.type == 'boolean_operator':
                complexity += 1

        for child in n.children:
            traverse(child)

    traverse(node)
    return complexity
