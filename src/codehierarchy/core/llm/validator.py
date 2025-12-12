"""
Summary validation for LLM-generated content.

This module provides heuristic validation for AI-generated summaries
to catch obvious issues like summaries that are too short or missing
key information about the code being documented.

The validation is intentionally lightweight to avoid being too strict
while still catching low-quality outputs.
"""

from typing import Tuple


def validate_summary(
    summary: str,
    node_name: str,
    min_length: int = 50,
    max_length: int = 1000
) -> Tuple[bool, float]:
    """
    Validate a generated summary and compute a quality score.

    Performs basic heuristic checks to ensure the summary is useful:
    - Length within acceptable bounds
    - Contains the node name (soft requirement)

    Args:
        summary: The generated summary text to validate.
        node_name: The name of the code element being summarized.
        min_length: Minimum acceptable length in characters.
        max_length: Maximum length (advisory, not enforced).

    Returns:
        Tuple of (is_valid, quality_score):
        - is_valid: True if summary meets minimum requirements.
        - quality_score: Float from 0.0 to 1.0 indicating quality.

    Example:
        >>> is_valid, score = validate_summary(
        ...     "The foo function processes input data...",
        ...     "foo"
        ... )
        >>> print(f"Valid: {is_valid}, Score: {score}")
    """
    # 1. Length check
    if len(summary) < min_length:
        return False, 0.0
    if len(summary) > max_length:
        # Just a warning, maybe not invalid
        pass

    # 2. Hallucination check (simple heuristic)
    # If summary mentions "function" but node is a class, etc.
    # Or if it mentions names not in the code.
    # This is hard to do without context.

    # 3. Key info check
    # Should probably mention the node name
    if node_name not in summary:
        # Not necessarily a failure, but lower score
        return True, 0.5

    return True, 1.0
