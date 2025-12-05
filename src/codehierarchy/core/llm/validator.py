from typing import Tuple
from codehierarchy.analysis.parser.node_extractor import NodeInfo

def validate_summary(summary: str, node_name: str, min_length: int = 50, max_length: int = 1000) -> Tuple[bool, float]:
    """
    Validate a generated summary.
    Returns (is_valid, quality_score).
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
