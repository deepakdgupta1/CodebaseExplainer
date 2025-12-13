"""
Prompt Registry for managing summarization prompts.

Provides a registry for storing, retrieving, and templating
prompts for different summary types and domains.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from string import Template


logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """A prompt template with metadata."""
    id: str
    name: str
    template: str
    summary_type: str  # extractive, abstractive, hierarchical, custom
    description: str = ""
    variables: List[str] = field(default_factory=list)
    version: str = "1.0"


# Default prompts for code summarization
DEFAULT_PROMPTS = {
    "code_abstractive": PromptTemplate(
        id="code_abstractive",
        name="Code Abstractive Summary",
        template="""Summarize the following code in a concise paragraph.
Focus on what the code DOES, not how it does it.
Include the main purpose, key functionality, and any notable patterns.

Code:
```
${code}
```

Context:
${context}

Summary:""",
        summary_type="abstractive",
        description="Generates a high-level summary of code functionality",
        variables=["code", "context"]
    ),
    
    "code_extractive": PromptTemplate(
        id="code_extractive",
        name="Code Extractive Summary",
        template="""Extract the key information from this code:
- Purpose
- Inputs/Outputs
- Key functions/methods
- Dependencies

Code:
```
${code}
```

Key Points:""",
        summary_type="extractive",
        description="Extracts key facts from code",
        variables=["code"]
    ),
    
    "code_hierarchical": PromptTemplate(
        id="code_hierarchical",
        name="Hierarchical Code Summary",
        template="""Create a hierarchical summary of this code with three levels:

1. ONE-LINER: A single sentence summary
2. PARAGRAPH: A 3-5 sentence explanation
3. DETAILED: Key components and their purposes

Code:
```
${code}
```

File: ${file_path}

---
ONE-LINER:
[Write one sentence]

PARAGRAPH:
[Write 3-5 sentences]

DETAILED:
[List key components]""",
        summary_type="hierarchical",
        description="Multi-level summary for different detail needs",
        variables=["code", "file_path"]
    ),
    
    "function_doc": PromptTemplate(
        id="function_doc",
        name="Function Documentation",
        template="""Generate a docstring for this function:

```
${code}
```

Include:
- Brief description
- Args (with types)
- Returns
- Raises (if any)

Docstring:""",
        summary_type="abstractive",
        description="Generates function documentation",
        variables=["code"]
    ),
}


class PromptRegistry:
    """
    Registry for prompt templates.

    Supports:
    - Default prompts for common use cases
    - Custom prompt registration
    - Template variable substitution
    - Versioning
    """

    def __init__(self) -> None:
        """Initialize the registry with default prompts."""
        self._prompts: Dict[str, PromptTemplate] = {}
        
        # Load defaults
        for prompt_id, prompt in DEFAULT_PROMPTS.items():
            self.register(prompt)

    def register(self, prompt: PromptTemplate) -> None:
        """
        Register a prompt template.

        Args:
            prompt: PromptTemplate to register.
        """
        self._prompts[prompt.id] = prompt
        logger.debug(f"Registered prompt: {prompt.id}")

    def get(self, prompt_id: str) -> Optional[PromptTemplate]:
        """
        Get a prompt by ID.

        Args:
            prompt_id: Prompt identifier.

        Returns:
            PromptTemplate if found, None otherwise.
        """
        return self._prompts.get(prompt_id)

    def render(self, prompt_id: str, **variables) -> str:
        """
        Render a prompt with variable substitution.

        Args:
            prompt_id: Prompt identifier.
            **variables: Variables to substitute.

        Returns:
            Rendered prompt string.

        Raises:
            KeyError: If prompt not found.
            ValueError: If required variables missing.
        """
        prompt = self._prompts.get(prompt_id)
        if not prompt:
            raise KeyError(f"Prompt not found: {prompt_id}")
        
        # Check required variables
        missing = [v for v in prompt.variables if v not in variables]
        if missing:
            raise ValueError(f"Missing required variables: {missing}")
        
        # Substitute using Template
        template = Template(prompt.template)
        return template.safe_substitute(**variables)

    def list_prompts(self, summary_type: Optional[str] = None) -> List[PromptTemplate]:
        """
        List available prompts.

        Args:
            summary_type: Filter by summary type.

        Returns:
            List of matching prompts.
        """
        prompts = list(self._prompts.values())
        if summary_type:
            prompts = [p for p in prompts if p.summary_type == summary_type]
        return prompts

    def delete(self, prompt_id: str) -> bool:
        """
        Delete a prompt.

        Args:
            prompt_id: Prompt identifier.

        Returns:
            True if deleted, False if not found.
        """
        if prompt_id in self._prompts:
            del self._prompts[prompt_id]
            return True
        return False
