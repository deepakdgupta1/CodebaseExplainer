"""Unit tests for PromptRegistry."""

import pytest
from lss.core.prompt_registry import PromptRegistry, PromptTemplate, DEFAULT_PROMPTS


class TestPromptRegistry:
    """Tests for PromptRegistry class."""

    @pytest.fixture
    def registry(self):
        """Create fresh registry."""
        return PromptRegistry()

    def test_init_loads_defaults(self, registry):
        """Test that default prompts are loaded."""
        prompts = registry.list_prompts()
        
        assert len(prompts) >= 3
        assert registry.get("code_abstractive") is not None

    def test_register_custom_prompt(self, registry):
        """Test registering custom prompt."""
        template = PromptTemplate(
            id="custom_test",
            name="Custom Test",
            template="Summarize: ${code}",
            summary_type="custom",
            variables=["code"]
        )
        
        registry.register(template)
        
        assert registry.get("custom_test") is not None

    def test_get_nonexistent(self, registry):
        """Test getting nonexistent prompt."""
        result = registry.get("does_not_exist")
        assert result is None

    def test_render_simple(self, registry):
        """Test rendering a prompt."""
        result = registry.render(
            "code_extractive",
            code="def hello(): pass"
        )
        
        assert "def hello(): pass" in result

    def test_render_missing_variable(self, registry):
        """Test render with missing required variable."""
        with pytest.raises(ValueError) as exc:
            registry.render("code_abstractive")
        
        assert "Missing required variables" in str(exc.value)

    def test_render_nonexistent_prompt(self, registry):
        """Test render with nonexistent prompt."""
        with pytest.raises(KeyError):
            registry.render("does_not_exist", code="test")

    def test_list_prompts_by_type(self, registry):
        """Test filtering prompts by type."""
        abstractive = registry.list_prompts(summary_type="abstractive")
        
        for prompt in abstractive:
            assert prompt.summary_type == "abstractive"

    def test_delete_prompt(self, registry):
        """Test deleting a prompt."""
        registry.register(PromptTemplate(
            id="to_delete",
            name="Delete Me",
            template="test",
            summary_type="test",
            variables=[]
        ))
        
        result = registry.delete("to_delete")
        
        assert result is True
        assert registry.get("to_delete") is None

    def test_delete_nonexistent(self, registry):
        """Test deleting nonexistent prompt."""
        result = registry.delete("does_not_exist")
        assert result is False


class TestPromptTemplate:
    """Tests for PromptTemplate dataclass."""

    def test_create_template(self):
        """Test creating a template."""
        template = PromptTemplate(
            id="test",
            name="Test Template",
            template="Hello ${name}",
            summary_type="custom",
            variables=["name"]
        )
        
        assert template.id == "test"
        assert template.version == "1.0"

    def test_default_prompts_structure(self):
        """Test that default prompts have proper structure."""
        for prompt_id, prompt in DEFAULT_PROMPTS.items():
            assert prompt.id == prompt_id
            assert len(prompt.template) > 0
            assert prompt.summary_type in ["extractive", "abstractive", "hierarchical"]
