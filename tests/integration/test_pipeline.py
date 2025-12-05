import pytest
from pathlib import Path
from codehierarchy.core.pipeline.orchestrator import Orchestrator
from codehierarchy.config.schema import Config

# Integration test requires real components or heavy mocking.
# We'll do a "smoke test" that runs the pipeline on a tiny dummy repo
# but mocks the LLM to avoid API calls and time.

from unittest.mock import patch, MagicMock

@pytest.fixture
def test_repo(tmp_path):
    repo = tmp_path / "test_repo"
    repo.mkdir()
    (repo / "main.py").write_text("def main(): pass")
    return repo

def test_pipeline_end_to_end(test_repo, tmp_path):
    config = Config()
    config.system.output_dir = tmp_path / "output"
    
    # Mock LLM and Embedder to be fast
    with patch('codehierarchy.core.pipeline.orchestrator.DeepSeekSummarizer') as MockSummarizer, \
         patch('codehierarchy.core.pipeline.orchestrator.HighQualityEmbedder') as MockEmbedder, \
         patch('codehierarchy.core.pipeline.orchestrator.ParallelParser') as MockParser:
    
        # Configure ParallelParser to return dummy results
        mock_parser_instance = MockParser.return_value
        
        from codehierarchy.analysis.parser.node_extractor import NodeInfo
        from codehierarchy.analysis.parser.parallel_parser import ParseResult
        
        dummy_node = NodeInfo(
            type='function',
            name='main',
            line=1,
            end_line=1,
            source_code='def main(): pass',
            docstring=None,
            signature='def main()',
            complexity=1,
            loc=1
        )
        
        mock_parser_instance.parse_repository.return_value = {
            test_repo / "main.py": ParseResult(nodes=[dummy_node], edges=[], complexity=1)
        }

        mock_summ = MockSummarizer.return_value
        mock_summ.summarize_batch.return_value = {
            f"{test_repo}/main.py:main:1": "Summary of main"
        }

        mock_emb = MockEmbedder.return_value
        mock_emb.build_index.return_value = (MagicMock(), {})

        orchestrator = Orchestrator(config)
        results = orchestrator.run_pipeline(test_repo)

        assert results['graph'].number_of_nodes() > 0
        assert len(results['summaries']) > 0
        assert (config.system.output_dir / "performance-metrics.json").exists()
