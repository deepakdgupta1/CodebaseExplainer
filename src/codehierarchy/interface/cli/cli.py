import click
from pathlib import Path
import sys
from codehierarchy.config.loader import load_config
from codehierarchy.core.pipeline.orchestrator import Orchestrator
from codehierarchy.core.search.search_engine import EnterpriseSearchEngine
from codehierarchy.interface.output.markdown_generator import MarkdownGenerator
from codehierarchy.utils.logger import setup_logging

@click.group()
def cli():
    """CodeHierarchy Explainer CLI"""
    pass

@cli.command()
@click.argument('repo_path', type=click.Path(exists=True))
@click.option('--config', 'config_path', type=click.Path(exists=True), help='Path to config file')
@click.option('--output', 'output_dir', type=click.Path(), help='Output directory')
@click.option('--verbose', is_flag=True, help='Enable verbose logging')
def analyze(repo_path, config_path, output_dir, verbose):
    """Analyze a repository and generate documentation."""
    setup_logging(verbose)
    
    try:
        config = load_config(Path(config_path) if config_path else None)
        if output_dir:
            config.system.output_dir = Path(output_dir)
            
        orchestrator = Orchestrator(config)
        results = orchestrator.run_pipeline(Path(repo_path))
        
        if results:
            generator = MarkdownGenerator(config.system.output_dir)
            generator.generate_documentation(results['graph'], results['summaries'])
            print(f"\n✅ Analysis complete! Output in {config.system.output_dir}")
        else:
            print("\n❌ Analysis failed or produced no results.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

@cli.command()
@click.argument('query')
@click.option('--index-dir', type=click.Path(exists=True), required=True, help='Path to index directory')
@click.option('--mode', type=click.Choice(['keyword', 'semantic', 'hybrid']), default='hybrid')
def search(query, index_dir, mode):
    """Search the codebase knowledge base."""
    try:
        engine = EnterpriseSearchEngine(Path(index_dir))
        results = engine.search(query, mode=mode)
        
        print(f"\n🔍 Search results for '{query}' ({mode}):\n")
        for i, res in enumerate(results, 1):
            print(f"{i}. {res.name} ({res.file}) [Score: {res.score:.2f}]")
            print(f"   {res.summary[:150]}...\n")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

def main():
    cli()

if __name__ == "__main__":
    main()
