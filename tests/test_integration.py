"""End-to-end integration tests for LEGIONHERCULES."""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from legionhercules.codegen.sandbox import CodeSandbox, SandboxResult, SandboxStatus
from legionhercules.codegen.generator import CodeGenerator, GenerationStatus
from legionhercules.research.engine import ResearchEngine, ResearchConfig, ResearchStage
from legionhercules.research.summarizer import SummaryStrategy


async def test_sandbox_to_codegen_pipeline():
    """Test CodeGenerator → Sandbox execution pipeline."""
    print("\n[Test] CodeGenerator → Sandbox pipeline...")
    
    # Create generator
    generator = CodeGenerator()
    
    # Generate code from template
    result = await generator.generate_from_template(
        template_name="function",
        function_name="calculate_factorial",
        parameters="n: int",
        docstring="Calculate factorial of a number",
        body="""    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result""",
        return_value="result"
    )
    
    assert result.status == GenerationStatus.SUCCESS
    assert "def calculate_factorial" in result.code
    
    # Execute in sandbox using CodeSandbox directly
    from legionhercules.codegen.sandbox import CodeSandbox
    sandbox = CodeSandbox()
    exec_result = await sandbox.execute(result.code)
    
    assert exec_result is not None
    assert exec_result.status == SandboxStatus.SUCCESS
    
    print(f"✓ Pipeline: Generated → Validated → Executed in sandbox")
    print(f"  Code length: {len(result.code)} chars")
    print(f"  Execution: {exec_result.status.value}")
    
    sandbox.cleanup()
    await generator.cleanup()
    return True


async def test_research_pipeline():
    """Test full research pipeline with mock sources."""
    print("\n[Test] ResearchEngine pipeline...")
    
    engine = ResearchEngine()
    
    # Use mock sources to avoid network dependency
    mock_sources = [
        "https://docs.python.org/3/tutorial/",
        "https://realpython.com/python-basics/",
    ]
    
    config = ResearchConfig(
        max_sources=2,
        min_credibility_score=0.0,  # Allow all for testing
        verify_sources=True,
        summary_strategy=SummaryStrategy.HYBRID,
    )
    
    # Track progress
    progress_stages = []
    def on_progress(stage, message):
        progress_stages.append(stage)
        print(f"  [{stage.value}] {message}")
    
    engine.pipeline.on_progress(on_progress)
    
    # Run research
    report = await engine.research(
        query="Python programming basics",
        sources=mock_sources,
        config=config,
        use_cache=False,
    )
    
    # Verify report structure
    assert report.query == "Python programming basics"
    assert report.config == config
    assert report.stage == ResearchStage.COMPLETE
    assert report.completed_at is not None
    
    # Verify sources were processed
    assert len(report.sources) > 0
    
    # Verify at least some progress stages were hit
    assert len(progress_stages) > 0
    
    print(f"✓ Research pipeline complete")
    print(f"  Sources: {len(report.sources)}")
    print(f"  Stages: {[s.value for s in progress_stages]}")
    print(f"  Synthesis length: {len(report.synthesis)} chars")
    
    await engine.close()
    return True


async def test_cross_module_integration():
    """Test integration between codegen and research modules."""
    print("\n[Test] Cross-module integration...")
    
    # Generate code that uses research module concepts
    generator = CodeGenerator()
    
    code = '''
"""Example module using research concepts."""

from dataclasses import dataclass
from typing import Optional

@dataclass
class SourceInfo:
    """Information about a source."""
    url: str
    credibility: float = 0.5
    verified: bool = False
    
    def to_dict(self):
        return {
            "url": self.url,
            "credibility": self.credibility,
            "verified": self.verified,
        }

def analyze_source(url: str) -> SourceInfo:
    """Analyze a source and return info."""
    # Simple heuristic for demo
    credibility = 0.8 if ".edu" in url else 0.5
    verified = url.startswith("https://")
    return SourceInfo(url=url, credibility=credibility, verified=verified)

# Test the function
if __name__ == "__main__":
    result = analyze_source("https://example.edu")
    print(f"Source: {result.url}")
    print(f"Credibility: {result.credibility}")
    print(f"Verified: {result.verified}")
    assert result.credibility == 0.8
    assert result.verified is True
    print("All tests passed!")
'''
    
    # Generate custom code
    result = await generator.generate_custom(code)
    assert result.status == GenerationStatus.SUCCESS
    
    # Execute in sandbox using CodeSandbox directly
    from legionhercules.codegen.sandbox import CodeSandbox
    sandbox = CodeSandbox()
    exec_result = await sandbox.execute(result.code)
    
    assert exec_result.status == SandboxStatus.SUCCESS
    assert "All tests passed!" in exec_result.stdout
    
    print(f"✓ Cross-module integration working")
    print(f"  Generated code executed successfully")
    print(f"  Output: {exec_result.stdout.strip()}")
    
    sandbox.cleanup()
    await generator.cleanup()
    return True


async def test_error_handling_integration():
    """Test error handling across modules."""
    print("\n[Test] Error handling integration...")
    
    generator = CodeGenerator()
    
    # Generate code with syntax error
    bad_code = '''
def broken_function(
    print("Missing closing parenthesis"
'''
    
    result = await generator.generate_custom(bad_code)
    
    # Should fail validation
    assert result.status == GenerationStatus.VALIDATION_FAILED
    assert result.validation_errors is not None
    
    print(f"✓ Error handling working")
    print(f"  Syntax error caught: {len(result.validation_errors)} errors")
    
    # Generate code with dangerous import
    dangerous_code = '''
import os
os.system("echo 'This should be blocked'")
'''
    
    result = await generator.generate_custom(dangerous_code)
    
    if result.status == GenerationStatus.SUCCESS:
        # Try to execute - should be blocked by sandbox
        from legionhercules.codegen.sandbox import CodeSandbox
        sandbox = CodeSandbox()
        exec_result = await sandbox.execute(result.code)
        assert exec_result.status == SandboxStatus.SECURITY_VIOLATION
        print(f"  Dangerous code blocked by sandbox")
        sandbox.cleanup()
    else:
        print(f"  Dangerous code blocked by validation")
    
    await generator.cleanup()
    return True


async def test_research_config_variations():
    """Test research with different configurations."""
    print("\n[Test] Research config variations...")
    
    engine = ResearchEngine()
    
    mock_sources = ["https://docs.python.org/3/"]
    
    # Test with different strategies
    strategies = [SummaryStrategy.EXTRACTIVE, SummaryStrategy.ABSTRACTIVE]
    
    for strategy in strategies:
        config = ResearchConfig(
            max_sources=1,
            summary_strategy=strategy,
            verify_sources=False,  # Skip verification for speed
        )
        
        report = await engine.research(
            query="Python documentation",
            sources=mock_sources,
            config=config,
            use_cache=False,
        )
        
        assert report.stage == ResearchStage.COMPLETE
        print(f"  ✓ Strategy {strategy.value}: {len(report.sources)} sources")
    
    await engine.close()
    return True


async def test_report_generation():
    """Test research report generation and export."""
    print("\n[Test] Research report generation...")
    
    engine = ResearchEngine()
    
    mock_sources = ["https://docs.python.org/3/"]
    
    config = ResearchConfig(
        max_sources=1,
        verify_sources=True,
        extract_code=True,
    )
    
    report = await engine.research(
        query="Python tutorial",
        sources=mock_sources,
        config=config,
        use_cache=False,
    )
    
    # Export to dict
    report_dict = report.to_dict()
    
    assert "query" in report_dict
    assert "sources" in report_dict
    assert "synthesis" in report_dict
    assert "credibility_summary" in report_dict
    assert "code_examples" in report_dict
    
    print(f"✓ Report generation working")
    print(f"  Keys: {list(report_dict.keys())}")
    print(f"  Avg credibility: {report_dict.get('avg_credibility', 'N/A')}")
    
    await engine.close()
    return True


async def run_all_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("LEGIONHERCULES Integration Test Suite")
    print("=" * 60)
    
    tests = [
        ("Sandbox → CodeGen Pipeline", test_sandbox_to_codegen_pipeline),
        ("Research Pipeline", test_research_pipeline),
        ("Cross-Module Integration", test_cross_module_integration),
        ("Error Handling", test_error_handling_integration),
        ("Research Config Variations", test_research_config_variations),
        ("Report Generation", test_report_generation),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            success = await test_func()
            if success:
                passed += 1
            else:
                failed += 1
                print(f"✗ {name} returned False")
        except Exception as e:
            failed += 1
            print(f"✗ {name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"Integration Tests: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
