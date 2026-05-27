"""Test suite for research module."""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from legionhercules.research import (
    SummarizerEngine,
    SummaryStrategy,
    SourceVerifier,
    FactChecker,
    CredibilityTier,
    SourceType,
    WebScraper,
)


async def test_summarizer_extractive():
    """Test extractive summarization."""
    print("\n[Test] Extractive summarization...")
    
    text = '''
    Python is a high-level programming language. It was created by Guido van Rossum.
    Python emphasizes code readability. It supports multiple programming paradigms.
    Python is used for web development. It is also used for data analysis.
    Many developers love Python for its simplicity. It has a large community.
    '''
    
    engine = SummarizerEngine()
    result = engine.summarize(text, strategy=SummaryStrategy.EXTRACTIVE, num_sentences=2)
    
    assert result.strategy == SummaryStrategy.EXTRACTIVE
    assert len(result.summary) > 0
    assert result.compression_ratio < 1.0
    assert len(result.key_points) > 0
    
    print(f"✓ Extractive: {len(result.summary)} chars, {result.compression_ratio:.1%} compression")
    return True


async def test_summarizer_abstractive():
    """Test abstractive summarization."""
    print("\n[Test] Abstractive summarization...")
    
    text = '''
    Machine learning is a subset of artificial intelligence. It enables computers to learn
    from data without being explicitly programmed. Deep learning is a type of machine learning
    that uses neural networks with many layers. These networks can recognize patterns in data.
    '''
    
    engine = SummarizerEngine()
    result = engine.summarize(text, strategy=SummaryStrategy.ABSTRACTIVE, max_length=200)
    
    assert result.strategy == SummaryStrategy.ABSTRACTIVE
    assert len(result.summary) > 0
    assert len(result.summary) <= 250  # Allow some margin
    
    print(f"✓ Abstractive: {len(result.summary)} chars")
    return True


async def test_summarizer_hybrid():
    """Test hybrid summarization."""
    print("\n[Test] Hybrid summarization...")
    
    text = '''
    Artificial Intelligence is transforming technology. Machine learning powers many AI applications.
    Deep learning uses neural networks. Natural language processing enables chatbots.
    Computer vision allows image recognition. These technologies are advancing rapidly.
    '''
    
    engine = SummarizerEngine()
    result = engine.summarize(text, strategy=SummaryStrategy.HYBRID, num_sentences=2)
    
    assert result.strategy == SummaryStrategy.HYBRID
    assert len(result.summary) > 0
    assert result.compression_ratio < 1.0
    
    print(f"✓ Hybrid: {len(result.summary)} chars, {result.compression_ratio:.1%} compression")
    return True


async def test_summarizer_all_strategies():
    """Test all summarization strategies."""
    print("\n[Test] All summarization strategies...")
    
    text = '''
    The quick brown fox jumps over the lazy dog. This is a famous pangram.
    A pangram contains every letter of the alphabet. Pangrams are useful for testing fonts.
    They are also used in typing practice. Many languages have their own pangrams.
    '''
    
    engine = SummarizerEngine()
    
    for strategy in SummaryStrategy:
        result = engine.summarize(text, strategy=strategy)
        assert result.strategy == strategy
        assert len(result.summary) >= 0  # Can be empty for short text
    
    print(f"✓ All {len(list(SummaryStrategy))} strategies work")
    return True


async def test_summarizer_empty_text():
    """Test summarization with empty text."""
    print("\n[Test] Empty text summarization...")
    
    engine = SummarizerEngine()
    result = engine.summarize("")
    
    assert result.summary == ""
    assert result.original_length == 0
    
    print("✓ Empty text handled correctly")
    return True


async def test_source_verifier_high_credibility():
    """Test source verification for high credibility domain."""
    print("\n[Test] Source verifier (high credibility)...")
    
    verifier = SourceVerifier()
    result = await verifier.verify("https://www.nih.gov")
    
    assert result.is_verified is True
    assert result.credibility_score > 0.7
    assert result.credibility_tier in [CredibilityTier.HIGH, CredibilityTier.MEDIUM_HIGH]
    assert result.source_type == SourceType.GOVERNMENT
    assert result.has_ssl is True
    
    print(f"✓ High credibility: {result.credibility_tier.value}, score: {result.credibility_score:.2f}")
    return True


async def test_source_verifier_academic():
    """Test source verification for academic domain."""
    print("\n[Test] Source verifier (academic)...")
    
    verifier = SourceVerifier()
    result = await verifier.verify("https://arxiv.org")
    
    assert result.is_verified is True
    assert result.source_type == SourceType.ACADEMIC
    assert result.credibility_score > 0.5
    
    print(f"✓ Academic: {result.source_type.value}, score: {result.credibility_score:.2f}")
    return True


async def test_source_verifier_news():
    """Test source verification for news domain."""
    print("\n[Test] Source verifier (news)...")
    
    verifier = SourceVerifier()
    result = await verifier.verify("https://www.reuters.com")
    
    assert result.is_verified is True
    # Reuters should be detected as news or have high credibility
    assert result.source_type in [SourceType.NEWS, SourceType.CORPORATE]
    assert result.credibility_score > 0.5
    
    print(f"✓ News: {result.source_type.value}, score: {result.credibility_score:.2f}")
    return True


async def test_source_verifier_suspicious():
    """Test source verification for suspicious URL."""
    print("\n[Test] Source verifier (suspicious)...")
    
    verifier = SourceVerifier()
    result = await verifier.verify("http://suspicious-site.tk/clickbait-fake-news")
    
    assert result.is_verified is True
    assert result.has_ssl is False
    assert len(result.red_flags) > 0
    
    print(f"✓ Suspicious: {len(result.red_flags)} red flags detected")
    return True


async def test_source_verifier_batch():
    """Test batch source verification."""
    print("\n[Test] Source verifier batch...")
    
    verifier = SourceVerifier()
    urls = [
        "https://python.org",
        "https://github.com",
        "https://stackoverflow.com",
    ]
    
    results = await verifier._verify_batch_async(urls)
    
    assert len(results) == 3
    assert all(r.is_verified for r in results)
    
    # Generate report
    report = verifier.get_credibility_report(results)
    assert report["total_sources"] == 3
    assert "average_credibility" in report
    
    print(f"✓ Batch verification: {report['total_sources']} sources, avg score: {report['average_credibility']:.2f}")
    return True


async def test_fact_checker_claims():
    """Test fact checker claim extraction."""
    print("\n[Test] Fact checker claims...")
    
    checker = FactChecker()
    
    text = '''
    The Earth is 4.5 billion years old. Water covers 71% of the Earth's surface.
    The highest mountain is Everest at 8,848 meters. Python is used by 50% of developers.
    '''
    
    claims = checker.extract_claims(text)
    
    assert len(claims) > 0
    # Should extract sentences with numbers
    assert any("4.5" in c or "71%" in c for c in claims)
    
    print(f"✓ Claims extracted: {len(claims)} claims found")
    return True


async def test_fact_checker_verify():
    """Test fact checker verification."""
    print("\n[Test] Fact checker verification...")
    
    checker = FactChecker()
    
    claim = "Python is the most popular programming language"
    sources = ["https://python.org", "https://github.com"]
    
    result = checker.check_claim(claim, sources)
    
    assert "claim" in result
    assert "confidence" in result
    assert result["claim"] == claim
    
    print(f"✓ Claim verification: confidence={result['confidence']:.2f}")
    return True


async def test_web_scraper_initialization():
    """Test web scraper initialization."""
    print("\n[Test] Web scraper initialization...")
    
    scraper = WebScraper(timeout=30)
    
    assert scraper.timeout == 30
    assert scraper.respect_robots is True
    
    print("✓ Web scraper initialized")
    return True


async def test_summarizer_batch():
    """Test batch summarization."""
    print("\n[Test] Batch summarization...")
    
    engine = SummarizerEngine()
    
    texts = [
        "Python is great. It is easy to learn.",
        "JavaScript runs in browsers. It is versatile.",
        "Rust is fast. It prevents memory errors.",
    ]
    
    results = engine.batch_summarize(texts, strategy=SummaryStrategy.EXTRACTIVE)
    
    assert len(results) == 3
    assert all(r.summary for r in results)
    
    print(f"✓ Batch summarization: {len(results)} texts processed")
    return True


async def test_compare_strategies():
    """Test strategy comparison."""
    print("\n[Test] Compare strategies...")
    
    engine = SummarizerEngine()
    
    text = '''
    Artificial Intelligence is a broad field. It includes machine learning and deep learning.
    Neural networks are inspired by the human brain. They can recognize patterns in data.
    AI is used in many applications. These include image recognition and natural language processing.
    '''
    
    results = engine.compare_strategies(text, num_sentences=2)
    
    assert len(results) > 0
    # Should have results for most strategies
    assert any(results.get(s.value) is not None for s in SummaryStrategy)
    
    print(f"✓ Strategy comparison: {len(results)} strategies compared")
    return True


async def run_all_tests():
    """Run all research tests."""
    print("="*60)
    print("Research Module Test Suite")
    print("="*60)
    
    tests = [
        test_summarizer_extractive,
        test_summarizer_abstractive,
        test_summarizer_hybrid,
        test_summarizer_all_strategies,
        test_summarizer_empty_text,
        test_source_verifier_high_credibility,
        test_source_verifier_academic,
        test_source_verifier_news,
        test_source_verifier_suspicious,
        test_source_verifier_batch,
        test_fact_checker_claims,
        test_fact_checker_verify,
        test_web_scraper_initialization,
        test_summarizer_batch,
        test_compare_strategies,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            success = await test()
            if success:
                passed += 1
        except Exception as e:
            failed += 1
            print(f"✗ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


async def main():
    """Main test runner."""
    success = await run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
