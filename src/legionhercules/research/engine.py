"""Enhanced Research Engine with full pipeline integration."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict, Callable
from datetime import datetime
from enum import Enum

from legionhercules.research.scraper import WebScraper, ScrapedContent, ContentType
from legionhercules.research.summarizer import (
    SummarizerEngine,
    SummaryResult,
    SummaryStrategy,
)
from legionhercules.research.verifier import (
    SourceVerifier,
    VerificationResult,
    CredibilityTier,
    SourceType,
)
from legionhercules.utils.logging import get_logger

logger = get_logger(__name__)


class ResearchStage(Enum):
    """Stages of the research pipeline."""
    SEARCH = "search"
    SCRAPE = "scrape"
    VERIFY = "verify"
    SUMMARIZE = "summarize"
    SYNTHESIZE = "synthesize"
    COMPLETE = "complete"


@dataclass
class ResearchConfig:
    """Configuration for research operations."""
    max_sources: int = 5
    min_credibility_score: float = 0.3
    summary_strategy: SummaryStrategy = SummaryStrategy.HYBRID
    verify_sources: bool = True
    extract_code: bool = True
    timeout_per_source: int = 30
    parallel_verification: bool = True
    cache_results: bool = True


@dataclass
class EnrichedSource:
    """Source with scraped content, summary, and verification."""
    url: str
    scraped_content: Optional[ScrapedContent] = None
    summary: Optional[SummaryResult] = None
    verification: Optional[VerificationResult] = None
    credibility_score: float = 0.0
    relevance_score: float = 0.0
    extracted_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "url": self.url,
            "credibility_score": self.credibility_score,
            "relevance_score": self.relevance_score,
            "scraped": self.scraped_content.to_dict() if self.scraped_content else None,
            "summary": self.summary.to_dict() if self.summary else None,
            "verification": self.verification.to_dict() if self.verification else None,
            "extracted_at": self.extracted_at.isoformat(),
        }


@dataclass
class ResearchReport:
    """Complete research report with all findings."""
    query: str
    config: ResearchConfig
    sources: List[EnrichedSource] = field(default_factory=list)
    synthesis: str = ""
    key_findings: List[str] = field(default_factory=list)
    code_examples: List[Dict[str, Any]] = field(default_factory=list)
    credibility_summary: Dict[str, Any] = field(default_factory=dict)
    stage: ResearchStage = ResearchStage.SEARCH
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "config": {
                "max_sources": self.config.max_sources,
                "min_credibility_score": self.config.min_credibility_score,
                "summary_strategy": self.config.summary_strategy.value,
                "verify_sources": self.config.verify_sources,
            },
            "sources": [s.to_dict() for s in self.sources],
            "synthesis": self.synthesis,
            "key_findings": self.key_findings,
            "code_examples": self.code_examples,
            "credibility_summary": self.credibility_summary,
            "stage": self.stage.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "source_count": len(self.sources),
            "avg_credibility": sum(s.credibility_score for s in self.sources) / len(self.sources) if self.sources else 0,
        }


class ResearchPipeline:
    """Pipeline for orchestrating research operations."""
    
    def __init__(
        self,
        scraper: Optional[WebScraper] = None,
        summarizer: Optional[SummarizerEngine] = None,
        verifier: Optional[SourceVerifier] = None,
    ):
        self.scraper = scraper or WebScraper()
        self.summarizer = summarizer or SummarizerEngine()
        self.verifier = verifier or SourceVerifier()
        self._progress_callbacks: List[Callable[[ResearchStage, str], None]] = []
    
    def on_progress(self, callback: Callable[[ResearchStage, str], None]) -> None:
        """Register a progress callback."""
        self._progress_callbacks.append(callback)
    
    def _notify_progress(self, stage: ResearchStage, message: str) -> None:
        """Notify all progress callbacks."""
        for callback in self._progress_callbacks:
            try:
                callback(stage, message)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
    
    async def execute(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        config: Optional[ResearchConfig] = None,
    ) -> ResearchReport:
        """Execute the full research pipeline."""
        config = config or ResearchConfig()
        report = ResearchReport(query=query, config=config)
        
        try:
            # Stage 1: Search for sources if not provided
            if not sources:
                self._notify_progress(ResearchStage.SEARCH, f"Searching for sources: {query}")
                sources = await self._search_sources(query, config.max_sources * 2)
            
            if not sources:
                logger.warning("No sources found for query")
                report.stage = ResearchStage.COMPLETE
                report.completed_at = datetime.now()
                return report
            
            # Stage 2: Scrape content
            self._notify_progress(ResearchStage.SCRAPE, f"Scraping {len(sources)} sources")
            enriched_sources = await self._scrape_sources(sources, config)
            
            # Stage 3: Verify sources (if enabled)
            if config.verify_sources:
                self._notify_progress(ResearchStage.VERIFY, "Verifying source credibility")
                enriched_sources = await self._verify_sources(enriched_sources, config)
                
                # Filter by credibility
                enriched_sources = [
                    s for s in enriched_sources
                    if s.credibility_score >= config.min_credibility_score
                ]
            
            # Stage 4: Summarize content
            self._notify_progress(ResearchStage.SUMMARIZE, "Summarizing content")
            enriched_sources = await self._summarize_sources(enriched_sources, config)
            
            # Stage 5: Synthesize findings
            self._notify_progress(ResearchStage.SYNTHESIZE, "Synthesizing findings")
            report.sources = enriched_sources[:config.max_sources]
            report.synthesis = self._synthesize_findings(query, report.sources)
            report.key_findings = self._extract_key_findings(report.sources)
            report.credibility_summary = self._summarize_credibility(report.sources)
            
            # Extract code examples if enabled
            if config.extract_code:
                report.code_examples = self._extract_code_examples(report.sources)
            
            report.stage = ResearchStage.COMPLETE
            report.completed_at = datetime.now()
            
            self._notify_progress(ResearchStage.COMPLETE, f"Research complete: {len(report.sources)} sources")
            
        except Exception as e:
            logger.error(f"Research pipeline failed: {e}")
            report.synthesis = f"Research failed: {str(e)}"
        
        return report
    
    async def _search_sources(self, query: str, max_results: int) -> List[str]:
        """Search for relevant sources."""
        try:
            from duckduckgo_search import DDGS
            
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                return [r["href"] for r in results if "href" in r][:max_results]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def _scrape_sources(
        self,
        urls: List[str],
        config: ResearchConfig,
    ) -> List[EnrichedSource]:
        """Scrape content from sources."""
        sources = []
        
        for url in urls:
            try:
                content = await self.scraper.scrape(url)
                if content:
                    sources.append(EnrichedSource(
                        url=url,
                        scraped_content=content,
                    ))
            except Exception as e:
                logger.warning(f"Failed to scrape {url}: {e}")
        
        return sources
    
    async def _verify_sources(
        self,
        sources: List[EnrichedSource],
        config: ResearchConfig,
    ) -> List[EnrichedSource]:
        """Verify and score source credibility."""
        if config.parallel_verification:
            # Verify in parallel
            tasks = [self._verify_single_source(s) for s in sources]
            return await asyncio.gather(*tasks)
        else:
            # Verify sequentially
            results = []
            for source in sources:
                result = await self._verify_single_source(source)
                results.append(result)
            return results
    
    async def _verify_single_source(self, source: EnrichedSource) -> EnrichedSource:
        """Verify a single source."""
        try:
            verification = await self.verifier.verify(source.url)
            source.verification = verification
            source.credibility_score = verification.credibility_score
        except Exception as e:
            logger.warning(f"Verification failed for {source.url}: {e}")
            source.credibility_score = 0.0
        
        return source
    
    async def _summarize_sources(
        self,
        sources: List[EnrichedSource],
        config: ResearchConfig,
    ) -> List[EnrichedSource]:
        """Summarize content from sources."""
        for source in sources:
            if source.scraped_content and source.scraped_content.content:
                try:
                    summary = await self.summarizer.summarize(
                        source.scraped_content.content,
                        strategy=config.summary_strategy,
                    )
                    source.summary = summary
                except Exception as e:
                    logger.warning(f"Summarization failed for {source.url}: {e}")
        
        return sources
    
    def _synthesize_findings(self, query: str, sources: List[EnrichedSource]) -> str:
        """Synthesize findings from all sources."""
        if not sources:
            return "No relevant sources found."
        
        parts = []
        parts.append(f"# Research Report: {query}\n")
        parts.append(f"**Sources analyzed:** {len(sources)}\n")
        
        # Credibility overview
        avg_credibility = sum(s.credibility_score for s in sources) / len(sources)
        parts.append(f"**Average credibility:** {avg_credibility:.2f}/1.0\n")
        
        # Source summaries
        parts.append("\n## Source Summaries\n")
        for i, source in enumerate(sources, 1):
            parts.append(f"\n### {i}. {source.scraped_content.title if source.scraped_content else 'Unknown'}")
            parts.append(f"**URL:** {source.url}")
            parts.append(f"**Credibility:** {source.credibility_score:.2f}")
            
            if source.summary:
                parts.append(f"**Summary:** {source.summary.summary}")
            elif source.scraped_content:
                parts.append(f"**Summary:** {source.scraped_content.summary[:300]}...")
            
            if source.verification:
                parts.append(f"**Source type:** {source.verification.source_type.value}")
        
        # Key findings
        parts.append("\n## Key Findings\n")
        findings = self._extract_key_findings(sources)
        for finding in findings[:10]:
            parts.append(f"- {finding}")
        
        return '\n'.join(parts)
    
    def _extract_key_findings(self, sources: List[EnrichedSource]) -> List[str]:
        """Extract key findings from sources."""
        findings = []
        
        for source in sources:
            if source.summary and source.summary.key_points:
                findings.extend(source.summary.key_points)
            elif source.scraped_content and source.scraped_content.summary:
                # Extract first sentence as key finding
                first_sentence = source.scraped_content.summary.split('.')[0]
                if len(first_sentence) > 20:
                    findings.append(first_sentence)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_findings = []
        for f in findings:
            if f not in seen:
                seen.add(f)
                unique_findings.append(f)
        
        return unique_findings
    
    def _summarize_credibility(self, sources: List[EnrichedSource]) -> Dict[str, Any]:
        """Summarize credibility metrics."""
        if not sources:
            return {"status": "no_sources"}
        
        scores = [s.credibility_score for s in sources]
        verifications = [s for s in sources if s.verification]
        
        return {
            "average_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "high_credibility_count": sum(1 for s in scores if s >= 0.7),
            "medium_credibility_count": sum(1 for s in scores if 0.4 <= s < 0.7),
            "low_credibility_count": sum(1 for s in scores if s < 0.4),
            "verified_count": len(verifications),
            "ssl_enabled_count": sum(1 for s in verifications if s.verification.has_ssl),
        }
    
    def _extract_code_examples(self, sources: List[EnrichedSource]) -> List[Dict[str, Any]]:
        """Extract code examples from sources."""
        examples = []
        
        for source in sources:
            if source.scraped_content and source.scraped_content.code_blocks:
                for code in source.scraped_content.code_blocks[:5]:  # Limit per source
                    examples.append({
                        "code": code[:500],  # Truncate long code
                        "source": source.url,
                        "title": source.scraped_content.title,
                    })
        
        return examples
    
    async def close(self) -> None:
        """Clean up resources."""
        await self.scraper.close()


class ResearchEngine:
    """High-level research engine with caching and history."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.pipeline = ResearchPipeline()
        self.cache: Dict[str, ResearchReport] = {}
        self.history: List[ResearchReport] = []
        self.cache_dir = cache_dir
    
    async def research(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        config: Optional[ResearchConfig] = None,
        use_cache: bool = True,
    ) -> ResearchReport:
        """Conduct research with caching."""
        config = config or ResearchConfig()
        cache_key = f"{query}:{hash(tuple(sources or []))}:{config.max_sources}"
        
        # Check cache
        if use_cache and cache_key in self.cache:
            logger.info(f"Returning cached research for: {query}")
            return self.cache[cache_key]
        
        # Execute pipeline
        report = await self.pipeline.execute(query, sources, config)
        
        # Cache and store
        if use_cache:
            self.cache[cache_key] = report
        self.history.append(report)
        
        return report
    
    def get_history(self) -> List[ResearchReport]:
        """Get research history."""
        return self.history.copy()
    
    def clear_cache(self) -> None:
        """Clear research cache."""
        self.cache.clear()
    
    async def close(self) -> None:
        """Clean up resources."""
        await self.pipeline.close()
