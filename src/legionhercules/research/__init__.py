"""Research module for LEGIONHERCULES."""

from legionhercules.research.scraper import (
    WebScraper,
    ScrapedContent,
    ContentType,
)
from legionhercules.research.summarizer import (
    SummarizerEngine,
    SummaryResult,
    SummaryStrategy,
    ExtractiveSummarizer,
    AbstractiveSummarizer,
    HybridSummarizer,
)
from legionhercules.research.verifier import (
    SourceVerifier,
    FactChecker,
    VerificationResult,
    CredibilityTier,
    SourceType,
)
from legionhercules.research.engine import (
    ResearchEngine,
    ResearchPipeline,
    ResearchConfig,
    ResearchReport,
    ResearchStage,
    EnrichedSource,
)

__all__ = [
    # Scraper
    "WebScraper",
    "ScrapedContent",
    "ContentType",
    # Summarizer
    "SummarizerEngine",
    "SummaryResult",
    "SummaryStrategy",
    "ExtractiveSummarizer",
    "AbstractiveSummarizer",
    "HybridSummarizer",
    # Verifier
    "SourceVerifier",
    "FactChecker",
    "VerificationResult",
    "CredibilityTier",
    "SourceType",
    # Engine
    "ResearchEngine",
    "ResearchPipeline",
    "ResearchConfig",
    "ResearchReport",
    "ResearchStage",
    "EnrichedSource",
]
