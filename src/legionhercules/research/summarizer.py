"""Content summarization module with multiple strategies."""

from __future__ import annotations

import re
import heapq
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict, Callable
from enum import Enum
from collections import Counter

from legionhercules.utils.logging import get_logger

logger = get_logger(__name__)


class SummaryStrategy(Enum):
    """Summarization strategies."""
    EXTRACTIVE = "extractive"
    ABSTRACTIVE = "abstractive"
    HYBRID = "hybrid"
    KEY_SENTENCES = "key_sentences"
    TF_IDF = "tf_idf"


@dataclass
class SummaryResult:
    """Result of summarization."""
    summary: str
    strategy: SummaryStrategy
    original_length: int
    summary_length: int
    compression_ratio: float
    key_points: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "strategy": self.strategy.value,
            "original_length": self.original_length,
            "summary_length": self.summary_length,
            "compression_ratio": self.compression_ratio,
            "key_points": self.key_points,
            "metadata": self.metadata,
        }


class TextPreprocessor:
    """Preprocess text for summarization."""

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^\w\s.!?;:]', '', text)
        return text.strip()

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def split_paragraphs(text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = text.split('\n\n')
        return [p.strip() for p in paragraphs if p.strip()]


class ExtractiveSummarizer:
    """Extractive summarization using various algorithms."""

    def __init__(self):
        self.preprocessor = TextPreprocessor()

    def summarize(
        self,
        text: str,
        num_sentences: int = 3,
        algorithm: str = "frequency",
    ) -> SummaryResult:
        """Generate extractive summary."""
        cleaned_text = self.preprocessor.clean_text(text)
        sentences = self.preprocessor.split_sentences(cleaned_text)

        if len(sentences) <= num_sentences:
            return SummaryResult(
                summary=cleaned_text,
                strategy=SummaryStrategy.EXTRACTIVE,
                original_length=len(text),
                summary_length=len(cleaned_text),
                compression_ratio=1.0,
                key_points=sentences[:5],
            )

        if algorithm == "frequency":
            scored_sentences = self._score_by_frequency(sentences)
        elif algorithm == "position":
            scored_sentences = self._score_by_position(sentences)
        elif algorithm == "length":
            scored_sentences = self._score_by_length(sentences)
        else:
            scored_sentences = self._score_by_frequency(sentences)

        # Select top sentences maintaining original order
        top_sentences = heapq.nlargest(num_sentences, scored_sentences, key=lambda x: x[1])
        top_sentences.sort(key=lambda x: x[0])  # Sort by original index

        summary_sentences = [sentences[i] for i, _ in top_sentences]
        summary = ' '.join(summary_sentences)

        return SummaryResult(
            summary=summary,
            strategy=SummaryStrategy.EXTRACTIVE,
            original_length=len(text),
            summary_length=len(summary),
            compression_ratio=len(summary) / len(text) if text else 1.0,
            key_points=summary_sentences,
            metadata={"algorithm": algorithm, "sentences_selected": len(top_sentences)},
        )

    def _score_by_frequency(self, sentences: List[str]) -> List[tuple[int, float]]:
        """Score sentences by word frequency."""
        # Count word frequencies (excluding stop words)
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
            'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'under', 'and', 'but', 'or', 'yet', 'so',
            'if', 'because', 'although', 'though', 'while', 'where',
        }

        word_freq = Counter()
        for sentence in sentences:
            words = re.findall(r'\b\w+\b', sentence.lower())
            for word in words:
                if word not in stop_words and len(word) > 2:
                    word_freq[word] += 1

        # Score each sentence
        scored = []
        for i, sentence in enumerate(sentences):
            words = re.findall(r'\b\w+\b', sentence.lower())
            score = sum(word_freq.get(word, 0) for word in words if word not in stop_words)
            # Normalize by sentence length
            score = score / max(len(words), 1)
            scored.append((i, score))

        return scored

    def _score_by_position(self, sentences: List[str]) -> List[tuple[int, float]]:
        """Score sentences by position (earlier sentences are more important)."""
        scored = []
        total = len(sentences)
        for i, _ in enumerate(sentences):
            # Exponential decay based on position
            score = 1.0 / (1 + i * 0.5)
            scored.append((i, score))
        return scored

    def _score_by_length(self, sentences: List[str]) -> List[tuple[int, float]]:
        """Score sentences by length (prefer medium-length sentences)."""
        scored = []
        lengths = [len(s) for s in sentences]
        avg_length = sum(lengths) / len(lengths) if lengths else 1

        for i, sentence in enumerate(sentences):
            length = len(sentence)
            # Prefer sentences close to average length
            score = 1.0 / (1 + abs(length - avg_length) / avg_length)
            scored.append((i, score))

        return scored


class AbstractiveSummarizer:
    """Abstractive summarization (simplified version without external LLM)."""

    def __init__(self):
        self.preprocessor = TextPreprocessor()

    def summarize(
        self,
        text: str,
        max_length: int = 200,
    ) -> SummaryResult:
        """Generate abstractive-like summary."""
        cleaned_text = self.preprocessor.clean_text(text)
        sentences = self.preprocessor.split_sentences(cleaned_text)

        if not sentences:
            return SummaryResult(
                summary="",
                strategy=SummaryStrategy.ABSTRACTIVE,
                original_length=len(text),
                summary_length=0,
                compression_ratio=0.0,
            )

        # Extract key phrases and combine
        key_phrases = self._extract_key_phrases(sentences)
        key_points = self._generate_key_points(sentences)

        # Create summary by combining key phrases and first sentences
        summary_parts = []

        # Add first sentence (usually contains main topic)
        if sentences:
            summary_parts.append(sentences[0])

        # Add key phrases
        if key_phrases:
            summary_parts.append("Key aspects: " + ", ".join(key_phrases[:5]))

        # Add concluding sentence if available
        if len(sentences) > 1:
            summary_parts.append(sentences[-1])

        summary = ' '.join(summary_parts)

        # Truncate if too long
        if len(summary) > max_length:
            summary = summary[:max_length].rsplit(' ', 1)[0] + '...'

        return SummaryResult(
            summary=summary,
            strategy=SummaryStrategy.ABSTRACTIVE,
            original_length=len(text),
            summary_length=len(summary),
            compression_ratio=len(summary) / len(text) if text else 1.0,
            key_points=key_points,
        )

    def _extract_key_phrases(self, sentences: List[str]) -> List[str]:
        """Extract key noun phrases from sentences."""
        phrases = []
        # Simple noun phrase extraction (adjective + noun patterns)
        for sentence in sentences:
            # Look for capitalized phrases
            matches = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', sentence)
            phrases.extend(matches)

            # Look for quoted phrases
            quoted = re.findall(r'"([^"]+)"', sentence)
            phrases.extend(quoted)

        # Remove duplicates and short phrases
        phrases = list(set(p for p in phrases if len(p) > 3))
        return phrases[:10]

    def _generate_key_points(self, sentences: List[str]) -> List[str]:
        """Generate key points from sentences."""
        key_points = []

        for sentence in sentences[:5]:  # Limit to first 5 sentences
            # Look for sentences with important indicators
            if any(indicator in sentence.lower() for indicator in [
                'important', 'key', 'main', 'primary', 'essential',
                'significant', 'critical', 'crucial', 'vital'
            ]):
                key_points.append(sentence)

        return key_points[:5]


class HybridSummarizer:
    """Hybrid summarization combining extractive and abstractive approaches."""

    def __init__(self):
        self.extractive = ExtractiveSummarizer()
        self.abstractive = AbstractiveSummarizer()

    def summarize(
        self,
        text: str,
        num_sentences: int = 3,
        max_length: int = 300,
    ) -> SummaryResult:
        """Generate hybrid summary."""
        # First, get extractive summary
        extractive_result = self.extractive.summarize(text, num_sentences)

        # Then, apply abstractive techniques to the extractive summary
        abstractive_result = self.abstractive.summarize(
            extractive_result.summary,
            max_length=max_length
        )

        # Combine results
        combined_summary = self._combine_summaries(
            extractive_result.summary,
            abstractive_result.summary
        )

        return SummaryResult(
            summary=combined_summary,
            strategy=SummaryStrategy.HYBRID,
            original_length=len(text),
            summary_length=len(combined_summary),
            compression_ratio=len(combined_summary) / len(text) if text else 1.0,
            key_points=extractive_result.key_points + abstractive_result.key_points,
            metadata={
                "extractive_ratio": extractive_result.compression_ratio,
                "abstractive_ratio": abstractive_result.compression_ratio,
            },
        )

    def _combine_summaries(self, extractive: str, abstractive: str) -> str:
        """Combine extractive and abstractive summaries."""
        # Use extractive as base, enhance with abstractive insights
        if len(abstractive) < len(extractive) * 0.5:
            return extractive

        # Merge unique information from both
        extractive_sentences = set(TextPreprocessor.split_sentences(extractive))
        abstractive_sentences = set(TextPreprocessor.split_sentences(abstractive))

        # Combine and maintain order
        all_sentences = list(extractive_sentences | abstractive_sentences)
        return ' '.join(all_sentences[:5])  # Limit to 5 sentences


class SummarizerEngine:
    """Main summarization engine with multiple strategies."""

    def __init__(self):
        self.extractive = ExtractiveSummarizer()
        self.abstractive = AbstractiveSummarizer()
        self.hybrid = HybridSummarizer()

    def summarize(
        self,
        text: str,
        strategy: SummaryStrategy = SummaryStrategy.HYBRID,
        **kwargs,
    ) -> SummaryResult:
        """Summarize text using specified strategy."""
        if not text or not text.strip():
            return SummaryResult(
                summary="",
                strategy=strategy,
                original_length=0,
                summary_length=0,
                compression_ratio=0.0,
            )

        if strategy == SummaryStrategy.EXTRACTIVE:
            num_sentences = kwargs.get('num_sentences', 3)
            algorithm = kwargs.get('algorithm', 'frequency')
            return self.extractive.summarize(text, num_sentences, algorithm)

        elif strategy == SummaryStrategy.ABSTRACTIVE:
            max_length = kwargs.get('max_length', 200)
            return self.abstractive.summarize(text, max_length)

        elif strategy == SummaryStrategy.HYBRID:
            num_sentences = kwargs.get('num_sentences', 3)
            max_length = kwargs.get('max_length', 300)
            return self.hybrid.summarize(text, num_sentences, max_length)

        elif strategy == SummaryStrategy.KEY_SENTENCES:
            return self._key_sentences_summary(text, kwargs.get('num_points', 5))

        elif strategy == SummaryStrategy.TF_IDF:
            return self._tf_idf_summary(text, kwargs.get('num_sentences', 3))

        else:
            return self.hybrid.summarize(text)

    def _key_sentences_summary(
        self,
        text: str,
        num_points: int = 5,
    ) -> SummaryResult:
        """Generate summary with key sentences as bullet points."""
        result = self.extractive.summarize(text, num_sentences=num_points * 2)

        # Format as bullet points
        key_sentences = result.key_points[:num_points]
        bullet_summary = '\n'.join(f"• {s}" for s in key_sentences)

        return SummaryResult(
            summary=bullet_summary,
            strategy=SummaryStrategy.KEY_SENTENCES,
            original_length=len(text),
            summary_length=len(bullet_summary),
            compression_ratio=len(bullet_summary) / len(text) if text else 1.0,
            key_points=key_sentences,
        )

    def _tf_idf_summary(
        self,
        text: str,
        num_sentences: int = 3,
    ) -> SummaryResult:
        """Generate summary using TF-IDF scoring."""
        # Simplified TF-IDF (term frequency without IDF)
        sentences = TextPreprocessor.split_sentences(text)

        if len(sentences) <= num_sentences:
            return SummaryResult(
                summary=text,
                strategy=SummaryStrategy.TF_IDF,
                original_length=len(text),
                summary_length=len(text),
                compression_ratio=1.0,
                key_points=sentences,
            )

        # Calculate term frequencies for each sentence
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            words = re.findall(r'\b\w+\b', sentence.lower())
            # Score based on unique word count
            score = len(set(words)) / max(len(words), 1)
            sentence_scores.append((i, score))

        # Select top sentences
        top_sentences = heapq.nlargest(num_sentences, sentence_scores, key=lambda x: x[1])
        top_sentences.sort(key=lambda x: x[0])

        summary_sentences = [sentences[i] for i, _ in top_sentences]
        summary = ' '.join(summary_sentences)

        return SummaryResult(
            summary=summary,
            strategy=SummaryStrategy.TF_IDF,
            original_length=len(text),
            summary_length=len(summary),
            compression_ratio=len(summary) / len(text) if text else 1.0,
            key_points=summary_sentences,
            metadata={"scoring_method": "tf-idf-simplified"},
        )

    def batch_summarize(
        self,
        texts: List[str],
        strategy: SummaryStrategy = SummaryStrategy.HYBRID,
        **kwargs,
    ) -> List[SummaryResult]:
        """Summarize multiple texts."""
        return [self.summarize(text, strategy, **kwargs) for text in texts]

    def compare_strategies(
        self,
        text: str,
        **kwargs,
    ) -> Dict[str, SummaryResult]:
        """Compare different summarization strategies."""
        results = {}
        for strategy in SummaryStrategy:
            try:
                results[strategy.value] = self.summarize(text, strategy, **kwargs)
            except Exception as e:
                logger.error(f"Strategy {strategy.value} failed: {e}")
                results[strategy.value] = None
        return results
