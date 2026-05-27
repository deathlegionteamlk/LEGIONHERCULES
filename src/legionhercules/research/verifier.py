"""Source verification and credibility scoring module."""

from __future__ import annotations

import re
import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict, Set
from datetime import datetime, timedelta
from enum import Enum
from urllib.parse import urlparse

from legionhercules.utils.logging import get_logger

logger = get_logger(__name__)


class CredibilityTier(Enum):
    """Credibility tiers for sources."""
    HIGH = "high"           # Academic, government, established institutions
    MEDIUM_HIGH = "medium_high"  # Major news outlets, reputable blogs
    MEDIUM = "medium"       # General websites with good reputation
    MEDIUM_LOW = "medium_low"    # Newer sites, less established
    LOW = "low"             # Unknown, suspicious, or flagged sources
    UNVERIFIED = "unverified"    # Could not verify


class SourceType(Enum):
    """Types of sources."""
    ACADEMIC = "academic"
    GOVERNMENT = "government"
    NEWS = "news"
    BLOG = "blog"
    CORPORATE = "corporate"
    WIKI = "wiki"
    SOCIAL = "social"
    FORUM = "forum"
    UNKNOWN = "unknown"


@dataclass
class VerificationResult:
    """Result of source verification."""
    url: str
    is_verified: bool
    credibility_score: float  # 0.0 to 1.0
    credibility_tier: CredibilityTier
    source_type: SourceType
    domain_age_days: Optional[int] = None
    has_ssl: bool = False
    is_blacklisted: bool = False
    cross_references: List[str] = field(default_factory=list)
    red_flags: List[str] = field(default_factory=list)
    verification_date: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "is_verified": self.is_verified,
            "credibility_score": self.credibility_score,
            "credibility_tier": self.credibility_tier.value,
            "source_type": self.source_type.value,
            "domain_age_days": self.domain_age_days,
            "has_ssl": self.has_ssl,
            "is_blacklisted": self.is_blacklisted,
            "cross_references": self.cross_references,
            "red_flags": self.red_flags,
            "verification_date": self.verification_date.isoformat(),
            "metadata": self.metadata,
        }


class SourceVerifier:
    """Verifies sources and assesses credibility."""

    # High credibility domains
    HIGH_CREDIBILITY_DOMAINS = {
        # Academic
        'edu', 'ac.uk', 'ac.jp', 'ac.au', 'uni-', 'university',
        'arxiv.org', 'scholar.google', 'ieee.org', 'acm.org',
        'nature.com', 'science.org', 'cell.com', 'pnas.org',
        'jstor.org', 'pubmed.ncbi.nlm.nih.gov',

        # Government
        'gov', 'gov.uk', 'gov.au', 'europa.eu', 'un.org',
        'who.int', 'cdc.gov', 'fda.gov', 'nih.gov', 'nasa.gov',

        # Major institutions
        'mit.edu', 'stanford.edu', 'harvard.edu', 'ox.ac.uk',
        'cam.ac.uk', 'berkeley.edu', 'caltech.edu',
    }

    # Medium-high credibility news sources
    REPUTABLE_NEWS_DOMAINS = {
        'reuters.com', 'apnews.com', 'bloomberg.com', 'ft.com',
        'wsj.com', 'nytimes.com', 'washingtonpost.com', 'theguardian.com',
        'bbc.com', 'bbc.co.uk', 'economist.com', 'npr.org',
        'pbs.org', 'cbsnews.com', 'nbcnews.com', 'abcnews.go.com',
        'cnn.com', 'aljazeera.com', 'dw.com', 'france24.com',
        'time.com', 'newsweek.com', 'usatoday.com', 'latimes.com',
    }

    # Known low credibility / suspicious patterns
    SUSPICIOUS_PATTERNS = {
        'clickbait', 'fake', 'scam', 'spam',
        'conspiracy', 'hoax', 'misleading',
    }

    # Blacklisted domains (example)
    BLACKLISTED_DOMAINS = {
        'example-fake-news.com',
        'known-hoax-site.org',
    }

    def __init__(self):
        self.verification_cache: Dict[str, VerificationResult] = {}
        self.cache_ttl = timedelta(hours=24)

    async def verify(self, url: str, check_cross_references: bool = False) -> VerificationResult:
        """Verify a source URL."""
        # Check cache
        if url in self.verification_cache:
            cached = self.verification_cache[url]
            if datetime.now() - cached.verification_date < self.cache_ttl:
                return cached

        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # Initialize result
        result = VerificationResult(
            url=url,
            is_verified=False,
            credibility_score=0.5,
            credibility_tier=CredibilityTier.UNVERIFIED,
            source_type=SourceType.UNKNOWN,
            has_ssl=parsed.scheme == 'https',
        )

        # Check blacklist
        if self._is_blacklisted(domain):
            result.is_blacklisted = True
            result.credibility_score = 0.0
            result.credibility_tier = CredibilityTier.LOW
            result.red_flags.append("Domain is blacklisted")
            self.verification_cache[url] = result
            return result

        # Determine source type
        result.source_type = self._determine_source_type(domain, url)

        # Calculate credibility
        result.credibility_score = self._calculate_credibility_score(domain, result)
        result.credibility_tier = self._score_to_tier(result.credibility_score)

        # Check for red flags
        result.red_flags = self._check_red_flags(url, domain)

        # Cross-reference check
        if check_cross_references:
            result.cross_references = await self._check_cross_references(url)

        result.is_verified = True
        self.verification_cache[url] = result

        return result

    def _is_blacklisted(self, domain: str) -> bool:
        """Check if domain is blacklisted."""
        return domain in self.BLACKLISTED_DOMAINS

    def _determine_source_type(self, domain: str, url: str) -> SourceType:
        """Determine the type of source."""
        # Normalize domain by removing www. prefix for matching
        normalized_domain = domain.replace('www.', '')
        
        # Academic
        if any(indicator in domain for indicator in ['.edu', 'ac.uk', 'ac.jp', 'arxiv']):
            return SourceType.ACADEMIC

        # Government
        if any(indicator in domain for indicator in ['.gov', 'europa.eu', 'un.org']):
            return SourceType.GOVERNMENT

        # Wiki
        if 'wikipedia.org' in domain or 'wikimedia' in domain:
            return SourceType.WIKI

        # News
        if any(indicator in domain for indicator in [
            'news', 'times', 'post', 'herald', 'tribune',
            'gazette', 'chronicle', 'daily'
        ]) or normalized_domain in self.REPUTABLE_NEWS_DOMAINS:
            return SourceType.NEWS

        # Corporate
        if any(indicator in domain for indicator in [
            'corp', 'inc', 'ltd', 'company', 'enterprise'
        ]):
            return SourceType.CORPORATE

        # Blog
        if any(indicator in domain for indicator in [
            'blog', 'medium.com', 'substack', 'wordpress'
        ]):
            return SourceType.BLOG

        # Social
        if any(indicator in domain for indicator in [
            'twitter.com', 'x.com', 'facebook.com', 'linkedin.com',
            'instagram.com', 'reddit.com', 'youtube.com'
        ]):
            return SourceType.SOCIAL

        # Forum
        if any(indicator in domain for indicator in [
            'forum', 'discuss', 'stackexchange', 'stackoverflow'
        ]):
            return SourceType.FORUM

        return SourceType.UNKNOWN

    def _calculate_credibility_score(self, domain: str, result: VerificationResult) -> float:
        """Calculate credibility score (0.0 to 1.0)."""
        score = 0.5  # Base score

        # High credibility domains
        if any(indicator in domain for indicator in self.HIGH_CREDIBILITY_DOMAINS):
            score += 0.4

        # Reputable news
        elif domain in self.REPUTABLE_NEWS_DOMAINS:
            score += 0.3

        # SSL certificate
        if result.has_ssl:
            score += 0.05

        # Source type adjustments
        type_adjustments = {
            SourceType.ACADEMIC: 0.15,
            SourceType.GOVERNMENT: 0.15,
            SourceType.WIKI: 0.05,
            SourceType.NEWS: 0.1,
            SourceType.CORPORATE: 0.05,
            SourceType.BLOG: 0.0,
            SourceType.FORUM: -0.05,
            SourceType.SOCIAL: -0.1,
            SourceType.UNKNOWN: -0.1,
        }
        score += type_adjustments.get(result.source_type, 0)

        # Domain age (if known)
        if result.domain_age_days:
            if result.domain_age_days > 365 * 10:  # 10+ years
                score += 0.1
            elif result.domain_age_days > 365 * 5:  # 5+ years
                score += 0.05
            elif result.domain_age_days < 365:  # Less than 1 year
                score -= 0.1

        return max(0.0, min(1.0, score))

    def _score_to_tier(self, score: float) -> CredibilityTier:
        """Convert score to credibility tier."""
        if score >= 0.9:
            return CredibilityTier.HIGH
        elif score >= 0.75:
            return CredibilityTier.MEDIUM_HIGH
        elif score >= 0.6:
            return CredibilityTier.MEDIUM
        elif score >= 0.4:
            return CredibilityTier.MEDIUM_LOW
        else:
            return CredibilityTier.LOW

    def _check_red_flags(self, url: str, domain: str) -> List[str]:
        """Check for red flags in URL."""
        flags = []

        # Suspicious patterns
        url_lower = url.lower()
        for pattern in self.SUSPICIOUS_PATTERNS:
            if pattern in url_lower:
                flags.append(f"Contains suspicious term: {pattern}")

        # No SSL
        if not url.startswith('https'):
            flags.append("No SSL certificate (HTTP)")

        # Excessive subdomains
        subdomain_count = len(domain.split('.')) - 2
        if subdomain_count > 3:
            flags.append(f"Excessive subdomains ({subdomain_count})")

        # Suspicious TLDs
        suspicious_tlds = {'.tk', '.ml', '.ga', '.cf', '.top', '.xyz'}
        if any(domain.endswith(tld) for tld in suspicious_tlds):
            flags.append("Suspicious top-level domain")

        # Numbers in domain (often spam)
        if re.search(r'\d{4,}', domain):
            flags.append("Contains excessive numbers")

        return flags

    async def _check_cross_references(self, url: str) -> List[str]:
        """Check for cross-references to this URL."""
        # Simplified implementation - in production would search for citations
        # For now, return empty list
        return []

    def verify_batch(
        self,
        urls: List[str],
        check_cross_references: bool = False,
    ) -> List[VerificationResult]:
        """Verify multiple URLs."""
        import asyncio
        return asyncio.run(self._verify_batch_async(urls, check_cross_references))

    async def _verify_batch_async(
        self,
        urls: List[str],
        check_cross_references: bool = False,
    ) -> List[VerificationResult]:
        """Async batch verification."""
        tasks = [self.verify(url, check_cross_references) for url in urls]
        return await asyncio.gather(*tasks)

    def get_credibility_report(
        self,
        results: List[VerificationResult],
    ) -> Dict[str, Any]:
        """Generate credibility report for multiple sources."""
        if not results:
            return {"error": "No sources to analyze"}

        scores = [r.credibility_score for r in results]
        tiers = [r.credibility_tier for r in results]

        return {
            "total_sources": len(results),
            "average_credibility": sum(scores) / len(scores),
            "high_credibility_count": sum(1 for t in tiers if t == CredibilityTier.HIGH),
            "medium_credibility_count": sum(1 for t in tiers if t in [
                CredibilityTier.MEDIUM_HIGH, CredibilityTier.MEDIUM
            ]),
            "low_credibility_count": sum(1 for t in tiers if t in [
                CredibilityTier.MEDIUM_LOW, CredibilityTier.LOW
            ]),
            "blacklisted_count": sum(1 for r in results if r.is_blacklisted),
            "ssl_enabled_count": sum(1 for r in results if r.has_ssl),
            "sources_by_type": self._count_by_type(results),
            "red_flags_total": sum(len(r.red_flags) for r in results),
        }

    def _count_by_type(self, results: List[VerificationResult]) -> Dict[str, int]:
        """Count sources by type."""
        counts = {}
        for result in results:
            type_name = result.source_type.value
            counts[type_name] = counts.get(type_name, 0) + 1
        return counts


class FactChecker:
    """Basic fact checking capabilities."""

    def __init__(self):
        self.claims_database: Dict[str, Any] = {}

    def check_claim(self, claim: str, sources: List[str]) -> Dict[str, Any]:
        """Check a claim against sources."""
        # Simplified implementation
        # In production, would use NLP to extract claims and verify

        result = {
            "claim": claim,
            "verified": False,
            "confidence": 0.0,
            "supporting_sources": [],
            "contradicting_sources": [],
            "notes": [],
        }

        # Check for obvious falsehoods
        false_patterns = [
            r'\balways\b',  # Absolute statements
            r'\bnever\b',
            r'\b100%\b',   # Unlikely precision
            r'\beveryone knows\b',  # Appeal to common knowledge
        ]

        for pattern in false_patterns:
            if re.search(pattern, claim, re.IGNORECASE):
                result["notes"].append(f"Contains absolute language: {pattern}")
                result["confidence"] -= 0.1

        # Check source support
        if sources:
            result["supporting_sources"] = sources[:3]
            result["confidence"] += min(len(sources) * 0.1, 0.3)

        result["verified"] = result["confidence"] > 0.5
        return result

    def extract_claims(self, text: str) -> List[str]:
        """Extract verifiable claims from text."""
        claims = []

        # Look for sentences with numbers/statistics
        stat_pattern = r'[^.!?]*\b\d+(?:\.\d+)?%?\b[^.!?]*[.!?]'
        stat_claims = re.findall(stat_pattern, text)
        claims.extend(stat_claims)

        # Look for comparative statements
        comp_pattern = r'[^.!?]*\b(?:more than|less than|better than|worse than)\b[^.!?]*[.!?]'
        comp_claims = re.findall(comp_pattern, text, re.IGNORECASE)
        claims.extend(comp_claims)

        return list(set(claims))[:10]  # Remove duplicates, limit to 10
