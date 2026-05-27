"""Web scraping and research module for LEGIONHERCULES."""

from __future__ import annotations

import re
import json
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict
from datetime import datetime
from urllib.parse import urljoin, urlparse
from enum import Enum

from legionhercules.utils.logging import get_logger

logger = get_logger(__name__)


class ContentType(Enum):
    """Type of content extracted."""
    ARTICLE = "article"
    DOCUMENTATION = "documentation"
    CODE = "code"
    API_REFERENCE = "api_reference"
    TUTORIAL = "tutorial"
    BLOG = "blog"
    UNKNOWN = "unknown"


@dataclass
class ScrapedContent:
    """Content extracted from a webpage."""
    url: str
    title: str = ""
    content: str = ""
    summary: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    content_type: ContentType = ContentType.UNKNOWN
    extracted_at: datetime = field(default_factory=datetime.now)
    links: List[str] = field(default_factory=list)
    headings: List[str] = field(default_factory=list)
    code_blocks: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "content": self.content[:5000] if len(self.content) > 5000 else self.content,
            "summary": self.summary,
            "metadata": self.metadata,
            "content_type": self.content_type.value,
            "extracted_at": self.extracted_at.isoformat(),
            "links": self.links[:20],  # Limit links
            "headings": self.headings[:10],
            "code_blocks_count": len(self.code_blocks),
        }


class WebScraper:
    """Scraper for extracting content from web pages."""

    def __init__(self, timeout: int = 30, respect_robots: bool = True):
        self.timeout = timeout
        self.respect_robots = respect_robots
        self._session = None

    async def _get_session(self):
        """Get or create HTTP session."""
        if self._session is None:
            import httpx
            self._session = httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
                headers={
                    "User-Agent": "LEGIONHERCULES/1.0 Research Bot",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                },
            )
        return self._session

    async def scrape(self, url: str) -> Optional[ScrapedContent]:
        """Scrape content from a URL."""
        logger.info(f"Scraping: {url}")
        
        try:
            session = await self._get_session()
            response = await session.get(url)
            response.raise_for_status()
            
            html = response.text
            return self._parse_html(url, html)
            
        except Exception as e:
            logger.error(f"Failed to scrape {url}: {e}")
            return None

    def _parse_html(self, url: str, html: str) -> ScrapedContent:
        """Parse HTML content."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.warning("BeautifulSoup not available, using basic parsing")
            return self._basic_parse(url, html)

        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract title
        title = ""
        if soup.title:
            title = soup.title.string.strip() if soup.title.string else ""
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Extract main content
        main_content = ""
        
        # Try to find main content area
        for selector in ["main", "article", ".content", "#content", ".main", "#main"]:
            element = soup.select_one(selector)
            if element:
                main_content = element.get_text(separator='\n', strip=True)
                break
        
        if not main_content:
            main_content = soup.get_text(separator='\n', strip=True)
        
        # Clean up whitespace
        main_content = re.sub(r'\n{3,}', '\n\n', main_content)
        
        # Extract headings
        headings = []
        for h in soup.find_all(['h1', 'h2', 'h3']):
            text = h.get_text(strip=True)
            if text:
                headings.append(text)
        
        # Extract links
        links = []
        base_url = urlparse(url)
        for a in soup.find_all('a', href=True):
            href = a['href']
            absolute_url = urljoin(url, href)
            parsed = urlparse(absolute_url)
            if parsed.netloc == base_url.netloc:
                links.append(absolute_url)
        
        # Extract code blocks
        code_blocks = []
        for code in soup.find_all(['code', 'pre']):
            text = code.get_text(strip=True)
            if len(text) > 20:
                code_blocks.append(text)
        
        # Determine content type
        content_type = self._detect_content_type(url, title, main_content)
        
        # Create summary
        summary = self._generate_summary(main_content)
        
        return ScrapedContent(
            url=url,
            title=title,
            content=main_content,
            summary=summary,
            content_type=content_type,
            links=list(set(links))[:50],
            headings=headings[:20],
            code_blocks=code_blocks[:20],
            metadata={
                "content_length": len(main_content),
                "headings_count": len(headings),
                "links_count": len(links),
                "code_blocks_count": len(code_blocks),
            },
        )

    def _basic_parse(self, url: str, html: str) -> ScrapedContent:
        """Basic HTML parsing without BeautifulSoup."""
        # Extract title
        title_match = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
        title = title_match.group(1).strip() if title_match else ""
        
        # Remove script and style tags
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        
        # Extract text content
        text = re.sub(r'<[^>]+>', ' ', html)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Extract links
        links = re.findall(r'href=["\']([^"\']+)["\']', html)
        absolute_links = [urljoin(url, link) for link in links]
        
        return ScrapedContent(
            url=url,
            title=title,
            content=text[:10000],
            summary=text[:500],
            links=list(set(absolute_links))[:30],
        )

    def _detect_content_type(self, url: str, title: str, content: str) -> ContentType:
        """Detect the type of content."""
        url_lower = url.lower()
        title_lower = title.lower()
        content_lower = content.lower()
        
        # Check URL patterns
        if any(x in url_lower for x in ['/docs/', '/documentation/', '/api/', '/reference/']):
            if 'api' in url_lower or 'api' in title_lower:
                return ContentType.API_REFERENCE
            return ContentType.DOCUMENTATION
        
        if '/blog/' in url_lower or 'blog' in url_lower:
            return ContentType.BLOG
        
        if '/tutorial/' in url_lower or 'tutorial' in title_lower:
            return ContentType.TUTORIAL
        
        # Check content patterns
        code_indicators = ['def ', 'class ', 'import ', 'function', 'const ', 'let ']
        code_score = sum(1 for indicator in code_indicators if indicator in content)
        
        if code_score > 5:
            return ContentType.CODE
        
        return ContentType.ARTICLE

    def _generate_summary(self, content: str, max_length: int = 500) -> str:
        """Generate a summary of content."""
        # Simple extractive summarization
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        if len(sentences) <= 3:
            return content[:max_length]
        
        # Take first few sentences
        summary_sentences = []
        current_length = 0
        
        for sentence in sentences[:5]:
            if current_length + len(sentence) > max_length:
                break
            summary_sentences.append(sentence)
            current_length += len(sentence)
        
        return ' '.join(summary_sentences)

    async def close(self) -> None:
        """Close HTTP session."""
        if self._session:
            await self._session.aclose()
            self._session = None


class ResearchEngine:
    """Engine for conducting research and gathering information."""

    def __init__(self, scraper: Optional[WebScraper] = None):
        self.scraper = scraper or WebScraper()
        self.research_history: List[Dict[str, Any]] = []

    async def research(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        max_sources: int = 5,
    ) -> Dict[str, Any]:
        """Conduct research on a topic."""
        logger.info(f"Starting research: {query}")
        
        # If no sources provided, search for them
        if not sources:
            sources = await self._search_sources(query, max_sources)
        
        # Scrape each source
        results = []
        for url in sources[:max_sources]:
            content = await self.scraper.scrape(url)
            if content:
                results.append(content)
        
        # Synthesize findings
        synthesis = self._synthesize(query, results)
        
        research_result = {
            "query": query,
            "sources_count": len(results),
            "sources": [r.to_dict() for r in results],
            "synthesis": synthesis,
            "timestamp": datetime.now().isoformat(),
        }
        
        self.research_history.append(research_result)
        return research_result

    async def _search_sources(self, query: str, max_results: int = 5) -> List[str]:
        """Search for relevant sources."""
        # Use DuckDuckGo search
        try:
            from duckduckgo_search import DDGS
            
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results * 2))
                return [r["href"] for r in results if "href" in r][:max_results]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def _synthesize(self, query: str, results: List[ScrapedContent]) -> str:
        """Synthesize research findings."""
        if not results:
            return "No relevant information found."
        
        synthesis_parts = []
        
        # Summary of sources
        synthesis_parts.append(f"Research on: {query}\n")
        synthesis_parts.append(f"Found {len(results)} relevant sources:\n")
        
        for i, result in enumerate(results, 1):
            synthesis_parts.append(f"\n{i}. {result.title}")
            synthesis_parts.append(f"   URL: {result.url}")
            synthesis_parts.append(f"   Summary: {result.summary[:200]}...")
        
        # Key findings
        synthesis_parts.append("\n\nKey Findings:")
        
        # Extract key points from summaries
        for result in results:
            if result.summary:
                # Take first sentence as key point
                first_sentence = result.summary.split('.')[0]
                if len(first_sentence) > 20:
                    synthesis_parts.append(f"- {first_sentence}")
        
        return '\n'.join(synthesis_parts)

    async def extract_code_examples(
        self,
        url: str,
        language: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Extract code examples from documentation."""
        content = await self.scraper.scrape(url)
        
        if not content or not content.code_blocks:
            return []
        
        examples = []
        for code in content.code_blocks:
            # Detect language
            detected_lang = language or self._detect_language(code)
            
            examples.append({
                "code": code,
                "language": detected_lang,
                "source": url,
                "title": content.title,
            })
        
        return examples

    def _detect_language(self, code: str) -> str:
        """Detect programming language from code."""
        code_lower = code.lower()
        
        indicators = {
            'python': ['def ', 'import ', 'class ', 'print(', 'None', 'True', 'False'],
            'javascript': ['const ', 'let ', 'var ', 'function', '=>', 'console.log'],
            'typescript': ['interface ', 'type ', ': ', 'as ', 'readonly'],
            'java': ['public class', 'private ', 'protected ', 'System.out'],
            'go': ['package ', 'func ', 'import (', 'fmt.'],
            'rust': ['fn ', 'let mut', 'impl ', 'use '],
            'bash': ['#!/bin/bash', 'echo ', 'if [', 'fi'],
        }
        
        scores = {}
        for lang, patterns in indicators.items():
            scores[lang] = sum(1 for pattern in patterns if pattern in code)
        
        if scores:
            return max(scores, key=scores.get)
        
        return 'unknown'

    async def close(self) -> None:
        """Clean up resources."""
        await self.scraper.close()
