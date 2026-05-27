"""Reflection engine for self-improving agent loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict
from datetime import datetime
from enum import Enum

from legionhercules.utils.logging import get_logger

logger = get_logger(__name__)


class ReflectionStatus(Enum):
    """Status of reflection."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ReflectionResult:
    """Result of a reflection cycle."""
    task_id: str
    status: ReflectionStatus
    success: bool
    original_output: str
    reflection_analysis: str
    improvements_suggested: List[str]
    improved_output: Optional[str] = None
    confidence_score: float = 0.0  # 0.0 to 1.0
    iteration: int = 1
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "success": self.success,
            "original_output": self.original_output,
            "reflection_analysis": self.reflection_analysis,
            "improvements_suggested": self.improvements_suggested,
            "improved_output": self.improved_output,
            "confidence_score": self.confidence_score,
            "iteration": self.iteration,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class ReflectionEngine:
    """Engine for reflecting on task execution and suggesting improvements."""

    REFLECTION_PROMPT_TEMPLATE = """You are a critical reflection system. Analyze the following task execution and provide constructive feedback.

Task: {task_description}
Original Output:
{original_output}

Expected Outcome: {expected_outcome}

Please analyze:
1. What was done well?
2. What could be improved?
3. Are there any errors or issues?
4. What specific changes would make this better?

Provide your analysis in a structured format."""

    IMPROVEMENT_PROMPT_TEMPLATE = """Based on the following reflection analysis, generate an improved version of the output.

Original Output:
{original_output}

Reflection Analysis:
{reflection_analysis}

Suggested Improvements:
{improvements}

Please provide the improved output addressing all the suggested improvements."""

    def __init__(self, llm_provider=None):
        self.llm_provider = llm_provider
        self.reflection_history: List[ReflectionResult] = []

    async def reflect(
        self,
        task_id: str,
        task_description: str,
        original_output: str,
        expected_outcome: Optional[str] = None,
        iteration: int = 1,
    ) -> ReflectionResult:
        """Reflect on task execution and suggest improvements."""
        logger.info(f"Starting reflection for task {task_id}, iteration {iteration}")
        
        result = ReflectionResult(
            task_id=task_id,
            status=ReflectionStatus.IN_PROGRESS,
            success=False,
            original_output=original_output,
            reflection_analysis="",
            improvements_suggested=[],
            iteration=iteration,
        )

        try:
            # Generate reflection analysis
            reflection_prompt = self.REFLECTION_PROMPT_TEMPLATE.format(
                task_description=task_description,
                original_output=original_output,
                expected_outcome=expected_outcome or "High quality, accurate, and complete result",
            )

            if self.llm_provider:
                reflection_response = await self.llm_provider.generate(reflection_prompt)
                reflection_analysis = reflection_response.content
            else:
                reflection_analysis = self._default_reflection(
                    task_description, original_output
                )

            result.reflection_analysis = reflection_analysis

            # Parse improvements from analysis
            improvements = self._extract_improvements(reflection_analysis)
            result.improvements_suggested = improvements

            # Determine if improvements are needed
            needs_improvement = self._needs_improvement(reflection_analysis)
            
            if needs_improvement and improvements:
                # Generate improved output
                improvement_prompt = self.IMPROVEMENT_PROMPT_TEMPLATE.format(
                    original_output=original_output,
                    reflection_analysis=reflection_analysis,
                    improvements="\n".join(f"- {imp}" for imp in improvements),
                )

                if self.llm_provider:
                    improved_response = await self.llm_provider.generate(improvement_prompt)
                    result.improved_output = improved_response.content
                else:
                    result.improved_output = original_output

                result.confidence_score = self._calculate_confidence(
                    reflection_analysis, improvements
                )
            else:
                result.improved_output = original_output
                result.confidence_score = 0.9  # High confidence if no improvements needed

            result.status = ReflectionStatus.COMPLETED
            result.success = True

        except Exception as e:
            logger.error(f"Reflection failed: {e}")
            result.status = ReflectionStatus.FAILED
            result.reflection_analysis = f"Reflection error: {str(e)}"

        self.reflection_history.append(result)
        return result

    def _default_reflection(
        self, task_description: str, original_output: str
    ) -> str:
        """Generate default reflection without LLM."""
        analysis = []
        
        # Check output length
        if len(original_output) < 50:
            analysis.append("Output appears too brief. Consider expanding with more detail.")
        elif len(original_output) > 5000:
            analysis.append("Output is quite long. Consider summarizing key points.")

        # Check for common issues
        if "error" in original_output.lower():
            analysis.append("Output contains error messages. Review and fix issues.")
        
        if "TODO" in original_output or "FIXME" in original_output:
            analysis.append("Output contains TODO/FIXME markers. Complete these items.")

        if not analysis:
            analysis.append("Output appears complete and well-structured.")

        return "\n".join(f"{i+1}. {item}" for i, item in enumerate(analysis))

    def _extract_improvements(self, reflection_analysis: str) -> List[str]:
        """Extract improvement suggestions from reflection."""
        improvements = []
        
        # Look for common improvement patterns
        lines = reflection_analysis.split("\n")
        for line in lines:
            line = line.strip()
            # Look for bullet points or numbered items that suggest improvements
            if line.startswith(("-", "*", "•")) or (line[0].isdigit() and "." in line[:3]):
                content = line.lstrip("- *•0123456789. ")
                if any(keyword in content.lower() for keyword in [
                    "improve", "add", "fix", "consider", "should", "could", "better",
                    "enhance", "expand", "clarify", "correct", "update"
                ]):
                    improvements.append(content)

        return improvements

    def _needs_improvement(self, reflection_analysis: str) -> bool:
        """Determine if improvements are needed based on reflection."""
        # Check for positive indicators that no improvements are needed
        positive_indicators = [
            "well done", "excellent", "perfect", "complete", "satisfactory",
            "no issues", "no improvements", "good job", "accurate"
        ]
        
        analysis_lower = reflection_analysis.lower()
        
        # If many positive indicators, probably doesn't need improvement
        positive_count = sum(1 for indicator in positive_indicators if indicator in analysis_lower)
        
        return positive_count < 2

    def _calculate_confidence(self, reflection_analysis: str, improvements: List[str]) -> float:
        """Calculate confidence score based on reflection."""
        base_score = 0.7
        
        # More improvements = lower confidence in original
        improvement_penalty = len(improvements) * 0.05
        
        # Check for confidence indicators
        if "high confidence" in reflection_analysis.lower():
            base_score += 0.2
        elif "low confidence" in reflection_analysis.lower():
            base_score -= 0.2

        return max(0.0, min(1.0, base_score - improvement_penalty))

    def get_reflection_history(self, task_id: Optional[str] = None) -> List[ReflectionResult]:
        """Get reflection history, optionally filtered by task."""
        if task_id:
            return [r for r in self.reflection_history if r.task_id == task_id]
        return self.reflection_history

    def get_improvement_stats(self) -> Dict[str, Any]:
        """Get statistics on improvements."""
        if not self.reflection_history:
            return {}

        total = len(self.reflection_history)
        successful = sum(1 for r in self.reflection_history if r.success)
        avg_confidence = sum(r.confidence_score for r in self.reflection_history) / total
        
        return {
            "total_reflections": total,
            "successful_reflections": successful,
            "success_rate": successful / total if total > 0 else 0,
            "average_confidence": avg_confidence,
            "total_improvements_suggested": sum(
                len(r.improvements_suggested) for r in self.reflection_history
            ),
        }
