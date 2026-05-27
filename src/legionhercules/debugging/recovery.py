"""Autonomous debugging and error recovery system."""

from __future__ import annotations

import re
import traceback
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict, Callable, Awaitable
from datetime import datetime
from enum import Enum

from legionhercules.utils.logging import get_logger

logger = get_logger(__name__)


class ErrorType(Enum):
    """Types of errors that can be recovered from."""
    SYNTAX_ERROR = "syntax_error"
    IMPORT_ERROR = "import_error"
    ATTRIBUTE_ERROR = "attribute_error"
    KEY_ERROR = "key_error"
    INDEX_ERROR = "index_error"
    TYPE_ERROR = "type_error"
    VALUE_ERROR = "value_error"
    FILE_NOT_FOUND = "file_not_found"
    PERMISSION_ERROR = "permission_error"
    CONNECTION_ERROR = "connection_error"
    TIMEOUT_ERROR = "timeout_error"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """Recovery strategies."""
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    SUBSTITUTE = "substitute"
    ROLLBACK = "rollback"
    ABORT = "abort"


@dataclass
class ErrorContext:
    """Context of an error."""
    error_type: ErrorType
    error_message: str
    traceback_str: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    function_name: Optional[str] = None
    context_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.error_type.value,
            "error_message": self.error_message,
            "traceback": self.traceback_str,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "function_name": self.function_name,
            "context_data": self.context_data,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class RecoveryAction:
    """A recovery action."""
    strategy: RecoveryStrategy
    description: str
    action: Callable[[], Awaitable[Any]]
    confidence: float = 0.5
    max_attempts: int = 3


@dataclass
class RecoveryResult:
    """Result of a recovery attempt."""
    success: bool
    strategy_used: RecoveryStrategy
    attempts: int
    original_error: ErrorContext
    final_result: Any = None
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "strategy_used": self.strategy_used.value,
            "attempts": self.attempts,
            "original_error": self.original_error.to_dict(),
            "final_result": self.final_result,
            "error_message": self.error_message,
            "timestamp": self.timestamp.isoformat(),
        }


class ErrorClassifier:
    """Classifies errors into types."""

    ERROR_PATTERNS = {
        ErrorType.SYNTAX_ERROR: [
            r"SyntaxError",
            r"IndentationError",
            r"invalid syntax",
        ],
        ErrorType.IMPORT_ERROR: [
            r"ImportError",
            r"ModuleNotFoundError",
            r"No module named",
        ],
        ErrorType.ATTRIBUTE_ERROR: [
            r"AttributeError",
            r"has no attribute",
        ],
        ErrorType.KEY_ERROR: [
            r"KeyError",
        ],
        ErrorType.INDEX_ERROR: [
            r"IndexError",
            r"list index out of range",
        ],
        ErrorType.TYPE_ERROR: [
            r"TypeError",
        ],
        ErrorType.VALUE_ERROR: [
            r"ValueError",
        ],
        ErrorType.FILE_NOT_FOUND: [
            r"FileNotFoundError",
            r"No such file",
        ],
        ErrorType.PERMISSION_ERROR: [
            r"PermissionError",
            r"Permission denied",
        ],
        ErrorType.CONNECTION_ERROR: [
            r"ConnectionError",
            r"Connection refused",
            r"Connection reset",
        ],
        ErrorType.TIMEOUT_ERROR: [
            r"TimeoutError",
            r"timed out",
        ],
    }

    @classmethod
    def classify(cls, error: Exception) -> ErrorType:
        """Classify an error."""
        error_str = f"{type(error).__name__}: {str(error)}"
        
        for error_type, patterns in cls.ERROR_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, error_str, re.IGNORECASE):
                    return error_type
        
        return ErrorType.UNKNOWN

    @classmethod
    def extract_context(cls, error: Exception) -> ErrorContext:
        """Extract error context from exception."""
        error_type = cls.classify(error)
        error_message = str(error)
        traceback_str = traceback.format_exc()
        
        # Extract file and line info from traceback
        file_path = None
        line_number = None
        function_name = None
        
        tb = traceback.extract_tb(error.__traceback__)
        if tb:
            last_frame = tb[-1]
            file_path = last_frame.filename
            line_number = last_frame.lineno
            function_name = last_frame.name
        
        return ErrorContext(
            error_type=error_type,
            error_message=error_message,
            traceback_str=traceback_str,
            file_path=file_path,
            line_number=line_number,
            function_name=function_name,
        )


class RecoveryEngine:
    """Engine for autonomous error recovery."""

    def __init__(self, llm_provider=None):
        self.llm_provider = llm_provider
        self.recovery_strategies: Dict[ErrorType, List[RecoveryAction]] = {}
        self.recovery_history: List[RecoveryResult] = []
        self._register_default_strategies()

    def _register_default_strategies(self) -> None:
        """Register default recovery strategies."""
        # Connection errors - retry with backoff
        self.register_strategy(
            ErrorType.CONNECTION_ERROR,
            RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                description="Retry with exponential backoff",
                action=self._retry_with_backoff,
                confidence=0.8,
            )
        )
        
        # Timeout errors - retry with increased timeout
        self.register_strategy(
            ErrorType.TIMEOUT_ERROR,
            RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                description="Retry with increased timeout",
                action=self._retry_with_timeout,
                confidence=0.7,
            )
        )
        
        # Import errors - try alternative imports
        self.register_strategy(
            ErrorType.IMPORT_ERROR,
            RecoveryAction(
                strategy=RecoveryStrategy.SUBSTITUTE,
                description="Try alternative import",
                action=self._try_alternative_import,
                confidence=0.6,
            )
        )
        
        # File not found - check alternatives
        self.register_strategy(
            ErrorType.FILE_NOT_FOUND,
            RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK,
                description="Use fallback path",
                action=self._use_fallback_path,
                confidence=0.7,
            )
        )

    def register_strategy(
        self,
        error_type: ErrorType,
        action: RecoveryAction,
    ) -> None:
        """Register a recovery strategy."""
        if error_type not in self.recovery_strategies:
            self.recovery_strategies[error_type] = []
        self.recovery_strategies[error_type].append(action)

    async def attempt_recovery(
        self,
        error: Exception,
        operation: Callable[[], Awaitable[Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> RecoveryResult:
        """Attempt to recover from an error."""
        error_context = ErrorClassifier.extract_context(error)
        error_context.context_data = context or {}
        
        logger.info(f"Attempting recovery for {error_context.error_type.value}")
        
        # Get strategies for this error type
        strategies = self.recovery_strategies.get(error_context.error_type, [])
        
        if not strategies:
            # Try generic retry
            strategies = [
                RecoveryAction(
                    strategy=RecoveryStrategy.RETRY,
                    description="Generic retry",
                    action=self._simple_retry,
                    confidence=0.3,
                )
            ]
        
        # Try each strategy
        for strategy in sorted(strategies, key=lambda s: s.confidence, reverse=True):
            for attempt in range(strategy.max_attempts):
                try:
                    logger.info(f"Trying {strategy.description} (attempt {attempt + 1})")
                    
                    result = await strategy.action(operation, error_context, attempt)
                    
                    recovery_result = RecoveryResult(
                        success=True,
                        strategy_used=strategy.strategy,
                        attempts=attempt + 1,
                        original_error=error_context,
                        final_result=result,
                    )
                    
                    self.recovery_history.append(recovery_result)
                    return recovery_result
                    
                except Exception as e:
                    logger.warning(f"Recovery attempt failed: {e}")
                    continue
        
        # All strategies failed
        recovery_result = RecoveryResult(
            success=False,
            strategy_used=RecoveryStrategy.ABORT,
            attempts=sum(s.max_attempts for s in strategies),
            original_error=error_context,
            error_message="All recovery strategies failed",
        )
        
        self.recovery_history.append(recovery_result)
        return recovery_result

    async def execute_with_recovery(
        self,
        operation: Callable[[], Awaitable[Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Execute an operation with automatic recovery."""
        try:
            return await operation()
        except Exception as e:
            result = await self.attempt_recovery(e, operation, context)
            
            if result.success:
                return result.final_result
            else:
                raise

    # Default recovery actions
    async def _retry_with_backoff(
        self,
        operation: Callable[[], Awaitable[Any]],
        context: ErrorContext,
        attempt: int,
    ) -> Any:
        """Retry with exponential backoff."""
        import asyncio
        delay = 2 ** attempt
        await asyncio.sleep(delay)
        return await operation()

    async def _retry_with_timeout(
        self,
        operation: Callable[[], Awaitable[Any]],
        context: ErrorContext,
        attempt: int,
    ) -> Any:
        """Retry with increased timeout."""
        # This would need to be implemented at the operation level
        return await operation()

    async def _simple_retry(
        self,
        operation: Callable[[], Awaitable[Any]],
        context: ErrorContext,
        attempt: int,
    ) -> Any:
        """Simple retry without delay."""
        return await operation()

    async def _try_alternative_import(
        self,
        operation: Callable[[], Awaitable[Any]],
        context: ErrorContext,
        attempt: int,
    ) -> Any:
        """Try alternative import paths."""
        # This is a placeholder - would need module-specific logic
        return await operation()

    async def _use_fallback_path(
        self,
        operation: Callable[[], Awaitable[Any]],
        context: ErrorContext,
        attempt: int,
    ) -> Any:
        """Use fallback file path."""
        # This is a placeholder - would need path-specific logic
        return await operation()

    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get recovery statistics."""
        if not self.recovery_history:
            return {}
        
        total = len(self.recovery_history)
        successful = sum(1 for r in self.recovery_history if r.success)
        
        by_type = {}
        for result in self.recovery_history:
            error_type = result.original_error.error_type.value
            if error_type not in by_type:
                by_type[error_type] = {"total": 0, "success": 0}
            by_type[error_type]["total"] += 1
            if result.success:
                by_type[error_type]["success"] += 1
        
        return {
            "total_attempts": total,
            "successful_recoveries": successful,
            "success_rate": successful / total if total > 0 else 0,
            "by_error_type": by_type,
        }


class AutoDebugger:
    """Automatic debugging assistant."""

    def __init__(self, llm_provider=None):
        self.llm_provider = llm_provider
        self.fix_history: List[Dict[str, Any]] = []

    async def analyze_error(self, error: Exception, code: Optional[str] = None) -> Dict[str, Any]:
        """Analyze an error and suggest fixes."""
        context = ErrorClassifier.extract_context(error)
        
        analysis = {
            "error_type": context.error_type.value,
            "error_message": context.error_message,
            "location": f"{context.file_path}:{context.line_number}" if context.file_path else "unknown",
            "suggested_fixes": [],
        }
        
        # Generate suggestions based on error type
        if context.error_type == ErrorType.SYNTAX_ERROR:
            analysis["suggested_fixes"] = [
                "Check for missing colons, parentheses, or quotes",
                "Verify indentation is consistent",
                "Check for typos in keywords",
            ]
        elif context.error_type == ErrorType.IMPORT_ERROR:
            analysis["suggested_fixes"] = [
                f"Install missing package: pip install {self._extract_module_name(context.error_message)}",
                "Check if module name is correct",
                "Verify virtual environment is activated",
            ]
        elif context.error_type == ErrorType.ATTRIBUTE_ERROR:
            analysis["suggested_fixes"] = [
                "Check if object has the attribute",
                "Verify object type is correct",
                "Check for typos in attribute name",
            ]
        elif context.error_type == ErrorType.KEY_ERROR:
            analysis["suggested_fixes"] = [
                "Check if key exists before accessing",
                "Use .get() method with default value",
                "Verify dictionary contents",
            ]
        
        # Use LLM for advanced analysis if available
        if self.llm_provider and code:
            llm_suggestions = await self._llm_analyze(error, code, context)
            analysis["suggested_fixes"].extend(llm_suggestions)
        
        return analysis

    def _extract_module_name(self, error_message: str) -> str:
        """Extract module name from import error."""
        match = re.search(r"No module named ['\"]([^'\"]+)['\"]", error_message)
        return match.group(1) if match else "unknown"

    async def _llm_analyze(
        self,
        error: Exception,
        code: str,
        context: ErrorContext,
    ) -> List[str]:
        """Use LLM to analyze error."""
        prompt = f"""Analyze this Python error and suggest fixes:

Error: {context.error_message}
Location: {context.file_path}:{context.line_number}

Code:
```python
{code}
```

Provide 2-3 specific suggestions to fix this error."""
        
        try:
            response = await self.llm_provider.generate(prompt)
            # Parse suggestions from response
            suggestions = [s.strip() for s in response.content.split("\n") if s.strip().startswith("-")]
            return suggestions
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return []

    async def suggest_fix(self, error: Exception, code: str) -> Optional[str]:
        """Suggest a code fix using LLM."""
        if not self.llm_provider:
            return None
        
        context = ErrorClassifier.extract_context(error)
        
        prompt = f"""Fix this Python code that has an error:

Error: {context.error_message}
Line {context.line_number}: {context.function_name}

Code:
```python
{code}
```

Provide only the corrected code, no explanations."""
        
        try:
            response = await self.llm_provider.generate(prompt)
            
            # Extract code from response
            fixed_code = response.content
            if "```python" in fixed_code:
                fixed_code = fixed_code.split("```python")[1].split("```")[0]
            elif "```" in fixed_code:
                fixed_code = fixed_code.split("```")[1].split("```")[0]
            
            self.fix_history.append({
                "error": str(error),
                "original": code,
                "fixed": fixed_code,
                "timestamp": datetime.now().isoformat(),
            })
            
            return fixed_code.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate fix: {e}")
            return None
