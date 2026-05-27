"""Performance profiling and optimization suggestions."""

from __future__ import annotations

import time
import functools
import inspect
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict, Callable
from datetime import datetime
from collections import defaultdict
import asyncio

from legionhercules.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetric:
    """A single performance metric."""
    name: str
    duration_ms: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FunctionProfile:
    """Profile data for a function."""
    function_name: str
    module_name: str
    call_count: int = 0
    total_time_ms: float = 0.0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    last_called: Optional[datetime] = None

    def record_call(self, duration_ms: float) -> None:
        """Record a function call."""
        self.call_count += 1
        self.total_time_ms += duration_ms
        self.min_time_ms = min(self.min_time_ms, duration_ms)
        self.max_time_ms = max(self.max_time_ms, duration_ms)
        self.avg_time_ms = self.total_time_ms / self.call_count
        self.last_called = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "function": f"{self.module_name}.{self.function_name}",
            "call_count": self.call_count,
            "total_time_ms": round(self.total_time_ms, 2),
            "avg_time_ms": round(self.avg_time_ms, 2),
            "min_time_ms": round(self.min_time_ms, 2) if self.min_time_ms != float('inf') else 0,
            "max_time_ms": round(self.max_time_ms, 2),
            "last_called": self.last_called.isoformat() if self.last_called else None,
        }


class PerformanceProfiler:
    """Profiles function performance."""

    def __init__(self):
        self.profiles: Dict[str, FunctionProfile] = {}
        self.metrics: List[PerformanceMetric] = []
        self._active = False

    def start(self) -> None:
        """Start profiling."""
        self._active = True
        logger.info("Performance profiling started")

    def stop(self) -> None:
        """Stop profiling."""
        self._active = False
        logger.info("Performance profiling stopped")

    def profile(self, func: Callable) -> Callable:
        """Decorator to profile a function."""
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not self._active:
                return await func(*args, **kwargs)
            
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                duration = (time.perf_counter() - start) * 1000
                self._record_profile(func, duration)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not self._active:
                return func(*args, **kwargs)
            
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration = (time.perf_counter() - start) * 1000
                self._record_profile(func, duration)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    def _record_profile(self, func: Callable, duration_ms: float) -> None:
        """Record profile data."""
        key = f"{func.__module__}.{func.__name__}"
        
        if key not in self.profiles:
            self.profiles[key] = FunctionProfile(
                function_name=func.__name__,
                module_name=func.__module__,
            )
        
        self.profiles[key].record_call(duration_ms)

    def get_profile(self, function_name: str) -> Optional[FunctionProfile]:
        """Get profile for a specific function."""
        return self.profiles.get(function_name)

    def get_all_profiles(self) -> List[FunctionProfile]:
        """Get all function profiles."""
        return sorted(
            self.profiles.values(),
            key=lambda p: p.total_time_ms,
            reverse=True,
        )

    def get_slow_functions(self, threshold_ms: float = 100) -> List[FunctionProfile]:
        """Get functions slower than threshold."""
        return [
            p for p in self.profiles.values()
            if p.avg_time_ms > threshold_ms
        ]

    def get_hotspots(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Get top performance hotspots."""
        sorted_profiles = self.get_all_profiles()
        return [p.to_dict() for p in sorted_profiles[:top_n]]

    def reset(self) -> None:
        """Reset all profiling data."""
        self.profiles.clear()
        self.metrics.clear()
        logger.info("Profiling data reset")

    def generate_report(self) -> Dict[str, Any]:
        """Generate performance report."""
        profiles = self.get_all_profiles()
        
        if not profiles:
            return {"message": "No profiling data available"}
        
        total_calls = sum(p.call_count for p in profiles)
        total_time = sum(p.total_time_ms for p in profiles)
        
        return {
            "summary": {
                "total_functions": len(profiles),
                "total_calls": total_calls,
                "total_time_ms": round(total_time, 2),
                "avg_time_per_call_ms": round(total_time / total_calls, 2) if total_calls > 0 else 0,
            },
            "hotspots": self.get_hotspots(10),
            "slow_functions": [p.to_dict() for p in self.get_slow_functions(100)],
        }


class OptimizationAdvisor:
    """Provides optimization suggestions."""

    SUGGESTIONS = {
        "repeated_calls": {
            "pattern": lambda p: p.call_count > 100 and p.avg_time_ms > 10,
            "message": "Function '{name}' is called {count} times. Consider caching results.",
            "priority": "high",
        },
        "slow_function": {
            "pattern": lambda p: p.avg_time_ms > 500,
            "message": "Function '{name}' is slow ({avg:.1f}ms avg). Consider optimization or async.",
            "priority": "critical",
        },
        "memory_intensive": {
            "pattern": lambda p: p.max_time_ms > p.avg_time_ms * 10,
            "message": "Function '{name}' has high variance. Check for memory issues.",
            "priority": "medium",
        },
    }

    def __init__(self, profiler: PerformanceProfiler):
        self.profiler = profiler

    def analyze(self) -> List[Dict[str, Any]]:
        """Analyze profiles and generate suggestions."""
        suggestions = []
        
        for profile in self.profiler.profiles.values():
            for suggestion_type, config in self.SUGGESTIONS.items():
                if config["pattern"](profile):
                    suggestions.append({
                        "type": suggestion_type,
                        "priority": config["priority"],
                        "function": f"{profile.module_name}.{profile.function_name}",
                        "message": config["message"].format(
                            name=profile.function_name,
                            count=profile.call_count,
                            avg=profile.avg_time_ms,
                        ),
                        "metrics": profile.to_dict(),
                    })
        
        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        suggestions.sort(key=lambda s: priority_order.get(s["priority"], 4))
        
        return suggestions

    def get_quick_wins(self) -> List[Dict[str, Any]]:
        """Get quick optimization opportunities."""
        all_suggestions = self.analyze()
        return [s for s in all_suggestions if s["priority"] in ["critical", "high"]]


class CodeOptimizer:
    """Suggests code optimizations."""

    PATTERNS = {
        "list_comprehension": {
            "anti_pattern": r"for\s+\w+\s+in\s+\w+:\s*\n\s*\w+\.append",
            "suggestion": "Use list comprehension for better performance",
            "example": "[x for x in items] instead of loop with append",
        },
        "dict_lookup": {
            "anti_pattern": r"if\s+\w+\s+in\s+\w+\.keys\(\)",
            "suggestion": "Use 'key in dict' instead of 'key in dict.keys()'",
            "example": "if key in my_dict: (faster)",
        },
        "string_concat": {
            "anti_pattern": r"\w+\s*\+=\s*['\"]",
            "suggestion": "Use str.join() for multiple string concatenations",
            "example": '"".join(strings) instead of += in loop',
        },
    }

    def __init__(self, llm_provider=None):
        self.llm_provider = llm_provider

    def analyze_code(self, code: str) -> List[Dict[str, Any]]:
        """Analyze code for optimization opportunities."""
        import re
        
        suggestions = []
        
        for pattern_name, config in self.PATTERNS.items():
            if re.search(config["anti_pattern"], code):
                suggestions.append({
                    "type": pattern_name,
                    "suggestion": config["suggestion"],
                    "example": config["example"],
                })
        
        return suggestions

    async def suggest_optimizations(self, code: str) -> List[Dict[str, Any]]:
        """Get AI-powered optimization suggestions."""
        if not self.llm_provider:
            return self.analyze_code(code)
        
        prompt = f"""Analyze this Python code and suggest performance optimizations:

```python
{code}
```

Provide specific, actionable suggestions with code examples."""
        
        try:
            response = await self.llm_provider.generate(prompt)
            
            # Parse suggestions from response
            suggestions = []
            lines = response.content.split("\n")
            current_suggestion = {}
            
            for line in lines:
                if line.strip().startswith(("1.", "2.", "3.", "-")):
                    if current_suggestion:
                        suggestions.append(current_suggestion)
                    current_suggestion = {
                        "type": "ai_suggestion",
                        "suggestion": line.strip().lstrip("123.- "),
                    }
                elif line.strip().startswith("```"):
                    continue
                elif current_suggestion and "example" not in current_suggestion:
                    current_suggestion["example"] = line.strip()
            
            if current_suggestion:
                suggestions.append(current_suggestion)
            
            return suggestions
            
        except Exception as e:
            logger.error(f"AI optimization analysis failed: {e}")
            return self.analyze_code(code)


class ResourceMonitor:
    """Monitors system resources."""

    def __init__(self):
        self.metrics_history: List[Dict[str, Any]] = []

    async def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        import psutil
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "cpu": {
                "percent": psutil.cpu_percent(interval=0.1),
                "count": psutil.cpu_count(),
            },
            "memory": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "percent": psutil.virtual_memory().percent,
            },
            "disk": {
                "total": psutil.disk_usage('/').total,
                "used": psutil.disk_usage('/').used,
                "free": psutil.disk_usage('/').free,
                "percent": psutil.disk_usage('/').percent,
            },
        }
        
        self.metrics_history.append(metrics)
        
        # Keep only last 1000 metrics
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
        
        return metrics

    def get_average_metrics(self, last_n: int = 100) -> Dict[str, Any]:
        """Get average metrics over last N samples."""
        if not self.metrics_history:
            return {}
        
        recent = self.metrics_history[-last_n:]
        
        return {
            "cpu_avg": sum(m["cpu"]["percent"] for m in recent) / len(recent),
            "memory_avg": sum(m["memory"]["percent"] for m in recent) / len(recent),
            "samples": len(recent),
        }

    def check_thresholds(
        self,
        cpu_threshold: float = 80.0,
        memory_threshold: float = 80.0,
        disk_threshold: float = 90.0,
    ) -> List[Dict[str, Any]]:
        """Check if any metrics exceed thresholds."""
        alerts = []
        
        if not self.metrics_history:
            return alerts
        
        latest = self.metrics_history[-1]
        
        if latest["cpu"]["percent"] > cpu_threshold:
            alerts.append({
                "type": "cpu",
                "message": f"CPU usage is {latest['cpu']['percent']:.1f}%",
                "threshold": cpu_threshold,
                "current": latest["cpu"]["percent"],
            })
        
        if latest["memory"]["percent"] > memory_threshold:
            alerts.append({
                "type": "memory",
                "message": f"Memory usage is {latest['memory']['percent']:.1f}%",
                "threshold": memory_threshold,
                "current": latest["memory"]["percent"],
            })
        
        if latest["disk"]["percent"] > disk_threshold:
            alerts.append({
                "type": "disk",
                "message": f"Disk usage is {latest['disk']['percent']:.1f}%",
                "threshold": disk_threshold,
                "current": latest["disk"]["percent"],
            })
        
        return alerts
