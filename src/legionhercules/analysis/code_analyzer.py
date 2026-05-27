"""Code analysis and refactoring capabilities."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from legionhercules.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CodeIssue:
    """Represents a code issue or suggestion."""
    severity: str  # error, warning, info
    category: str  # style, complexity, security, performance
    message: str
    file_path: str
    line_number: int
    column: int = 0
    suggestion: Optional[str] = None
    code_snippet: Optional[str] = None


@dataclass
class AnalysisReport:
    """Complete code analysis report."""
    file_path: str
    language: str
    issues: list[CodeIssue] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0  # 0-100 quality score
    
    def get_issues_by_severity(self, severity: str) -> list[CodeIssue]:
        return [i for i in self.issues if i.severity == severity]
    
    def get_issues_by_category(self, category: str) -> list[CodeIssue]:
        return [i for i in self.issues if i.category == category]


class CodeAnalyzer:
    """Analyzes code for quality, complexity, and refactoring opportunities."""
    
    def __init__(self):
        self.rules = self._load_rules()
    
    def _load_rules(self) -> dict:
        """Load analysis rules."""
        return {
            "python": {
                "max_line_length": 100,
                "max_function_length": 50,
                "max_complexity": 10,
                "indentation": 4,
            },
            "javascript": {
                "max_line_length": 100,
                "max_function_length": 50,
            },
        }
    
    def analyze_file(self, file_path: str) -> AnalysisReport:
        """Analyze a single file."""
        path = Path(file_path)
        language = self._detect_language(path)
        
        report = AnalysisReport(
            file_path=file_path,
            language=language,
        )
        
        if not path.exists():
            report.issues.append(CodeIssue(
                severity="error",
                category="file",
                message="File not found",
                file_path=file_path,
                line_number=0,
            ))
            return report
        
        content = path.read_text(encoding="utf-8", errors="ignore")
        lines = content.splitlines()
        
        # Basic metrics
        report.metrics["total_lines"] = len(lines)
        report.metrics["code_lines"] = len([l for l in lines if l.strip() and not l.strip().startswith('#')])
        report.metrics["blank_lines"] = len([l for l in lines if not l.strip()])
        report.metrics["comment_lines"] = len([l for l in lines if l.strip().startswith('#')])
        report.metrics["file_size"] = path.stat().st_size
        
        if language == "python":
            self._analyze_python(content, lines, report)
        elif language in ("javascript", "typescript"):
            self._analyze_javascript(content, lines, report)
        
        # Calculate score
        report.score = self._calculate_score(report)
        
        return report
    
    def analyze_project(
        self,
        project_path: str,
        exclude_patterns: Optional[list[str]] = None,
    ) -> list[AnalysisReport]:
        """Analyze entire project."""
        root = Path(project_path)
        exclude = exclude_patterns or ["node_modules", ".git", "__pycache__", "venv", ".venv"]
        
        reports = []
        for file_path in root.rglob("*"):
            if not file_path.is_file():
                continue
            
            rel_path = str(file_path.relative_to(root))
            if any(pattern in rel_path for pattern in exclude):
                continue
            
            if file_path.suffix in ('.py', '.js', '.ts', '.jsx', '.tsx'):
                try:
                    report = self.analyze_file(str(file_path))
                    reports.append(report)
                except Exception as e:
                    logger.warning(f"Failed to analyze {file_path}: {e}")
        
        return reports
    
    def _detect_language(self, path: Path) -> str:
        """Detect programming language from file extension."""
        ext_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'jsx',
            '.tsx': 'tsx',
            '.go': 'go',
            '.rs': 'rust',
            '.java': 'java',
        }
        return ext_map.get(path.suffix, 'unknown')
    
    def _analyze_python(self, content: str, lines: list[str], report: AnalysisReport) -> None:
        """Analyze Python code."""
        rules = self.rules.get("python", {})
        max_line_length = rules.get("max_line_length", 100)
        
        # Check line lengths
        for i, line in enumerate(lines, 1):
            if len(line) > max_line_length:
                report.issues.append(CodeIssue(
                    severity="warning",
                    category="style",
                    message=f"Line too long ({len(line)} > {max_line_length} characters)",
                    file_path=report.file_path,
                    line_number=i,
                    column=max_line_length,
                    suggestion="Break line into multiple lines",
                ))
        
        # Parse AST for deeper analysis
        try:
            tree = ast.parse(content)
            self._analyze_python_ast(tree, report)
        except SyntaxError as e:
            report.issues.append(CodeIssue(
                severity="error",
                category="syntax",
                message=f"Syntax error: {e}",
                file_path=report.file_path,
                line_number=e.lineno or 0,
                column=e.offset or 0,
            ))
    
    def _analyze_python_ast(self, tree: ast.AST, report: AnalysisReport) -> None:
        """Analyze Python AST."""
        for node in ast.walk(tree):
            # Check function complexity
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                complexity = self._calculate_complexity(node)
                if complexity > 10:
                    report.issues.append(CodeIssue(
                        severity="warning",
                        category="complexity",
                        message=f"Function '{node.name}' has high complexity (cyclomatic: {complexity})",
                        file_path=report.file_path,
                        line_number=node.lineno,
                        suggestion="Refactor into smaller functions",
                    ))
                
                # Check function length
                func_lines = node.end_lineno - node.lineno if node.end_lineno else 0
                if func_lines > 50:
                    report.issues.append(CodeIssue(
                        severity="warning",
                        category="complexity",
                        message=f"Function '{node.name}' is too long ({func_lines} lines)",
                        file_path=report.file_path,
                        line_number=node.lineno,
                        suggestion="Extract logic into helper functions",
                    ))
                
                # Check for missing docstrings
                if not ast.get_docstring(node):
                    report.issues.append(CodeIssue(
                        severity="info",
                        category="style",
                        message=f"Function '{node.name}' lacks docstring",
                        file_path=report.file_path,
                        line_number=node.lineno,
                        suggestion="Add docstring to document function purpose",
                    ))
            
            # Check for bare except
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    report.issues.append(CodeIssue(
                        severity="warning",
                        category="security",
                        message="Bare except clause found",
                        file_path=report.file_path,
                        line_number=node.lineno,
                        suggestion="Use 'except Exception:' instead of bare 'except:'",
                    ))
            
            # Check for mutable default arguments
            if isinstance(node, ast.FunctionDef):
                for default in node.args.defaults:
                    if isinstance(default, (ast.List, ast.Dict)):
                        report.issues.append(CodeIssue(
                            severity="warning",
                            category="security",
                            message="Mutable default argument detected",
                            file_path=report.file_path,
                            line_number=node.lineno,
                            suggestion="Use None as default and initialize mutable object inside function",
                        ))
    
    def _analyze_javascript(self, content: str, lines: list[str], report: AnalysisReport) -> None:
        """Analyze JavaScript/TypeScript code."""
        rules = self.rules.get("javascript", {})
        max_line_length = rules.get("max_line_length", 100)
        
        # Check line lengths
        for i, line in enumerate(lines, 1):
            if len(line) > max_line_length:
                report.issues.append(CodeIssue(
                    severity="warning",
                    category="style",
                    message=f"Line too long ({len(line)} > {max_line_length} characters)",
                    file_path=report.file_path,
                    line_number=i,
                ))
        
        # Check for console.log
        if 'console.log' in content:
            for i, line in enumerate(lines, 1):
                if 'console.log' in line:
                    report.issues.append(CodeIssue(
                        severity="info",
                        category="style",
                        message="console.log statement found",
                        file_path=report.file_path,
                        line_number=i,
                        suggestion="Remove debug logging before production",
                    ))
        
        # Check for eval
        if 'eval(' in content:
            report.issues.append(CodeIssue(
                severity="error",
                category="security",
                message="eval() usage detected",
                file_path=report.file_path,
                line_number=content[:content.index('eval(')].count('\n') + 1,
                suggestion="Avoid eval() due to security risks",
            ))
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity
    
    def _calculate_score(self, report: AnalysisReport) -> float:
        """Calculate quality score (0-100)."""
        score = 100.0
        
        # Deduct for issues
        for issue in report.issues:
            if issue.severity == "error":
                score -= 10
            elif issue.severity == "warning":
                score -= 5
            elif issue.severity == "info":
                score -= 1
        
        return max(0.0, score)
    
    def suggest_refactoring(self, file_path: str) -> list[dict[str, Any]]:
        """Suggest refactoring opportunities."""
        suggestions = []
        report = self.analyze_file(file_path)
        
        # Group issues by category
        complexity_issues = report.get_issues_by_category("complexity")
        style_issues = report.get_issues_by_category("style")
        
        if complexity_issues:
            suggestions.append({
                "type": "extract_functions",
                "description": "Extract complex functions into smaller units",
                "issues": [i.message for i in complexity_issues[:3]],
                "priority": "high",
            })
        
        if style_issues:
            suggestions.append({
                "type": "formatting",
                "description": "Fix code formatting and style issues",
                "issues": [i.message for i in style_issues[:5]],
                "priority": "medium",
            })
        
        return suggestions
