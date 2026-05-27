"""Natural language to command translation."""

from __future__ import annotations

import re
import json
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict, Callable
from datetime import datetime
from enum import Enum

from legionhercules.utils.logging import get_logger

logger = get_logger(__name__)


class CommandType(Enum):
    """Types of commands."""
    FILE_OPERATION = "file_operation"
    CODE_GENERATION = "code_generation"
    EXECUTION = "execution"
    RESEARCH = "research"
    ANALYSIS = "analysis"
    NAVIGATION = "navigation"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


@dataclass
class ParsedCommand:
    """A parsed command from natural language."""
    original_text: str
    command_type: CommandType
    action: str
    target: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_text": self.original_text,
            "command_type": self.command_type.value,
            "action": self.action,
            "target": self.target,
            "parameters": self.parameters,
            "confidence": self.confidence,
            "alternatives": self.alternatives,
        }


class CommandPattern:
    """Pattern for matching commands."""
    
    def __init__(
        self,
        name: str,
        command_type: CommandType,
        patterns: List[str],
        action: str,
        extract_params: Optional[Callable[[re.Match], Dict[str, Any]]] = None,
    ):
        self.name = name
        self.command_type = command_type
        self.patterns = [re.compile(p, re.IGNORECASE) for p in patterns]
        self.action = action
        self.extract_params = extract_params or self._default_extract

    def match(self, text: str) -> Optional[ParsedCommand]:
        """Match text against patterns."""
        for pattern in self.patterns:
            match = pattern.search(text)
            if match:
                params = self.extract_params(match)
                return ParsedCommand(
                    original_text=text,
                    command_type=self.command_type,
                    action=self.action,
                    target=params.get("target"),
                    parameters=params,
                    confidence=match.end() - match.start() / len(text),
                )
        return None

    def _default_extract(self, match: re.Match) -> Dict[str, Any]:
        """Default parameter extraction."""
        return match.groupdict()


class NLTranslator:
    """Natural language to command translator."""

    def __init__(self, llm_provider=None):
        self.llm_provider = llm_provider
        self.patterns: List[CommandPattern] = []
        self._register_default_patterns()

    def _register_default_patterns(self) -> None:
        """Register default command patterns."""
        
        # File operations
        self.patterns.append(CommandPattern(
            name="create_file",
            command_type=CommandType.FILE_OPERATION,
            patterns=[
                r"create\s+(?:a\s+)?(?:new\s+)?file\s+(?:called\s+)?['\"]?([^'\"]+)['\"]?",
                r"make\s+(?:a\s+)?(?:new\s+)?file\s+(?:called\s+)?['\"]?([^'\"]+)['\"]?",
                r"touch\s+['\"]?([^'\"]+)['\"]?",
            ],
            action="create_file",
        ))
        
        self.patterns.append(CommandPattern(
            name="read_file",
            command_type=CommandType.FILE_OPERATION,
            patterns=[
                r"(?:show|read|display|view)\s+(?:me\s+)?(?:the\s+)?(?:content(?:s)?\s+of\s+)?(?:file\s+)?['\"]?([^'\"]+)['\"]?",
                r"what['\']?s?\s+in\s+(?:the\s+)?file\s+['\"]?([^'\"]+)['\"]?",
            ],
            action="read_file",
        ))
        
        self.patterns.append(CommandPattern(
            name="edit_file",
            command_type=CommandType.FILE_OPERATION,
            patterns=[
                r"edit\s+(?:the\s+)?file\s+['\"]?([^'\"]+)['\"]?",
                r"modify\s+(?:the\s+)?file\s+['\"]?([^'\"]+)['\"]?",
                r"update\s+(?:the\s+)?file\s+['\"]?([^'\"]+)['\"]?",
            ],
            action="edit_file",
        ))
        
        self.patterns.append(CommandPattern(
            name="delete_file",
            command_type=CommandType.FILE_OPERATION,
            patterns=[
                r"delete\s+(?:the\s+)?file\s+['\"]?([^'\"]+)['\"]?",
                r"remove\s+(?:the\s+)?file\s+['\"]?([^'\"]+)['\"]?",
                r"rm\s+['\"]?([^'\"]+)['\"]?",
            ],
            action="delete_file",
        ))
        
        # Code generation
        self.patterns.append(CommandPattern(
            name="generate_code",
            command_type=CommandType.CODE_GENERATION,
            patterns=[
                r"(?:generate|create|write)\s+(?:a\s+)?(\w+)\s+(?:function|class|method)",
                r"(?:generate|create|write)\s+code\s+(?:to\s+)?(.+)",
                r"implement\s+(?:a\s+)?(\w+)",
            ],
            action="generate_code",
        ))
        
        self.patterns.append(CommandPattern(
            name="refactor_code",
            command_type=CommandType.CODE_GENERATION,
            patterns=[
                r"refactor\s+(?:the\s+)?(?:code\s+in\s+)?['\"]?([^'\"]+)['\"]?",
                r"improve\s+(?:the\s+)?(?:code\s+in\s+)?['\"]?([^'\"]+)['\"]?",
                r"optimize\s+(?:the\s+)?(?:code\s+in\s+)?['\"]?([^'\"]+)['\"]?",
            ],
            action="refactor_code",
        ))
        
        # Execution
        self.patterns.append(CommandPattern(
            name="run_command",
            command_type=CommandType.EXECUTION,
            patterns=[
                r"run\s+(?:the\s+)?(?:command\s+)?['\"]?([^'\"]+)['\"]?",
                r"execute\s+(?:the\s+)?(?:command\s+)?['\"]?([^'\"]+)['\"]?",
                r"(?:run|execute)\s+(?:the\s+)?tests",
                r"(?:run|execute)\s+(?:the\s+)?script\s+['\"]?([^'\"]+)['\"]?",
            ],
            action="run_command",
        ))
        
        # Research
        self.patterns.append(CommandPattern(
            name="research",
            command_type=CommandType.RESEARCH,
            patterns=[
                r"(?:research|search|find|look\s+up)\s+(?:information\s+)?(?:about\s+)?(.+)",
                r"what\s+(?:is|are)\s+(.+)",
                r"how\s+(?:do|does|can|to)\s+(.+)",
            ],
            action="research",
        ))
        
        # Analysis
        self.patterns.append(CommandPattern(
            name="analyze_code",
            command_type=CommandType.ANALYSIS,
            patterns=[
                r"analyze\s+(?:the\s+)?(?:code\s+in\s+)?['\"]?([^'\"]+)['\"]?",
                r"review\s+(?:the\s+)?(?:code\s+in\s+)?['\"]?([^'\"]+)['\"]?",
                r"check\s+(?:the\s+)?(?:code\s+in\s+)?['\"]?([^'\"]+)['\"]?",
            ],
            action="analyze_code",
        ))
        
        # Navigation
        self.patterns.append(CommandPattern(
            name="change_directory",
            command_type=CommandType.NAVIGATION,
            patterns=[
                r"(?:go|change|cd)\s+(?:to\s+)?(?:the\s+)?(?:directory\s+)?['\"]?([^'\"]+)['\"]?",
                r"navigate\s+(?:to\s+)?['\"]?([^'\"]+)['\"]?",
            ],
            action="change_directory",
        ))
        
        self.patterns.append(CommandPattern(
            name="list_files",
            command_type=CommandType.NAVIGATION,
            patterns=[
                r"(?:list|show|ls)\s+(?:the\s+)?files",
                r"what['\']?s?\s+in\s+(?:this\s+)?(?:directory|folder)",
                r"show\s+(?:me\s+)?(?:the\s+)?(?:directory\s+)?contents",
            ],
            action="list_files",
        ))
        
        # Configuration
        self.patterns.append(CommandPattern(
            name="set_config",
            command_type=CommandType.CONFIGURATION,
            patterns=[
                r"set\s+(?:the\s+)?(?:config|setting|configuration)\s+['\"]?([^'\"]+)['\"]?\s+to\s+['\"]?([^'\"]+)['\"]?",
                r"configure\s+['\"]?([^'\"]+)['\"]?\s+as\s+['\"]?([^'\"]+)['\"]?",
            ],
            action="set_config",
        ))

    def translate(self, text: str) -> ParsedCommand:
        """Translate natural language to command."""
        text = text.strip()
        
        # Try pattern matching first
        for pattern in self.patterns:
            result = pattern.match(text)
            if result and result.confidence > 0.5:
                logger.info(f"Matched pattern '{pattern.name}' with confidence {result.confidence:.2f}")
                return result
        
        # Fall back to LLM if available
        if self.llm_provider:
            return self._translate_with_llm(text)
        
        # Return unknown command
        return ParsedCommand(
            original_text=text,
            command_type=CommandType.UNKNOWN,
            action="unknown",
            confidence=0.0,
        )

    def _translate_with_llm(self, text: str) -> ParsedCommand:
        """Use LLM to translate natural language."""
        prompt = f"""Parse this natural language command into a structured format:

Input: "{text}"

Respond with JSON in this format:
{{
    "command_type": "file_operation|code_generation|execution|research|analysis|navigation|configuration|unknown",
    "action": "specific_action_name",
    "target": "target_file_or_object",
    "parameters": {{}}
}}

Only respond with the JSON, no other text."""
        
        try:
            response = self.llm_provider.generate(prompt)
            parsed = json.loads(response.content)
            
            return ParsedCommand(
                original_text=text,
                command_type=CommandType(parsed.get("command_type", "unknown")),
                action=parsed.get("action", "unknown"),
                target=parsed.get("target"),
                parameters=parsed.get("parameters", {}),
                confidence=0.8,
            )
        except Exception as e:
            logger.error(f"LLM translation failed: {e}")
            return ParsedCommand(
                original_text=text,
                command_type=CommandType.UNKNOWN,
                action="unknown",
                confidence=0.0,
            )

    def get_suggestions(self, partial_text: str) -> List[str]:
        """Get command suggestions for partial input."""
        suggestions = []
        partial_lower = partial_text.lower()
        
        # Common command starters
        starters = [
            ("create a file called ", CommandType.FILE_OPERATION),
            ("read file ", CommandType.FILE_OPERATION),
            ("edit file ", CommandType.FILE_OPERATION),
            ("delete file ", CommandType.FILE_OPERATION),
            ("generate a ", CommandType.CODE_GENERATION),
            ("run command ", CommandType.EXECUTION),
            ("execute ", CommandType.EXECUTION),
            ("research ", CommandType.RESEARCH),
            ("analyze ", CommandType.ANALYSIS),
            ("go to ", CommandType.NAVIGATION),
            ("list files", CommandType.NAVIGATION),
            ("set config ", CommandType.CONFIGURATION),
        ]
        
        for starter, cmd_type in starters:
            if starter.startswith(partial_lower) or partial_lower in starter:
                suggestions.append(starter.strip())
        
        return suggestions[:5]

    def explain_command(self, command: ParsedCommand) -> str:
        """Generate human-readable explanation of a command."""
        explanations = {
            "create_file": f"Create a new file named '{command.target}'",
            "read_file": f"Read and display the contents of '{command.target}'",
            "edit_file": f"Edit the file '{command.target}'",
            "delete_file": f"Delete the file '{command.target}'",
            "generate_code": f"Generate code: {command.parameters.get('description', command.target)}",
            "refactor_code": f"Refactor code in '{command.target}'",
            "run_command": f"Execute command: {command.target or command.parameters.get('command', 'unknown')}",
            "research": f"Research: {command.target}",
            "analyze_code": f"Analyze code in '{command.target}'",
            "change_directory": f"Change to directory '{command.target}'",
            "list_files": "List files in current directory",
            "set_config": f"Set configuration '{command.target}' to '{command.parameters.get('value')}'",
        }
        
        return explanations.get(command.action, f"Execute: {command.action}")


class CommandHistory:
    """Tracks command history for learning."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.commands: List[ParsedCommand] = []
        self.success_rates: Dict[str, Dict[str, Any]] = {}

    def add(self, command: ParsedCommand, success: bool = True) -> None:
        """Add a command to history."""
        self.commands.append(command)
        
        if len(self.commands) > self.max_size:
            self.commands = self.commands[-self.max_size:]
        
        # Track success rate
        key = f"{command.command_type.value}:{command.action}"
        if key not in self.success_rates:
            self.success_rates[key] = {"total": 0, "success": 0}
        
        self.success_rates[key]["total"] += 1
        if success:
            self.success_rates[key]["success"] += 1

    def get_frequent_commands(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get most frequently used commands."""
        from collections import Counter
        
        counts = Counter(f"{c.command_type.value}:{c.action}" for c in self.commands)
        
        return [
            {
                "command": cmd,
                "count": count,
                "success_rate": self.success_rates.get(cmd, {}).get("success", 0) / 
                               self.success_rates.get(cmd, {}).get("total", 1),
            }
            for cmd, count in counts.most_common(n)
        ]

    def get_similar_commands(self, text: str, n: int = 5) -> List[ParsedCommand]:
        """Get commands similar to given text."""
        # Simple similarity based on common words
        text_words = set(text.lower().split())
        
        scored = []
        for cmd in self.commands:
            cmd_words = set(cmd.original_text.lower().split())
            score = len(text_words & cmd_words) / len(text_words | cmd_words)
            scored.append((score, cmd))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [cmd for _, cmd in scored[:n]]
