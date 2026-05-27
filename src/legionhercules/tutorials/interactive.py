"""Interactive tutorial system for LEGIONHERCULES."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict, Callable, Awaitable
from datetime import datetime
from enum import Enum
import asyncio

from legionhercules.utils.logging import get_logger

logger = get_logger(__name__)


class TutorialStatus(Enum):
    """Status of a tutorial."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"


class StepStatus(Enum):
    """Status of a tutorial step."""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TutorialStep:
    """A single step in a tutorial."""
    id: str
    title: str
    description: str
    instruction: str
    expected_action: Optional[str] = None
    hint: Optional[str] = None
    validation: Optional[Callable[[str], bool]] = None
    status: StepStatus = StepStatus.PENDING
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "instruction": self.instruction,
            "expected_action": self.expected_action,
            "hint": self.hint,
            "status": self.status.value,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


@dataclass
class Tutorial:
    """A complete tutorial."""
    id: str
    title: str
    description: str
    category: str
    difficulty: str  # beginner, intermediate, advanced
    steps: List[TutorialStep] = field(default_factory=list)
    estimated_time_minutes: int = 10
    prerequisites: List[str] = field(default_factory=list)
    status: TutorialStatus = TutorialStatus.NOT_STARTED
    current_step: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def progress(self) -> float:
        """Calculate completion percentage."""
        if not self.steps:
            return 0.0
        completed = sum(1 for s in self.steps if s.status == StepStatus.COMPLETED)
        return completed / len(self.steps)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "category": self.category,
            "difficulty": self.difficulty,
            "steps": [s.to_dict() for s in self.steps],
            "estimated_time_minutes": self.estimated_time_minutes,
            "prerequisites": self.prerequisites,
            "status": self.status.value,
            "progress": self.progress,
            "current_step": self.current_step,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class TutorialLibrary:
    """Library of available tutorials."""

    def __init__(self):
        self.tutorials: Dict[str, Tutorial] = {}
        self._register_default_tutorials()

    def _register_default_tutorials(self) -> None:
        """Register default tutorials."""
        
        # Getting Started Tutorial
        getting_started = Tutorial(
            id="getting_started",
            title="Getting Started with LEGIONHERCULES",
            description="Learn the basics of using LEGIONHERCULES CLI",
            category="basics",
            difficulty="beginner",
            estimated_time_minutes=5,
            steps=[
                TutorialStep(
                    id="welcome",
                    title="Welcome",
                    description="Introduction to LEGIONHERCULES",
                    instruction="Welcome to LEGIONHERCULES! This tutorial will guide you through the basics.",
                    hint="Press Enter to continue",
                ),
                TutorialStep(
                    id="help_command",
                    title="Getting Help",
                    description="Learn how to get help",
                    instruction="Type 'help' or '?' to see available commands.",
                    expected_action="help",
                    hint="Type: help",
                ),
                TutorialStep(
                    id="list_files",
                    title="Listing Files",
                    description="List files in current directory",
                    instruction="Use the 'ls' or 'list' command to see files in the current directory.",
                    expected_action="ls",
                    hint="Type: ls",
                ),
                TutorialStep(
                    id="create_file",
                    title="Creating Files",
                    description="Create your first file",
                    instruction="Create a new file called 'hello.txt' using the 'create' command.",
                    expected_action="create hello.txt",
                    hint="Type: create hello.txt",
                ),
                TutorialStep(
                    id="read_file",
                    title="Reading Files",
                    description="Read file contents",
                    instruction="Read the contents of the file you just created.",
                    expected_action="read hello.txt",
                    hint="Type: read hello.txt",
                ),
                TutorialStep(
                    id="complete",
                    title="Congratulations!",
                    description="Tutorial complete",
                    instruction="You've completed the Getting Started tutorial! You now know the basics of LEGIONHERCULES.",
                ),
            ],
        )
        self.tutorials[getting_started.id] = getting_started
        
        # Advanced Features Tutorial
        advanced = Tutorial(
            id="advanced_features",
            title="Advanced Features",
            description="Learn about advanced LEGIONHERCULES features",
            category="advanced",
            difficulty="intermediate",
            estimated_time_minutes=15,
            prerequisites=["getting_started"],
            steps=[
                TutorialStep(
                    id="intro",
                    title="Advanced Features",
                    description="Introduction to advanced features",
                    instruction="This tutorial covers advanced features like parallel agents, checkpoints, and workflows.",
                ),
                TutorialStep(
                    id="parallel_agents",
                    title="Parallel Agents",
                    description="Using parallel agents",
                    instruction="Learn how to run multiple agents in parallel for faster task completion.",
                    expected_action="agents parallel",
                    hint="Type: agents parallel",
                ),
                TutorialStep(
                    id="checkpoints",
                    title="Checkpoints",
                    description="Using checkpoints",
                    instruction="Checkpoints let you save and restore your work. Try creating a checkpoint.",
                    expected_action="checkpoint create",
                    hint="Type: checkpoint create my-checkpoint",
                ),
                TutorialStep(
                    id="workflows",
                    title="Workflows",
                    description="Automating with workflows",
                    instruction="Workflows allow you to automate complex tasks. List available workflows.",
                    expected_action="workflow list",
                    hint="Type: workflow list",
                ),
                TutorialStep(
                    id="complete",
                    title="Tutorial Complete",
                    description="Advanced tutorial complete",
                    instruction="You've learned about LEGIONHERCULES advanced features!",
                ),
            ],
        )
        self.tutorials[advanced.id] = advanced
        
        # Natural Language Tutorial
        nlp_tutorial = Tutorial(
            id="natural_language",
            title="Natural Language Commands",
            description="Learn to use natural language with LEGIONHERCULES",
            category="nlp",
            difficulty="beginner",
            estimated_time_minutes=10,
            steps=[
                TutorialStep(
                    id="intro",
                    title="Natural Language",
                    description="Using natural language",
                    instruction="LEGIONHERCULES understands natural language. Try saying 'create a file called test.py'",
                    expected_action="create a file called test.py",
                    hint="Type: create a file called test.py",
                ),
                TutorialStep(
                    id="read_file",
                    title="Reading Files",
                    description="Natural language file reading",
                    instruction="Try reading a file using natural language: 'show me the contents of test.py'",
                    expected_action="show me the contents of test.py",
                    hint="Type: show me the contents of test.py",
                ),
                TutorialStep(
                    id="generate_code",
                    title="Code Generation",
                    description="Generating code with natural language",
                    instruction="Generate code using natural language: 'create a Python function that calculates factorial'",
                    expected_action="create a Python function that calculates factorial",
                    hint="Type: create a Python function that calculates factorial",
                ),
                TutorialStep(
                    id="complete",
                    title="Complete",
                    description="Natural language tutorial complete",
                    instruction="You can now use natural language commands with LEGIONHERCULES!",
                ),
            ],
        )
        self.tutorials[nlp_tutorial.id] = nlp_tutorial

    def get_tutorial(self, tutorial_id: str) -> Optional[Tutorial]:
        """Get a tutorial by ID."""
        return self.tutorials.get(tutorial_id)

    def list_tutorials(
        self,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
    ) -> List[Tutorial]:
        """List available tutorials with optional filtering."""
        tutorials = list(self.tutorials.values())
        
        if category:
            tutorials = [t for t in tutorials if t.category == category]
        
        if difficulty:
            tutorials = [t for t in tutorials if t.difficulty == difficulty]
        
        return tutorials

    def get_categories(self) -> List[str]:
        """Get all tutorial categories."""
        return list(set(t.category for t in self.tutorials.values()))


class TutorialRunner:
    """Runs interactive tutorials."""

    def __init__(self, library: Optional[TutorialLibrary] = None):
        self.library = library or TutorialLibrary()
        self.current_tutorial: Optional[Tutorial] = None
        self.current_step: Optional[TutorialStep] = None
        self.input_handler: Optional[Callable[[str], Awaitable[bool]]] = None
        self.output_handler: Optional[Callable[[str], Awaitable[None]]] = None

    def set_handlers(
        self,
        input_handler: Callable[[str], Awaitable[bool]],
        output_handler: Callable[[str], Awaitable[None]],
    ) -> None:
        """Set input and output handlers."""
        self.input_handler = input_handler
        self.output_handler = output_handler

    async def start_tutorial(self, tutorial_id: str) -> bool:
        """Start a tutorial."""
        tutorial = self.library.get_tutorial(tutorial_id)
        if not tutorial:
            logger.error(f"Tutorial not found: {tutorial_id}")
            return False
        
        self.current_tutorial = tutorial
        tutorial.status = TutorialStatus.IN_PROGRESS
        tutorial.started_at = datetime.now()
        tutorial.current_step = 0
        
        await self._show_step()
        return True

    async def process_input(self, user_input: str) -> bool:
        """Process user input during tutorial."""
        if not self.current_tutorial or not self.current_step:
            return False
        
        step = self.current_step
        
        # Check if input matches expected action
        if step.expected_action:
            if user_input.strip().lower() == step.expected_action.lower():
                await self._complete_current_step()
                return True
            else:
                await self._show_hint()
                return False
        else:
            # No expected action, just advance
            await self._complete_current_step()
            return True

    async def skip_step(self) -> None:
        """Skip current step."""
        if self.current_step:
            self.current_step.status = StepStatus.COMPLETED
            await self._advance_step()

    async def skip_tutorial(self) -> None:
        """Skip entire tutorial."""
        if self.current_tutorial:
            self.current_tutorial.status = TutorialStatus.SKIPPED
            self.current_tutorial = None
            self.current_step = None

    async def _show_step(self) -> None:
        """Display current step."""
        if not self.current_tutorial:
            return
        
        if self.current_tutorial.current_step >= len(self.current_tutorial.steps):
            await self._complete_tutorial()
            return
        
        step = self.current_tutorial.steps[self.current_tutorial.current_step]
        self.current_step = step
        step.status = StepStatus.ACTIVE
        
        if self.output_handler:
            await self.output_handler(f"\n{'='*50}")
            await self.output_handler(f"Step {self.current_tutorial.current_step + 1}/{len(self.current_tutorial.steps)}: {step.title}")
            await self.output_handler(f"{'='*50}")
            await self.output_handler(f"\n{step.description}\n")
            await self.output_handler(f"{step.instruction}")
            
            if step.expected_action:
                await self.output_handler(f"\n[Expected: {step.expected_action}]")

    async def _show_hint(self) -> None:
        """Show hint for current step."""
        if self.current_step and self.current_step.hint and self.output_handler:
            await self.output_handler(f"\n💡 Hint: {self.current_step.hint}")

    async def _complete_current_step(self) -> None:
        """Mark current step as complete and advance."""
        if self.current_step:
            self.current_step.status = StepStatus.COMPLETED
            self.current_step.completed_at = datetime.now()
            
            if self.output_handler:
                await self.output_handler("\n✅ Step completed!")
        
        await self._advance_step()

    async def _advance_step(self) -> None:
        """Advance to next step."""
        if not self.current_tutorial:
            return
        
        self.current_tutorial.current_step += 1
        
        if self.current_tutorial.current_step >= len(self.current_tutorial.steps):
            await self._complete_tutorial()
        else:
            await self._show_step()

    async def _complete_tutorial(self) -> None:
        """Complete the tutorial."""
        if not self.current_tutorial:
            return
        
        self.current_tutorial.status = TutorialStatus.COMPLETED
        self.current_tutorial.completed_at = datetime.now()
        
        if self.output_handler:
            await self.output_handler(f"\n{'='*50}")
            await self.output_handler(f"🎉 Tutorial Complete: {self.current_tutorial.title}")
            await self.output_handler(f"{'='*50}")
            await self.output_handler(f"\nYou've completed all {len(self.current_tutorial.steps)} steps!")
        
        self.current_tutorial = None
        self.current_step = None

    def get_progress(self) -> Optional[Dict[str, Any]]:
        """Get current tutorial progress."""
        if not self.current_tutorial:
            return None
        
        return {
            "tutorial": self.current_tutorial.title,
            "step": self.current_tutorial.current_step + 1,
            "total_steps": len(self.current_tutorial.steps),
            "progress": self.current_tutorial.progress,
        }


class TutorialProgressTracker:
    """Tracks user progress through tutorials."""

    def __init__(self):
        self.completed_tutorials: Dict[str, datetime] = {}
        self.in_progress: Optional[str] = None
        self.step_progress: Dict[str, int] = {}

    def mark_completed(self, tutorial_id: str) -> None:
        """Mark a tutorial as completed."""
        self.completed_tutorials[tutorial_id] = datetime.now()
        if self.in_progress == tutorial_id:
            self.in_progress = None

    def mark_in_progress(self, tutorial_id: str) -> None:
        """Mark a tutorial as in progress."""
        self.in_progress = tutorial_id

    def is_completed(self, tutorial_id: str) -> bool:
        """Check if tutorial is completed."""
        return tutorial_id in self.completed_tutorials

    def get_completed_count(self) -> int:
        """Get number of completed tutorials."""
        return len(self.completed_tutorials)

    def get_stats(self) -> Dict[str, Any]:
        """Get progress statistics."""
        return {
            "completed": len(self.completed_tutorials),
            "in_progress": self.in_progress,
            "completed_tutorials": list(self.completed_tutorials.keys()),
        }
