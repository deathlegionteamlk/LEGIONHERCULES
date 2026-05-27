"""Git workflow automation for LEGIONHERCULES."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from legionhercules.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GitStatus:
    """Git repository status."""
    branch: str
    is_clean: bool
    modified_files: list[str] = field(default_factory=list)
    staged_files: list[str] = field(default_factory=list)
    untracked_files: list[str] = field(default_factory=list)
    ahead: int = 0
    behind: int = 0


@dataclass
class GitCommit:
    """Git commit information."""
    hash: str
    message: str
    author: str
    date: str
    files_changed: list[str] = field(default_factory=list)


class GitWorkflowManager:
    """Manages Git workflows for autonomous development."""
    
    def __init__(self, repo_path: Optional[str] = None):
        """Initialize Git workflow manager.
        
        Args:
            repo_path: Path to git repository (default: current directory)
        """
        self.repo_path = Path(repo_path or ".").resolve()
        self._check_git_repo()
    
    def _check_git_repo(self) -> None:
        """Verify this is a git repository."""
        git_dir = self.repo_path / ".git"
        if not git_dir.exists():
            raise ValueError(f"Not a git repository: {self.repo_path}")
    
    def _run_git(self, args: list[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run a git command."""
        cmd = ["git", "-C", str(self.repo_path)] + args
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=check,
        )
        return result
    
    def get_status(self) -> GitStatus:
        """Get repository status."""
        # Get current branch
        branch_result = self._run_git(["rev-parse", "--abbrev-ref", "HEAD"])
        branch = branch_result.stdout.strip()
        
        # Get status
        status_result = self._run_git(["status", "--porcelain"])
        lines = status_result.stdout.strip().split('\n') if status_result.stdout else []
        
        modified = []
        staged = []
        untracked = []
        
        for line in lines:
            if not line:
                continue
            status = line[:2]
            filename = line[3:]
            
            if status[0] in 'MADRC':
                staged.append(filename)
            if status[1] in 'MD':
                modified.append(filename)
            if status == '??':
                untracked.append(filename)
        
        # Get ahead/behind
        ahead, behind = self._get_ahead_behind(branch)
        
        return GitStatus(
            branch=branch,
            is_clean=len(lines) == 0 or lines == [''],
            modified_files=modified,
            staged_files=staged,
            untracked_files=untracked,
            ahead=ahead,
            behind=behind,
        )
    
    def _get_ahead_behind(self, branch: str) -> tuple[int, int]:
        """Get commits ahead/behind remote."""
        try:
            result = self._run_git(
                ["rev-list", "--left-right", "--count", f"origin/{branch}...{branch}"],
                check=False
            )
            if result.returncode == 0:
                counts = result.stdout.strip().split()
                if len(counts) == 2:
                    return int(counts[1]), int(counts[0])
        except Exception:
            pass
        return 0, 0
    
    def stage_files(self, files: Optional[list[str]] = None) -> bool:
        """Stage files for commit.
        
        Args:
            files: Specific files to stage (None = all modified)
        
        Returns:
            True if successful
        """
        try:
            if files:
                self._run_git(["add"] + files)
            else:
                self._run_git(["add", "."])
            logger.info(f"Staged files: {files or 'all'}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to stage files: {e}")
            return False
    
    def unstage_files(self, files: Optional[list[str]] = None) -> bool:
        """Unstage files.
        
        Args:
            files: Specific files to unstage (None = all)
        """
        try:
            if files:
                self._run_git(["reset", "HEAD"] + files)
            else:
                self._run_git(["reset", "HEAD"])
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to unstage files: {e}")
            return False
    
    def commit(self, message: str, description: Optional[str] = None) -> bool:
        """Create a commit.
        
        Args:
            message: Commit message (required)
            description: Extended description (optional)
        
        Returns:
            True if successful
        """
        try:
            full_message = message
            if description:
                full_message += f"\n\n{description}"
            
            self._run_git(["commit", "-m", full_message])
            logger.info(f"Created commit: {message[:50]}...")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create commit: {e}")
            return False
    
    def create_branch(self, branch_name: str, base: Optional[str] = None) -> bool:
        """Create and checkout a new branch.
        
        Args:
            branch_name: Name for new branch
            base: Base branch/commit (default: current)
        
        Returns:
            True if successful
        """
        try:
            if base:
                self._run_git(["checkout", "-b", branch_name, base])
            else:
                self._run_git(["checkout", "-b", branch_name])
            logger.info(f"Created branch: {branch_name}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create branch: {e}")
            return False
    
    def switch_branch(self, branch_name: str) -> bool:
        """Switch to a branch.
        
        Args:
            branch_name: Branch to checkout
        
        Returns:
            True if successful
        """
        try:
            self._run_git(["checkout", branch_name])
            logger.info(f"Switched to branch: {branch_name}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to switch branch: {e}")
            return False
    
    def pull(self, remote: str = "origin", branch: Optional[str] = None) -> bool:
        """Pull changes from remote.
        
        Args:
            remote: Remote name
            branch: Branch to pull (default: current)
        
        Returns:
            True if successful
        """
        try:
            if branch:
                self._run_git(["pull", remote, branch])
            else:
                self._run_git(["pull", remote])
            logger.info(f"Pulled from {remote}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to pull: {e}")
            return False
    
    def push(self, remote: str = "origin", branch: Optional[str] = None, force: bool = False) -> bool:
        """Push changes to remote.
        
        Args:
            remote: Remote name
            branch: Branch to push (default: current)
            force: Force push
        
        Returns:
            True if successful
        """
        try:
            cmd = ["push", remote]
            if branch:
                cmd.append(branch)
            if force:
                cmd.append("--force-with-lease")
            
            self._run_git(cmd)
            logger.info(f"Pushed to {remote}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to push: {e}")
            return False
    
    def get_diff(self, staged: bool = False) -> str:
        """Get diff of changes.
        
        Args:
            staged: Show staged changes
        
        Returns:
            Diff output
        """
        try:
            if staged:
                result = self._run_git(["diff", "--staged"])
            else:
                result = self._run_git(["diff"])
            return result.stdout
        except subprocess.CalledProcessError:
            return ""
    
    def get_log(self, count: int = 10) -> list[GitCommit]:
        """Get commit history.
        
        Args:
            count: Number of commits to retrieve
        
        Returns:
            List of commits
        """
        try:
            result = self._run_git([
                "log",
                f"-{count}",
                "--pretty=format:%H|%s|%an|%ad",
                "--date=short",
            ])
            
            commits = []
            for line in result.stdout.strip().split('\n'):
                if '|' in line:
                    parts = line.split('|', 3)
                    if len(parts) >= 4:
                        commits.append(GitCommit(
                            hash=parts[0],
                            message=parts[1],
                            author=parts[2],
                            date=parts[3],
                        ))
            
            return commits
        except subprocess.CalledProcessError:
            return []
    
    def stash(self, message: Optional[str] = None) -> bool:
        """Stash current changes.
        
        Args:
            message: Stash message
        
        Returns:
            True if successful
        """
        try:
            cmd = ["stash", "push"]
            if message:
                cmd.extend(["-m", message])
            
            self._run_git(cmd)
            logger.info("Changes stashed")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to stash: {e}")
            return False
    
    def stash_pop(self) -> bool:
        """Pop stashed changes.
        
        Returns:
            True if successful
        """
        try:
            self._run_git(["stash", "pop"])
            logger.info("Stash popped")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to pop stash: {e}")
            return False
    
    def merge(self, branch: str, no_ff: bool = False) -> bool:
        """Merge a branch into current.
        
        Args:
            branch: Branch to merge
            no_ff: Create merge commit even if fast-forward possible
        
        Returns:
            True if successful
        """
        try:
            cmd = ["merge"]
            if no_ff:
                cmd.append("--no-ff")
            cmd.append(branch)
            
            self._run_git(cmd)
            logger.info(f"Merged branch: {branch}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to merge: {e}")
            return False
    
    def get_current_commit(self) -> Optional[str]:
        """Get current commit hash."""
        try:
            result = self._run_git(["rev-parse", "HEAD"])
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None
