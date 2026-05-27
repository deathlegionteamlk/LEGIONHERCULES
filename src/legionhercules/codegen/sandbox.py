"""Safe execution sandbox for generated code."""

from __future__ import annotations

import ast
import sys
import subprocess
import tempfile
import os
import resource
import signal
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict, Callable
from datetime import datetime
from enum import Enum

from legionhercules.utils.logging import get_logger

logger = get_logger(__name__)


class SandboxStatus(Enum):
    """Status of sandbox execution."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    MEMORY_EXCEEDED = "memory_exceeded"
    SECURITY_VIOLATION = "security_violation"


@dataclass
class SandboxResult:
    """Result of sandbox execution."""
    status: SandboxStatus
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    execution_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    output_files: Dict[str, str] = field(default_factory=dict)
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "return_code": self.return_code,
            "execution_time_ms": self.execution_time_ms,
            "memory_usage_mb": self.memory_usage_mb,
            "output_files": self.output_files,
            "error_message": self.error_message,
            "timestamp": self.timestamp.isoformat(),
        }


class SecurityChecker:
    """Checks code for security violations."""

    # Dangerous imports and functions
    DANGEROUS_IMPORTS = {
        'os.system', 'os.popen', 'os.exec', 'os.spawn',
        'subprocess.call', 'subprocess.run', 'subprocess.Popen',
        'eval', 'exec', 'compile',
        '__import__', 'importlib',
        'open', 'file',
        'shutil.rmtree', 'shutil.move',
        'socket', 'urllib.request', 'http.client',
        'pickle.loads', 'yaml.load',
    }

    DANGEROUS_BUILTINS = {
        'eval', 'exec', 'compile', '__import__',
        'open', 'file',
    }

    ALLOWED_MODULES = {
        'math', 'random', 'datetime', 'json', 're', 'string',
        'collections', 'itertools', 'functools', 'statistics',
        'typing', 'dataclasses', 'enum', 'pathlib', 'hashlib',
        'base64', 'binascii', 'uuid', 'copy', 'pprint', 'time', 'sys',
    }

    @classmethod
    def check_code(cls, code: str) -> Optional[str]:
        """Check code for security issues. Returns error message if unsafe."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return f"Syntax error: {e}"

        for node in ast.walk(tree):
            # Check for dangerous imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in cls.ALLOWED_MODULES:
                        return f"Import of '{alias.name}' is not allowed"

            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module not in cls.ALLOWED_MODULES:
                    return f"Import from '{node.module}' is not allowed"

            # Check for dangerous calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in cls.DANGEROUS_BUILTINS:
                        return f"Use of '{node.func.id}' is not allowed"

                elif isinstance(node.func, ast.Attribute):
                    # Check for dangerous module.function patterns
                    full_name = cls._get_full_name(node.func)
                    if full_name in cls.DANGEROUS_IMPORTS:
                        return f"Use of '{full_name}' is not allowed"

            # Check for file operations
            elif isinstance(node, ast.With):
                for item in node.items:
                    if isinstance(item.context_expr, ast.Call):
                        func = item.context_expr.func
                        if isinstance(func, ast.Name) and func.id == 'open':
                            return "File operations with 'open' are not allowed in sandbox"

        return None

    @classmethod
    def _get_full_name(cls, node: ast.Attribute) -> str:
        """Get full dotted name for an attribute access."""
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return '.'.join(reversed(parts))


class CodeSandbox:
    """Sandbox for safe code execution."""

    def __init__(
        self,
        timeout_seconds: int = 30,
        memory_limit_mb: int = 256,
        allowed_modules: Optional[List[str]] = None,
        working_directory: Optional[str] = None,
    ):
        self.timeout_seconds = timeout_seconds
        self.memory_limit_mb = memory_limit_mb
        self.allowed_modules = allowed_modules or []
        self.working_directory = working_directory or tempfile.mkdtemp(prefix="legion_sandbox_")
        Path(self.working_directory).mkdir(parents=True, exist_ok=True)

    async def execute(
        self,
        code: str,
        input_data: Optional[str] = None,
        capture_output: bool = True,
    ) -> SandboxResult:
        """Execute code in sandbox."""
        import time
        start_time = time.time()

        # Security check
        security_error = SecurityChecker.check_code(code)
        if security_error:
            return SandboxResult(
                status=SandboxStatus.SECURITY_VIOLATION,
                error_message=security_error,
                stderr=security_error,
            )

        # Write code to temp file
        code_file = Path(self.working_directory) / "script.py"
        code_file.write_text(code)

        # Prepare input
        input_file = None
        if input_data:
            input_file = Path(self.working_directory) / "input.txt"
            input_file.write_text(input_data)

        try:
            # Run in subprocess with limits
            cmd = [sys.executable, str(code_file)]

            # Set resource limits
            def preexec():
                # Memory limit
                resource.setrlimit(resource.RLIMIT_AS, (
                    self.memory_limit_mb * 1024 * 1024,
                    self.memory_limit_mb * 1024 * 1024
                ))
                # CPU time limit
                resource.setrlimit(resource.RLIMIT_CPU, (
                    self.timeout_seconds,
                    self.timeout_seconds + 1
                ))

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE if capture_output else None,
                stderr=subprocess.PIPE if capture_output else None,
                stdin=subprocess.PIPE if input_file else None,
                cwd=self.working_directory,
                preexec_fn=preexec,
            )

            # Wait with timeout
            try:
                stdout, stderr = process.communicate(
                    input=input_data.encode() if input_data else None,
                    timeout=self.timeout_seconds,
                )

                execution_time = (time.time() - start_time) * 1000

                # Check return code
                if process.returncode == 0:
                    status = SandboxStatus.SUCCESS
                elif process.returncode == -signal.SIGKILL:
                    status = SandboxStatus.MEMORY_EXCEEDED
                else:
                    status = SandboxStatus.ERROR

                return SandboxResult(
                    status=status,
                    stdout=stdout.decode() if stdout else "",
                    stderr=stderr.decode() if stderr else "",
                    return_code=process.returncode,
                    execution_time_ms=execution_time,
                )

            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                    process.wait(timeout=1)
                except:
                    pass
                return SandboxResult(
                    status=SandboxStatus.TIMEOUT,
                    error_message=f"Execution timed out after {self.timeout_seconds}s",
                    execution_time_ms=self.timeout_seconds * 1000,
                )

        except Exception as e:
            return SandboxResult(
                status=SandboxStatus.ERROR,
                error_message=str(e),
                stderr=str(e),
            )

    async def execute_function(
        self,
        code: str,
        function_name: str,
        args: List[Any] = None,
        kwargs: Dict[str, Any] = None,
    ) -> SandboxResult:
        """Execute a specific function from code."""
        # Wrap code to call specific function
        wrapper_code = f"""{code}

# Call the function and print result
if __name__ == "__main__":
    import json
    result = {function_name}(*{args or []}, **{kwargs or {}})
    print(json.dumps({{"result": result}}))
"""
        return await self.execute(wrapper_code)

    async def test_code(
        self,
        code: str,
        test_cases: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Test code against multiple test cases."""
        results = []
        passed = 0
        failed = 0

        for i, test in enumerate(test_cases):
            test_input = test.get("input", "")
            expected_output = test.get("expected_output", "")

            # Wrap code to replace input() with sys.stdin
            wrapped_code = f"""import sys

# Mock input() function
def input(prompt=''):
    return sys.stdin.readline().strip()

{code}
"""
            result = await self.execute(wrapped_code, input_data=test_input)

            test_result = {
                "test_case": i + 1,
                "passed": result.status == SandboxStatus.SUCCESS and expected_output.strip() in result.stdout.strip(),
                "status": result.status.value,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

            if test_result["passed"]:
                passed += 1
            else:
                failed += 1

            results.append(test_result)

        return {
            "total": len(test_cases),
            "passed": passed,
            "failed": failed,
            "results": results,
        }

    def cleanup(self) -> None:
        """Clean up sandbox directory."""
        import shutil
        try:
            shutil.rmtree(self.working_directory, ignore_errors=True)
            logger.info(f"Cleaned up sandbox: {self.working_directory}")
        except Exception as e:
            logger.error(f"Failed to cleanup sandbox: {e}")


class CodeValidator:
    """Validates generated code for correctness and safety."""

    @staticmethod
    def validate_syntax(code: str) -> tuple[bool, Optional[str]]:
        """Validate Python syntax."""
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"

    @staticmethod
    def validate_imports(code: str, allowed: Optional[List[str]] = None) -> tuple[bool, Optional[str]]:
        """Validate imports are allowed."""
        error = SecurityChecker.check_code(code)
        if error:
            return False, error
        return True, None

    @staticmethod
    def validate_structure(code: str, required_functions: Optional[List[str]] = None) -> tuple[bool, Optional[str]]:
        """Validate code structure."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, str(e)

        if required_functions:
            found_functions = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    found_functions.add(node.name)

            missing = set(required_functions) - found_functions
            if missing:
                return False, f"Missing required functions: {missing}"

        return True, None

    @classmethod
    def full_validate(
        cls,
        code: str,
        allowed_imports: Optional[List[str]] = None,
        required_functions: Optional[List[str]] = None,
    ) -> tuple[bool, List[str]]:
        """Run all validations."""
        errors = []

        # Syntax validation
        valid, error = cls.validate_syntax(code)
        if not valid:
            errors.append(f"Syntax: {error}")

        # Import validation
        valid, error = cls.validate_imports(code, allowed_imports)
        if not valid:
            errors.append(f"Imports: {error}")

        # Structure validation
        valid, error = cls.validate_structure(code, required_functions)
        if not valid:
            errors.append(f"Structure: {error}")

        return len(errors) == 0, errors
