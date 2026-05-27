"""Code generation module with sandbox integration."""

from __future__ import annotations

import re
import ast
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict, Callable, Union
from datetime import datetime
from enum import Enum
from pathlib import Path

from legionhercules.codegen.sandbox import CodeSandbox, SandboxResult, SandboxStatus, CodeValidator
from legionhercules.utils.logging import get_logger

logger = get_logger(__name__)


class GenerationStatus(Enum):
    """Status of code generation."""
    PENDING = "pending"
    GENERATING = "generating"
    VALIDATING = "validating"
    EXECUTING = "executing"
    SUCCESS = "success"
    VALIDATION_FAILED = "validation_failed"
    EXECUTION_FAILED = "execution_failed"
    ERROR = "error"


@dataclass
class GenerationResult:
    """Result of code generation and execution."""
    status: GenerationStatus
    code: str = ""
    description: str = ""
    sandbox_result: Optional[SandboxResult] = None
    validation_errors: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    iterations: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "code": self.code[:1000] if len(self.code) > 1000 else self.code,
            "description": self.description,
            "sandbox_result": self.sandbox_result.to_dict() if self.sandbox_result else None,
            "validation_errors": self.validation_errors,
            "execution_time_ms": self.execution_time_ms,
            "iterations": self.iterations,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class CodeTemplate:
    """Template for code generation."""
    name: str
    template: str
    description: str = ""
    required_params: List[str] = field(default_factory=list)

    def render(self, **kwargs) -> str:
        """Render template with parameters."""
        result = self.template
        for key, value in kwargs.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result


class CodeGenerator:
    """Generator for Python code with sandbox execution."""

    # Built-in templates
    TEMPLATES = {
        "function": CodeTemplate(
            name="function",
            template='''def {function_name}({parameters}):
    """{docstring}"""
    {body}
    return {return_value}
''',
            description="Generate a Python function",
            required_params=["function_name", "parameters", "body"],
        ),
        "class": CodeTemplate(
            name="class",
            template='''class {class_name}:
    """{docstring}"""
    
    def __init__(self{init_params}):
        {init_body}
    
    {methods}
''',
            description="Generate a Python class",
            required_params=["class_name"],
        ),
        "script": CodeTemplate(
            name="script",
            template='''#!/usr/bin/env python3
"""{description}"""

{imports}

def main():
    """Main function."""
    {body}

if __name__ == "__main__":
    main()
''',
            description="Generate a Python script",
            required_params=["description", "body"],
        ),
        "data_processing": CodeTemplate(
            name="data_processing",
            template='''import json
import statistics

def process_data(data):
    """Process input data and return statistics."""
    if not data:
        return {{"error": "Empty data"}}
    
    result = {{
        "count": len(data),
        "sum": sum(data),
        "mean": statistics.mean(data),
        "median": statistics.median(data),
        "min": min(data),
        "max": max(data),
    }}
    
    if len(data) > 1:
        result["stdev"] = statistics.stdev(data)
    
    return result

# Example usage
if __name__ == "__main__":
    sample_data = {sample_data}
    print(json.dumps(process_data(sample_data), indent=2))
''',
            description="Generate data processing code",
            required_params=["sample_data"],
        ),
        "algorithm": CodeTemplate(
            name="algorithm",
            template='''def {algorithm_name}(input_data):
    """
    {description}
    
    Args:
        input_data: Input data to process
        
    Returns:
        Processed result
    """
    {implementation}
    return result

# Test cases
if __name__ == "__main__":
    test_cases = {test_cases}
    for i, test in enumerate(test_cases, 1):
        result = {algorithm_name}(test["input"])
        print(f"Test {{i}}: {{'PASS' if result == test['expected'] else 'FAIL'}}")
        print(f"  Input: {{test['input']}}")
        print(f"  Expected: {{test['expected']}}, Got: {{result}}")
''',
            description="Generate algorithm implementation",
            required_params=["algorithm_name", "description", "implementation"],
        ),
    }

    def __init__(
        self,
        sandbox_timeout: int = 30,
        sandbox_memory_mb: int = 256,
        max_iterations: int = 3,
    ):
        self.sandbox_timeout = sandbox_timeout
        self.sandbox_memory_mb = sandbox_memory_mb
        self.max_iterations = max_iterations
        self._sandbox: Optional[CodeSandbox] = None

    async def _get_sandbox(self) -> CodeSandbox:
        """Get or create sandbox instance."""
        if self._sandbox is None:
            self._sandbox = CodeSandbox(
                timeout_seconds=self.sandbox_timeout,
                memory_limit_mb=self.sandbox_memory_mb,
            )
        return self._sandbox

    async def generate_from_template(
        self,
        template_name: str,
        validate: bool = True,
        execute: bool = False,
        **template_params,
    ) -> GenerationResult:
        """Generate code from a template."""
        start_time = datetime.now()
        
        if template_name not in self.TEMPLATES:
            return GenerationResult(
                status=GenerationStatus.ERROR,
                description=f"Template '{template_name}' not found",
                validation_errors=[f"Available templates: {list(self.TEMPLATES.keys())}"],
            )

        template = self.TEMPLATES[template_name]
        
        # Check required parameters
        missing_params = [
            param for param in template.required_params
            if param not in template_params
        ]
        if missing_params:
            return GenerationResult(
                status=GenerationStatus.ERROR,
                description=f"Missing required parameters: {missing_params}",
                validation_errors=[f"Required: {template.required_params}"],
            )

        # Generate code
        code = template.render(**template_params)
        
        result = GenerationResult(
            status=GenerationStatus.GENERATING,
            code=code,
            description=f"Generated from template: {template_name}",
        )

        # Validate if requested
        if validate:
            result = await self._validate_code(result)

        # Execute if requested and validation passed
        if execute and result.status not in [
            GenerationStatus.VALIDATION_FAILED,
            GenerationStatus.ERROR,
        ]:
            result = await self._execute_code(result)

        result.execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        return result

    async def generate_custom(
        self,
        code: str,
        description: str = "",
        validate: bool = True,
        execute: bool = False,
        test_cases: Optional[List[Dict[str, Any]]] = None,
    ) -> GenerationResult:
        """Generate/validate custom code."""
        start_time = datetime.now()
        
        result = GenerationResult(
            status=GenerationStatus.GENERATING,
            code=code,
            description=description or "Custom code",
        )

        # Validate if requested
        if validate:
            result = await self._validate_code(result)

        # Execute if requested
        if execute and result.status not in [
            GenerationStatus.VALIDATION_FAILED,
            GenerationStatus.ERROR,
        ]:
            if test_cases:
                result = await self._test_code(result, test_cases)
            else:
                result = await self._execute_code(result)

        result.execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        return result

    async def _validate_code(self, result: GenerationResult) -> GenerationResult:
        """Validate generated code."""
        result.status = GenerationStatus.VALIDATING
        
        is_valid, errors = CodeValidator.full_validate(result.code)
        
        if not is_valid:
            result.status = GenerationStatus.VALIDATION_FAILED
            result.validation_errors = errors
            logger.warning(f"Code validation failed: {errors}")
        else:
            result.status = GenerationStatus.SUCCESS
            logger.info("Code validation passed")
        
        return result

    async def _execute_code(self, result: GenerationResult) -> GenerationResult:
        """Execute code in sandbox."""
        result.status = GenerationStatus.EXECUTING
        
        sandbox = await self._get_sandbox()
        sandbox_result = await sandbox.execute(result.code)
        
        result.sandbox_result = sandbox_result
        
        if sandbox_result.status == SandboxStatus.SUCCESS:
            result.status = GenerationStatus.SUCCESS
            logger.info("Code execution succeeded")
        elif sandbox_result.status == SandboxStatus.SECURITY_VIOLATION:
            result.status = GenerationStatus.VALIDATION_FAILED
            result.validation_errors.append(f"Security: {sandbox_result.error_message}")
            logger.warning(f"Security violation: {sandbox_result.error_message}")
        else:
            result.status = GenerationStatus.EXECUTION_FAILED
            logger.warning(f"Execution failed: {sandbox_result.error_message}")
        
        return result

    async def _test_code(
        self,
        result: GenerationResult,
        test_cases: List[Dict[str, Any]],
    ) -> GenerationResult:
        """Test code against test cases."""
        result.status = GenerationStatus.EXECUTING
        
        sandbox = await self._get_sandbox()
        test_results = await sandbox.test_code(result.code, test_cases)
        
        result.metadata["test_results"] = test_results
        
        if test_results["passed"] == test_results["total"]:
            result.status = GenerationStatus.SUCCESS
            logger.info(f"All {test_results['total']} tests passed")
        else:
            result.status = GenerationStatus.EXECUTION_FAILED
            result.validation_errors.append(
                f"Tests: {test_results['passed']}/{test_results['total']} passed"
            )
            logger.warning(
                f"Tests failed: {test_results['passed']}/{test_results['total']}"
            )
        
        return result

    async def generate_with_iteration(
        self,
        generator_func: Callable[[], str],
        validator_func: Optional[Callable[[str], tuple[bool, str]]] = None,
        max_iterations: Optional[int] = None,
    ) -> GenerationResult:
        """Generate code with iterative refinement."""
        max_iter = max_iterations or self.max_iterations
        start_time = datetime.now()
        
        for iteration in range(max_iter):
            # Generate code
            code = generator_func()
            
            result = GenerationResult(
                status=GenerationStatus.GENERATING,
                code=code,
                description=f"Iteration {iteration + 1}/{max_iter}",
                iterations=iteration + 1,
            )
            
            # Validate
            result = await self._validate_code(result)
            
            if result.status == GenerationStatus.VALIDATION_FAILED:
                if iteration < max_iter - 1:
                    logger.info(f"Validation failed, retrying...")
                    continue
            
            # Custom validation
            if validator_func and result.status == GenerationStatus.SUCCESS:
                is_valid, message = validator_func(code)
                if not is_valid:
                    result.status = GenerationStatus.VALIDATION_FAILED
                    result.validation_errors.append(message)
                    if iteration < max_iter - 1:
                        logger.info(f"Custom validation failed: {message}, retrying...")
                        continue
            
            if result.status == GenerationStatus.SUCCESS:
                break
        
        result.execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        return result

    async def execute_function(
        self,
        code: str,
        function_name: str,
        args: List[Any] = None,
        kwargs: Dict[str, Any] = None,
    ) -> SandboxResult:
        """Execute a specific function from code."""
        sandbox = await self._get_sandbox()
        return await sandbox.execute_function(code, function_name, args, kwargs)

    def get_template_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available templates."""
        return {
            name: {
                "description": template.description,
                "required_params": template.required_params,
            }
            for name, template in self.TEMPLATES.items()
        }

    async def cleanup(self) -> None:
        """Clean up sandbox resources."""
        if self._sandbox:
            self._sandbox.cleanup()
            self._sandbox = None


class CodeRefiner:
    """Refines code based on execution results."""

    COMMON_FIXES = [
        # Fix common syntax errors
        (r"print\s+(\w+)", r"print(\1)"),  # Python 2 style print
        (r"except\s+(\w+),\s*(\w+):", r"except \1 as \2:"),  # Old except syntax
    ]

    @staticmethod
    def apply_syntax_fixes(code: str) -> str:
        """Apply common syntax fixes."""
        for pattern, replacement in CodeRefiner.COMMON_FIXES:
            code = re.sub(pattern, replacement, code)
        return code

    @staticmethod
    def extract_code_from_markdown(text: str) -> str:
        """Extract code from markdown code blocks."""
        # Look for python code blocks
        python_pattern = r"```python\n(.*?)\n```"
        matches = re.findall(python_pattern, text, re.DOTALL)
        if matches:
            return matches[0]
        
        # Look for any code blocks
        generic_pattern = r"```\n(.*?)\n```"
        matches = re.findall(generic_pattern, text, re.DOTALL)
        if matches:
            return matches[0]
        
        return text

    @staticmethod
    def add_error_handling(code: str, function_name: Optional[str] = None) -> str:
        """Add error handling to code."""
        if function_name:
            # Wrap specific function with try-except
            pattern = rf"(def {function_name}\([^)]*\):\n(?:\s+\"\"\"[^\"]*\"\"\"\n)?)"
            replacement = r"\1    try:\n        "
            # This is a simplified version - full implementation would parse AST
            return code
        else:
            # Wrap entire code in try-except
            return f"""try:
    {code.replace(chr(10), chr(10) + '    ')}
except Exception as e:
    print(f"Error: {{e}}")
    raise
"""


# Convenience functions
async def generate_function(
    function_name: str,
    parameters: str,
    body: str,
    docstring: str = "",
    return_value: str = "None",
    execute: bool = False,
) -> GenerationResult:
    """Quick function to generate a Python function."""
    generator = CodeGenerator()
    return await generator.generate_from_template(
        template_name="function",
        function_name=function_name,
        parameters=parameters,
        body=body,
        docstring=docstring,
        return_value=return_value,
        validate=True,
        execute=execute,
    )


async def generate_script(
    description: str,
    body: str,
    imports: str = "",
    execute: bool = False,
) -> GenerationResult:
    """Quick function to generate a Python script."""
    generator = CodeGenerator()
    return await generator.generate_from_template(
        template_name="script",
        description=description,
        body=body,
        imports=imports,
        validate=True,
        execute=execute,
    )
