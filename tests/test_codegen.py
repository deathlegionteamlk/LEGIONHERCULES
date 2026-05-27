"""Test suite for code generation module."""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from legionhercules.codegen import (
    CodeGenerator,
    CodeSandbox,
    GenerationStatus,
    SandboxStatus,
    CodeValidator,
    SecurityChecker,
    generate_function,
    generate_script,
)


async def test_code_generator_template():
    """Test code generation from template."""
    print("\n[Test] CodeGenerator template generation...")
    
    gen = CodeGenerator(sandbox_timeout=5, sandbox_memory_mb=64)
    
    result = await gen.generate_from_template(
        template_name='function',
        function_name='multiply',
        parameters='a, b',
        body='result = a * b',
        docstring='Multiply two numbers',
        return_value='result',
        validate=True,
        execute=False
    )
    
    assert result.status == GenerationStatus.SUCCESS
    assert 'def multiply(a, b):' in result.code
    assert 'result = a * b' in result.code
    
    print(f"✓ Template generation: {result.status.value}")
    await gen.cleanup()
    return True


async def test_code_generator_execution():
    """Test code generation with execution."""
    print("\n[Test] CodeGenerator with execution...")
    
    gen = CodeGenerator(sandbox_timeout=5, sandbox_memory_mb=64)
    
    result = await gen.generate_from_template(
        template_name='function',
        function_name='greet',
        parameters='name',
        body='message = f"Hello, {name}!"\n    print(message)',
        docstring='Greet someone',
        return_value='None',
        validate=True,
        execute=True
    )
    
    assert result.status == GenerationStatus.SUCCESS
    assert result.sandbox_result is not None
    assert result.sandbox_result.status == SandboxStatus.SUCCESS
    
    print(f"✓ Code generation with execution: {result.status.value}")
    await gen.cleanup()
    return True


async def test_code_generator_custom():
    """Test custom code generation."""
    print("\n[Test] CodeGenerator custom code...")
    
    gen = CodeGenerator(sandbox_timeout=5, sandbox_memory_mb=64)
    
    code = '''
import math
radius = 5
area = math.pi * radius ** 2
print(f"Area: {area:.2f}")
'''
    
    result = await gen.generate_custom(
        code=code,
        description="Calculate circle area",
        validate=True,
        execute=True
    )
    
    assert result.status == GenerationStatus.SUCCESS
    assert result.sandbox_result is not None
    assert "Area:" in result.sandbox_result.stdout
    
    print(f"✓ Custom code generation: {result.status.value}")
    await gen.cleanup()
    return True


async def test_code_generator_validation_failure():
    """Test code generator with invalid code."""
    print("\n[Test] CodeGenerator validation failure...")
    
    gen = CodeGenerator(sandbox_timeout=5, sandbox_memory_mb=64)
    
    # Code with syntax error
    code = 'def broken(\n    print("missing parenthesis")'
    
    result = await gen.generate_custom(
        code=code,
        validate=True,
        execute=False
    )
    
    assert result.status == GenerationStatus.VALIDATION_FAILED
    assert len(result.validation_errors) > 0
    
    print(f"✓ Validation failure detected: {result.status.value}")
    await gen.cleanup()
    return True


async def test_code_generator_security():
    """Test code generator security blocking."""
    print("\n[Test] CodeGenerator security blocking...")
    
    gen = CodeGenerator(sandbox_timeout=5, sandbox_memory_mb=64)
    
    # Dangerous code
    code = 'import os\nos.system("echo pwned")'
    
    result = await gen.generate_custom(
        code=code,
        validate=True,
        execute=True
    )
    
    assert result.status == GenerationStatus.VALIDATION_FAILED
    assert any('os' in err for err in result.validation_errors)
    
    print(f"✓ Security blocking: {result.status.value}")
    await gen.cleanup()
    return True


async def test_code_generator_test_cases():
    """Test code generator with test cases."""
    print("\n[Test] CodeGenerator with test cases...")
    
    gen = CodeGenerator(sandbox_timeout=5, sandbox_memory_mb=64)
    
    code = '''
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

n = int(input())
print(factorial(n))
'''
    
    test_cases = [
        {"input": "5\n", "expected_output": "120"},
        {"input": "3\n", "expected_output": "6"},
    ]
    
    result = await gen.generate_custom(
        code=code,
        validate=True,
        execute=False,
        test_cases=test_cases
    )
    
    assert result.status == GenerationStatus.SUCCESS
    assert "test_results" in result.metadata
    assert result.metadata["test_results"]["passed"] == 2
    
    print(f"✓ Test cases: {result.metadata['test_results']['passed']}/2 passed")
    await gen.cleanup()
    return True


async def test_template_info():
    """Test getting template information."""
    print("\n[Test] Template info...")
    
    gen = CodeGenerator()
    info = gen.get_template_info()
    
    assert 'function' in info
    assert 'script' in info
    assert 'class' in info
    assert 'required_params' in info['function']
    
    print(f"✓ Template info: {len(info)} templates available")
    return True


async def test_convenience_functions():
    """Test convenience functions."""
    print("\n[Test] Convenience functions...")
    
    # Test generate_function
    result = await generate_function(
        function_name='subtract',
        parameters='a, b',
        body='return a - b',
        docstring='Subtract b from a',
        execute=False
    )
    
    assert result.status == GenerationStatus.SUCCESS
    assert 'def subtract(a, b):' in result.code
    
    # Test generate_script
    result = await generate_script(
        description='Test script',
        body='print("Hello from script")',
        execute=True
    )
    
    assert result.status == GenerationStatus.SUCCESS
    assert result.sandbox_result is not None
    
    print("✓ Convenience functions work")
    return True


async def test_code_validator():
    """Test CodeValidator."""
    print("\n[Test] CodeValidator...")
    
    # Valid code
    valid, errors = CodeValidator.validate_syntax("x = 1 + 2")
    assert valid is True
    assert errors is None
    
    # Invalid code
    valid, errors = CodeValidator.validate_syntax("x = 1 +")
    assert valid is False
    assert errors is not None
    
    # Full validation
    valid, errors = CodeValidator.full_validate(
        "def test():\n    return 42",
        required_functions=["test"]
    )
    assert valid is True
    assert len(errors) == 0
    
    print("✓ CodeValidator works correctly")
    return True


async def test_security_checker():
    """Test SecurityChecker."""
    print("\n[Test] SecurityChecker...")
    
    # Safe code
    error = SecurityChecker.check_code("x = 1 + 2")
    assert error is None
    
    # Dangerous import
    error = SecurityChecker.check_code("import os")
    assert error is not None
    assert "os" in error
    
    # Dangerous builtin
    error = SecurityChecker.check_code("eval('1+1')")
    assert error is not None
    assert "eval" in error
    
    print("✓ SecurityChecker works correctly")
    return True


async def run_all_tests():
    """Run all codegen tests."""
    print("="*60)
    print("Code Generation Test Suite")
    print("="*60)
    
    tests = [
        test_code_generator_template,
        test_code_generator_execution,
        test_code_generator_custom,
        test_code_generator_validation_failure,
        test_code_generator_security,
        test_code_generator_test_cases,
        test_template_info,
        test_convenience_functions,
        test_code_validator,
        test_security_checker,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            success = await test()
            if success:
                passed += 1
        except Exception as e:
            failed += 1
            print(f"✗ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


async def main():
    """Main test runner."""
    success = await run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
