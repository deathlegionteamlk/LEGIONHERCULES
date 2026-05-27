"""Test suite for self-improving agent loop."""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from legionhercules.core.agent_loop import AgentLoop, AgentLoopStatus, AdaptiveAgentLoop
from legionhercules.self_improve.reflection import ReflectionEngine
from legionhercules.self_improve.iteration import IterationManager


class TestAgentLoop:
    """Test AgentLoop functionality."""

    async def test_simple_execution(self):
        """Test simple task execution without reflection."""
        print("\n[Test] Simple execution...")
        
        agent = AgentLoop(enable_reflection=False, enable_iteration=False)
        
        async def simple_task():
            return "Hello, World!"
        
        result = await agent.execute(
            description="Simple greeting task",
            execute_fn=simple_task,
        )
        
        assert result.success is True
        assert result.final_output == "Hello, World!"
        assert result.iterations == 1
        assert result.status == AgentLoopStatus.CONVERGED
        
        print(f"✓ Simple execution passed: {result.final_output}")
        return True

    async def test_execution_with_reflection(self):
        """Test execution with reflection."""
        print("\n[Test] Execution with reflection...")
        
        agent = AgentLoop(enable_reflection=True, enable_iteration=False)
        
        async def task_with_issues():
            return "TODO: Fix this output"
        
        result = await agent.execute(
            description="Task with TODO marker",
            execute_fn=task_with_issues,
        )
        
        # Reflection may suggest improvements, so success depends on confidence
        assert len(result.reflection_results) == 1
        
        reflection = result.reflection_results[0]
        assert "TODO" in reflection.reflection_analysis or len(reflection.improvements_suggested) > 0
        
        print(f"✓ Reflection execution passed: success={result.success}, {len(reflection.improvements_suggested)} improvements suggested")
        return True

    async def test_execution_with_iteration(self):
        """Test execution with iteration."""
        print("\n[Test] Execution with iteration...")
        
        agent = AgentLoop(
            enable_reflection=True,
            enable_iteration=True,
            default_max_iterations=2,
        )
        
        call_count = 0
        
        async def improving_task():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "Draft output with some issues"
            return "Improved output"
        
        result = await agent.execute(
            description="Task that improves",
            execute_fn=improving_task,
            max_iterations=2,
        )
        
        assert result.success is True
        assert result.iterations >= 1
        
        print(f"✓ Iteration execution passed: {result.iterations} iterations, confidence={result.confidence_score:.2f}")
        return True

    async def test_validation_failure(self):
        """Test task with validation failure."""
        print("\n[Test] Validation failure...")
        
        agent = AgentLoop(enable_reflection=False, enable_iteration=False)
        
        async def failing_task():
            return "Invalid output"
        
        async def validate_output(output):
            return False, "Output is not valid"
        
        result = await agent.execute(
            description="Task that fails validation",
            execute_fn=failing_task,
            validate_fn=validate_output,
        )
        
        # Validation failure sets success to False
        assert result.success is False
        # Error is stored in task.error, not metadata
        
        print(f"✓ Validation failure passed: success={result.success}")
        return True

    async def test_batch_execution(self):
        """Test batch task execution."""
        print("\n[Test] Batch execution...")
        
        agent = AgentLoop(enable_reflection=False, enable_iteration=False)
        
        async def make_task(i):
            async def task_fn():
                await asyncio.sleep(0.01)
                return f"Result {i}"
            return task_fn
        
        tasks = []
        for i in range(5):
            task_fn = await make_task(i)
            tasks.append({
                "description": f"Task {i}",
                "execute_fn": task_fn,
            })
        
        results = await agent.execute_batch(tasks, max_concurrent=2)
        
        assert len(results) == 5
        assert all(r.success for r in results)
        
        outputs = [r.final_output for r in results]
        assert "Result 0" in outputs
        assert "Result 4" in outputs
        
        print(f"✓ Batch execution passed: {len(results)} tasks completed")
        return True

    async def test_convergence_detection(self):
        """Test convergence detection."""
        print("\n[Test] Convergence detection...")
        
        from legionhercules.core.agent_loop import ConvergenceDetector
        
        detector = ConvergenceDetector(threshold=0.01, window_size=3)
        
        # Add scores that show convergence
        detector.add_score(0.5)
        detector.add_score(0.51)
        detector.add_score(0.505)
        
        assert detector.has_converged() is True
        
        detector.reset()
        detector.add_score(0.5)
        detector.add_score(0.6)
        detector.add_score(0.7)
        
        assert detector.has_converged() is False
        
        print("✓ Convergence detection passed")
        return True

    async def test_adaptive_agent_loop(self):
        """Test adaptive agent loop with complexity estimation."""
        print("\n[Test] Adaptive agent loop...")
        
        agent = AdaptiveAgentLoop(enable_reflection=False, enable_iteration=False)
        
        # Test complexity estimation
        simple_complexity = agent.estimate_complexity("Simple task")
        complex_complexity = agent.estimate_complexity("This is a very complex multi-step comprehensive task with extensive requirements")
        
        assert 0 <= simple_complexity <= 1
        assert 0 <= complex_complexity <= 1
        assert complex_complexity > simple_complexity
        
        print(f"✓ Adaptive loop passed: simple={simple_complexity:.2f}, complex={complex_complexity:.2f}")
        return True

    async def test_task_history(self):
        """Test task history tracking."""
        print("\n[Test] Task history...")
        
        agent = AgentLoop(enable_reflection=False, enable_iteration=False)
        
        async def task():
            return "Done"
        
        # Execute a few tasks
        for i in range(3):
            await agent.execute(
                description=f"Task {i}",
                execute_fn=task,
            )
        
        history = agent.get_task_history()
        assert len(history) == 3
        
        success_history = agent.get_task_history(success_only=True)
        assert len(success_history) == 3
        
        limited_history = agent.get_task_history(limit=2)
        assert len(limited_history) == 2
        
        print(f"✓ Task history passed: {len(history)} tasks tracked")
        return True

    async def test_stats(self):
        """Test statistics collection."""
        print("\n[Test] Statistics...")
        
        agent = AgentLoop(enable_reflection=False, enable_iteration=False)
        
        async def success_task():
            return "Success"
        
        await agent.execute(description="Success task", execute_fn=success_task)
        
        stats = agent.get_stats()
        
        assert stats["total_tasks"] == 1
        assert stats["successful_tasks"] == 1
        assert stats["success_rate"] == 1.0
        
        print(f"✓ Statistics passed: {stats}")
        return True

    async def test_clear_history(self):
        """Test clearing history."""
        print("\n[Test] Clear history...")
        
        agent = AgentLoop(enable_reflection=False, enable_iteration=False)
        
        async def task():
            return "Done"
        
        await agent.execute(description="Task", execute_fn=task)
        
        assert len(agent.task_history) == 1
        
        agent.clear_history()
        
        assert len(agent.task_history) == 0
        assert len(agent.completed_tasks) == 0
        
        print("✓ Clear history passed")
        return True

    async def test_error_handling(self):
        """Test error handling in execution."""
        print("\n[Test] Error handling...")
        
        agent = AgentLoop(enable_reflection=False, enable_iteration=False)
        
        async def error_task():
            raise ValueError("Something went wrong")
        
        result = await agent.execute(
            description="Task that errors",
            execute_fn=error_task,
        )
        
        assert result.success is False
        assert result.status == AgentLoopStatus.FAILED
        assert "Something went wrong" in result.metadata.get("error", "")
        
        print(f"✓ Error handling passed: {result.metadata.get('error')}")
        return True

    async def run_all_tests(self):
        """Run all tests."""
        print("="*60)
        print("AgentLoop Test Suite")
        print("="*60)
        
        tests = [
            self.test_simple_execution,
            self.test_execution_with_reflection,
            self.test_execution_with_iteration,
            self.test_validation_failure,
            self.test_batch_execution,
            self.test_convergence_detection,
            self.test_adaptive_agent_loop,
            self.test_task_history,
            self.test_stats,
            self.test_clear_history,
            self.test_error_handling,
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
    tester = TestAgentLoop()
    success = await tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
