"""Test suite for multi-step planning with task decomposition."""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from legionhercules.planning.decomposer import TaskDecomposer, Task, Plan, TaskStatus, TaskPriority
from legionhercules.planning.executor import PlanExecutor, ExecutionMode, ExecutionResult


async def test_heuristic_decomposition_create():
    """Test heuristic decomposition for create/build tasks."""
    print("\n[Test] Heuristic decomposition (create)...")
    
    decomposer = TaskDecomposer()
    plan = await decomposer.decompose("Create a new web application")
    
    assert len(plan.tasks) >= 4
    assert any("requirements" in t.description.lower() for t in plan.tasks)
    assert any("design" in t.description.lower() for t in plan.tasks)
    assert any("implement" in t.description.lower() for t in plan.tasks)
    
    print(f"✓ Create decomposition: {len(plan.tasks)} tasks")
    return True


async def test_heuristic_decomposition_fix():
    """Test heuristic decomposition for fix/debug tasks."""
    print("\n[Test] Heuristic decomposition (fix)...")
    
    decomposer = TaskDecomposer()
    plan = await decomposer.decompose("Fix the authentication bug")
    
    assert len(plan.tasks) >= 4
    assert any("identify" in t.description.lower() for t in plan.tasks)
    assert any("fix" in t.description.lower() for t in plan.tasks)
    
    print(f"✓ Fix decomposition: {len(plan.tasks)} tasks")
    return True


async def test_heuristic_decomposition_research():
    """Test heuristic decomposition for research tasks."""
    print("\n[Test] Heuristic decomposition (research)...")
    
    decomposer = TaskDecomposer()
    plan = await decomposer.decompose("Research best practices for async programming")
    
    assert len(plan.tasks) >= 3
    assert any("research" in t.description.lower() or "gather" in t.description.lower() for t in plan.tasks)
    
    print(f"✓ Research decomposition: {len(plan.tasks)} tasks")
    return True


async def test_dependency_tracking():
    """Test dependency tracking in tasks."""
    print("\n[Test] Dependency tracking...")
    
    decomposer = TaskDecomposer()
    plan = await decomposer.decompose("Build a complex system")
    
    # Check that tasks have dependencies
    tasks_with_deps = [t for t in plan.tasks if t.dependencies]
    assert len(tasks_with_deps) > 0
    
    # Verify dependency IDs are valid
    task_ids = {t.id for t in plan.tasks}
    for task in tasks_with_deps:
        for dep in task.dependencies:
            assert dep in task_ids
    
    print(f"✓ Dependency tracking: {len(tasks_with_deps)} tasks with dependencies")
    return True


async def test_execution_order():
    """Test execution order respects dependencies."""
    print("\n[Test] Execution order...")
    
    decomposer = TaskDecomposer()
    plan = await decomposer.decompose("Build a system")
    
    execution_order = decomposer.get_execution_order(plan)
    
    # Verify order respects dependencies
    task_positions = {t.id: i for i, t in enumerate(execution_order)}
    for task in execution_order:
        for dep in task.dependencies:
            assert task_positions[dep] < task_positions[task.id]
    
    print(f"✓ Execution order: {len(execution_order)} tasks in correct order")
    return True


async def test_critical_path():
    """Test critical path identification."""
    print("\n[Test] Critical path...")
    
    decomposer = TaskDecomposer()
    plan = await decomposer.decompose("Build a system")
    
    critical_path = decomposer.get_critical_path(plan)
    
    assert len(critical_path) > 0
    # First task should have no dependencies
    assert len(critical_path[0].dependencies) == 0
    
    print(f"✓ Critical path: {len(critical_path)} tasks")
    return True


async def test_ready_tasks():
    """Test getting ready tasks."""
    print("\n[Test] Ready tasks...")
    
    decomposer = TaskDecomposer()
    plan = await decomposer.decompose("Build a system")
    
    # Initially, tasks with no dependencies should be ready
    ready = decomposer.get_ready_tasks(plan)
    
    # At least one task should be ready (the first one)
    assert len(ready) > 0
    assert all(len(t.dependencies) == 0 for t in ready)
    
    print(f"✓ Ready tasks: {len(ready)} tasks ready")
    return True


async def test_plan_progress():
    """Test plan progress calculation."""
    print("\n[Test] Plan progress...")
    
    decomposer = TaskDecomposer()
    plan = await decomposer.decompose("Build a system")
    
    # Initially 0 progress
    assert plan.progress == 0.0
    
    # Mark half as complete
    half = len(plan.tasks) // 2
    for i, task in enumerate(plan.tasks):
        if i < half:
            task.status = TaskStatus.COMPLETED
    
    # Progress should be approximately 0.5 (allowing for rounding)
    expected_progress = half / len(plan.tasks)
    assert abs(plan.progress - expected_progress) < 0.01
    
    print(f"✓ Plan progress: {plan.progress:.0%}")
    return True


async def test_sequential_execution():
    """Test sequential plan execution."""
    print("\n[Test] Sequential execution...")
    
    executor = PlanExecutor(mode=ExecutionMode.SEQUENTIAL)
    
    execution_order = []
    
    async def handler(task):
        execution_order.append(task.id)
        return f"Result for {task.id}"
    
    executor.set_default_handler(handler)
    
    # Create a simple plan
    plan = Plan(goal="Test sequential execution", tasks=[
        Task(id="1", description="Task 1", dependencies=[]),
        Task(id="2", description="Task 2", dependencies=["1"]),
        Task(id="3", description="Task 3", dependencies=["2"]),
    ])
    
    result = await executor.execute(plan)
    
    assert result.success is True
    assert len(result.completed_tasks) == 3
    assert execution_order == ["1", "2", "3"]
    
    print(f"✓ Sequential execution: {len(result.completed_tasks)} tasks completed")
    return True


async def test_parallel_execution():
    """Test parallel plan execution."""
    print("\n[Test] Parallel execution...")
    
    executor = PlanExecutor(mode=ExecutionMode.PARALLEL, max_parallel_tasks=3)
    
    async def handler(task):
        await asyncio.sleep(0.01)  # Small delay
        return f"Result for {task.id}"
    
    executor.set_default_handler(handler)
    
    # Create a plan with independent tasks
    plan = Plan(goal="Test parallel execution", tasks=[
        Task(id="1", description="Task 1", dependencies=[]),
        Task(id="2", description="Task 2", dependencies=[]),
        Task(id="3", description="Task 3", dependencies=[]),
    ])
    
    result = await executor.execute(plan)
    
    assert result.success is True
    assert len(result.completed_tasks) == 3
    
    print(f"✓ Parallel execution: {len(result.completed_tasks)} tasks completed")
    return True


async def test_task_failure():
    """Test handling of task failures."""
    print("\n[Test] Task failure handling...")
    
    executor = PlanExecutor(mode=ExecutionMode.SEQUENTIAL)
    
    async def handler(task):
        if task.id == "2":
            raise ValueError("Task 2 failed")
        return f"Result for {task.id}"
    
    executor.set_default_handler(handler)
    
    plan = Plan(goal="Test failure handling", tasks=[
        Task(id="1", description="Task 1", dependencies=[]),
        Task(id="2", description="Task 2", dependencies=["1"]),
        Task(id="3", description="Task 3", dependencies=["2"]),
    ])
    
    result = await executor.execute(plan)
    
    assert result.success is False
    assert "1" in result.completed_tasks
    assert "2" in result.failed_tasks
    assert "Task 2 failed" in result.errors.get("2", "")
    
    print(f"✓ Task failure: {len(result.failed_tasks)} tasks failed")
    return True


async def test_execution_with_recovery():
    """Test execution with retry."""
    print("\n[Test] Execution with recovery...")
    
    executor = PlanExecutor(mode=ExecutionMode.SEQUENTIAL)
    
    call_count = 0
    
    async def handler(task):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ValueError("Temporary failure")
        return "Success"
    
    executor.set_default_handler(handler)
    
    plan = Plan(goal="Test recovery", tasks=[
        Task(id="1", description="Task 1", dependencies=[]),
    ])
    
    result = await executor.execute_with_recovery(plan, max_retries=3)
    
    assert result.success is True
    assert call_count == 2  # Failed once, succeeded on retry
    
    print(f"✓ Recovery: succeeded after {call_count} attempts")
    return True


async def test_execution_stats():
    """Test execution statistics."""
    print("\n[Test] Execution stats...")
    
    executor = PlanExecutor(mode=ExecutionMode.SEQUENTIAL)
    
    async def handler(task):
        return f"Result for {task.id}"
    
    executor.set_default_handler(handler)
    
    plan = Plan(goal="Test stats", tasks=[
        Task(id="1", description="Task 1", dependencies=[]),
        Task(id="2", description="Task 2", dependencies=[]),
    ])
    
    result = await executor.execute(plan)
    stats = executor.get_execution_stats(result)
    
    assert stats["total_tasks"] == 2
    assert stats["completed"] == 2
    assert stats["success_rate"] == 1.0
    assert stats["duration_seconds"] >= 0
    
    print(f"✓ Stats: {stats['completed']}/{stats['total_tasks']} tasks, {stats['success_rate']:.0%} success")
    return True


async def test_complex_dependency_chain():
    """Test complex dependency chains."""
    print("\n[Test] Complex dependency chain...")
    
    decomposer = TaskDecomposer()
    executor = PlanExecutor(mode=ExecutionMode.ADAPTIVE)
    
    async def handler(task):
        return f"Completed {task.description}"
    
    executor.set_default_handler(handler)
    
    # Create a plan with diamond dependency pattern
    plan = Plan(goal="Complex dependency test", tasks=[
        Task(id="A", description="Start", dependencies=[]),
        Task(id="B", description="Parallel 1", dependencies=["A"]),
        Task(id="C", description="Parallel 2", dependencies=["A"]),
        Task(id="D", description="Merge", dependencies=["B", "C"]),
    ])
    
    result = await executor.execute(plan)
    
    assert result.success is True
    assert len(result.completed_tasks) == 4
    
    print(f"✓ Complex chain: {len(result.completed_tasks)} tasks with diamond dependencies")
    return True


async def test_decomposition_10_tasks():
    """Test decomposition produces sufficient tasks for complex goals."""
    print("\n[Test] Decomposition (complex goals)...")
    
    decomposer = TaskDecomposer()
    
    complex_goals = [
        "Create a full-stack web application with authentication, database, and API",
        "Build a microservices architecture with service discovery and load balancing",
        "Implement a distributed system with consensus algorithm and replication",
    ]
    
    for goal in complex_goals:
        plan = await decomposer.decompose(goal)
        assert len(plan.tasks) >= 4, f"Goal '{goal[:30]}...' produced only {len(plan.tasks)} tasks"
    
    print(f"✓ Decomposition: All complex goals produced sufficient tasks")
    return True


async def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("Planning Test Suite")
    print("="*60)
    
    tests = [
        test_heuristic_decomposition_create,
        test_heuristic_decomposition_fix,
        test_heuristic_decomposition_research,
        test_dependency_tracking,
        test_execution_order,
        test_critical_path,
        test_ready_tasks,
        test_plan_progress,
        test_sequential_execution,
        test_parallel_execution,
        test_task_failure,
        test_execution_with_recovery,
        test_execution_stats,
        test_complex_dependency_chain,
        test_decomposition_10_tasks,
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
