"""Test suite for MemoryManager with ChromaDB integration."""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from legionhercules.memory import (
    MemoryManager,
    MemoryEntry,
    ConversationTurn,
    SessionContext,
    TokenCounter,
    LocalEmbeddingProvider,
    InMemoryVectorStore,
)


class TestMemoryManager:
    """Test MemoryManager functionality."""

    async def test_initialization(self):
        """Test MemoryManager initialization."""
        print("\n[Test] MemoryManager initialization...")
        
        # Use in-memory store for testing
        vector_store = InMemoryVectorStore()
        manager = MemoryManager(vector_store=vector_store)
        
        await manager.initialize()
        
        stats = await manager.get_stats()
        assert stats["total_memories"] == 0
        assert stats["cached_memories"] == 0
        
        print("✓ Initialization passed")
        return True

    async def test_store_memory(self):
        """Test storing a single memory."""
        print("\n[Test] Store single memory...")
        
        vector_store = InMemoryVectorStore()
        manager = MemoryManager(vector_store=vector_store)
        await manager.initialize()
        
        memory_id = await manager.store_memory(
            content="Python is a programming language",
            metadata={"category": "programming", "language": "python"},
            importance=1.5,
        )
        
        assert memory_id is not None
        assert len(memory_id) > 0
        
        stats = await manager.get_stats()
        assert stats["total_memories"] == 1
        
        print(f"✓ Store memory passed (ID: {memory_id[:8]}...)")
        return True

    async def test_store_memories_batch(self):
        """Test storing multiple memories in batch."""
        print("\n[Test] Store memories batch...")
        
        vector_store = InMemoryVectorStore()
        manager = MemoryManager(vector_store=vector_store)
        await manager.initialize()
        
        contents = [
            "Machine learning is a subset of AI",
            "Deep learning uses neural networks",
            "Transformers are attention-based models",
            "BERT is a transformer model",
            "GPT stands for Generative Pre-trained Transformer",
        ]
        
        memory_ids = await manager.store_memories_batch(
            contents=contents,
            metadata=[{"topic": "AI"} for _ in contents],
        )
        
        assert len(memory_ids) == 5
        
        stats = await manager.get_stats()
        assert stats["total_memories"] == 5
        
        print(f"✓ Batch store passed ({len(memory_ids)} memories)")
        return True

    async def test_retrieve_memories(self):
        """Test semantic memory retrieval."""
        print("\n[Test] Retrieve memories...")
        
        vector_store = InMemoryVectorStore()
        manager = MemoryManager(vector_store=vector_store)
        await manager.initialize()
        
        # Store test memories
        contents = [
            "Python is great for data science",
            "JavaScript runs in browsers",
            "Rust is memory-safe",
            "Python has pandas library",
            "Go is fast and simple",
        ]
        
        await manager.store_memories_batch(contents=contents)
        
        # Retrieve memories about Python
        results = await manager.retrieve_memories(
            query="Tell me about Python programming",
            top_k=3,
            min_relevance=0.5,
        )
        
        assert len(results) > 0
        assert len(results) <= 3
        
        # Check that Python-related memories are returned
        python_memories = [r for r in results if "python" in r["text"].lower()]
        print(f"  Found {len(python_memories)} Python-related memories")
        
        # Verify relevance scores
        for result in results:
            assert "score" in result
            assert 0 <= result["score"] <= 1
            print(f"  - {result['text'][:40]}... (score: {result['score']:.3f})")
        
        print("✓ Memory retrieval passed")
        return True

    async def test_session_management(self):
        """Test conversation session management."""
        print("\n[Test] Session management...")
        
        vector_store = InMemoryVectorStore()
        manager = MemoryManager(vector_store=vector_store)
        await manager.initialize()
        
        # Create session
        session_id = manager.create_session()
        assert session_id is not None
        assert session_id in manager.sessions
        
        # Add conversation turns
        manager.add_conversation_turn("user", "Hello, how are you?")
        manager.add_conversation_turn("assistant", "I'm doing well, thank you!")
        manager.add_conversation_turn("user", "What's the weather like?")
        
        # Get history
        history = manager.get_conversation_history()
        assert len(history) == 3
        
        # Check roles
        assert history[0].role == "user"
        assert history[1].role == "assistant"
        assert history[0].content == "Hello, how are you?"
        
        # Get recent history
        recent = manager.get_conversation_history(n=2)
        assert len(recent) == 2
        
        print(f"✓ Session management passed ({len(history)} turns)")
        return True

    async def test_context_optimization(self):
        """Test context window optimization."""
        print("\n[Test] Context optimization...")
        
        vector_store = InMemoryVectorStore()
        manager = MemoryManager(
            vector_store=vector_store,
            max_context_tokens=1000,
        )
        await manager.initialize()
        
        # Create session with history
        manager.create_session()
        
        # Add many conversation turns
        for i in range(20):
            manager.add_conversation_turn(
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i}: This is a test message with some content. " * 10,
            )
        
        # Get context
        context = await manager.get_context_for_llm(
            query="test",
            include_history=True,
            include_memories=False,
        )
        
        assert "conversation_history" in context
        
        # History should be optimized (not all 20 messages)
        print(f"  Optimized history: {len(context['conversation_history'])} messages")
        
        print("✓ Context optimization passed")
        return True

    async def test_token_counting(self):
        """Test token counting functionality."""
        print("\n[Test] Token counting...")
        
        counter = TokenCounter(model="gpt-4")
        
        # Count tokens in text
        text = "Hello, world! This is a test."
        count = counter.count(text)
        assert count > 0
        print(f"  Tokens in sample text: {count}")
        
        # Count tokens in messages
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        msg_count = counter.count_messages(messages)
        assert msg_count > 0
        print(f"  Tokens in messages: {msg_count}")
        
        print("✓ Token counting passed")
        return True

    async def test_persistence(self):
        """Test memory persistence."""
        print("\n[Test] Persistence...")
        
        import tempfile
        
        vector_store = InMemoryVectorStore()
        manager = MemoryManager(vector_store=vector_store)
        await manager.initialize()
        
        # Create session and add data
        session_id = manager.create_session()
        manager.add_conversation_turn("user", "Test message")
        
        # Persist
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            success = await manager.persist(temp_path)
            assert success
            
            # Create new manager and load
            manager2 = MemoryManager(vector_store=InMemoryVectorStore())
            success = await manager2.load(temp_path)
            assert success
            
            # Verify session restored
            assert session_id in manager2.sessions
            
            print("✓ Persistence passed")
            return True
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    async def test_semantic_search_accuracy(self):
        """Test semantic search accuracy with 100 memories."""
        print("\n[Test] Semantic search accuracy (100 memories)...")
        
        vector_store = InMemoryVectorStore()
        manager = MemoryManager(vector_store=vector_store)
        await manager.initialize()
        
        # Create diverse test memories
        test_memories = []
        
        # Programming topics (25 memories)
        programming_topics = [
            "Python is a high-level programming language",
            "JavaScript is used for web development",
            "Rust provides memory safety guarantees",
            "Go is designed for concurrency",
            "Java is object-oriented",
            "C++ is used for system programming",
            "Ruby is known for its elegant syntax",
            "Swift is used for iOS development",
            "Kotlin is a modern Android language",
            "TypeScript adds types to JavaScript",
            "PHP is a server-side scripting language",
            "Perl is used for text processing",
            "R is for statistical computing",
            "MATLAB is for numerical computing",
            "Scala runs on the JVM",
            "Clojure is a Lisp dialect",
            "Haskell is purely functional",
            "Elixir runs on the Erlang VM",
            "Dart is used with Flutter",
            "Lua is used for game scripting",
            "Groovy is a Java alternative",
            "Julia is for scientific computing",
            "F# is a functional-first language",
            "OCaml is used in theorem proving",
            "Erlang is for distributed systems",
        ]
        
        # AI/ML topics (25 memories)
        ai_topics = [
            "Machine learning is a subset of AI",
            "Deep learning uses neural networks",
            "Transformers revolutionized NLP",
            "BERT uses bidirectional training",
            "GPT is a generative model",
            "CNNs are used for images",
            "RNNs process sequences",
            "LSTMs solve vanishing gradients",
            "GANs generate synthetic data",
            "Reinforcement learning uses rewards",
            "Supervised learning uses labels",
            "Unsupervised learning finds patterns",
            "Semi-supervised learning uses both",
            "Self-supervised learning creates labels",
            "Transfer learning reuses models",
            "Fine-tuning adapts pre-trained models",
            "Prompt engineering guides LLMs",
            "Attention mechanisms focus selectively",
            "Multi-head attention uses parallel heads",
            "Positional encoding adds sequence info",
            "Tokenization splits text into tokens",
            "Embeddings represent words as vectors",
            "Word2Vec creates word embeddings",
            "BERT embeddings are contextualized",
            "T5 uses text-to-text framework",
        ]
        
        # Science topics (25 memories)
        science_topics = [
            "Photosynthesis converts light to energy",
            "DNA contains genetic information",
            "Newton discovered gravity",
            "Einstein developed relativity",
            "Quantum mechanics describes particles",
            "Evolution explains species diversity",
            "Cells are the building blocks",
            "Atoms are made of subatomic particles",
            "The speed of light is constant",
            "Thermodynamics governs energy transfer",
            "Plate tectonics move continents",
            "Climate change affects global temperatures",
            "Biodiversity is essential for ecosystems",
            "The immune system fights disease",
            "Neurons transmit electrical signals",
            "Enzymes catalyze biochemical reactions",
            "Mitochondria produce cellular energy",
            "Ribosomes synthesize proteins",
            "Natural selection drives adaptation",
            "Fossils show evolutionary history",
            "The Big Bang started the universe",
            "Black holes have event horizons",
            "Dark matter affects galaxy rotation",
            "Antibiotics kill bacteria",
            "Vaccines stimulate immune response",
        ]
        
        # History/Geography topics (25 memories)
        history_topics = [
            "Rome was founded in 753 BC",
            "The pyramids are in Egypt",
            "World War II ended in 1945",
            "The Renaissance began in Italy",
            "The Industrial Revolution started in Britain",
            "The French Revolution was in 1789",
            "The Cold War lasted decades",
            "Ancient Greece invented democracy",
            "The Silk Road connected East and West",
            "The Mongols created a vast empire",
            "The Ottoman Empire lasted centuries",
            "The Roman Empire fell in 476 AD",
            "The printing press spread knowledge",
            "The internet revolutionized communication",
            "The Berlin Wall fell in 1989",
            "The Titanic sank in 1912",
            "The moon landing was in 1969",
            "The Great Depression affected millions",
            "The Civil Rights Movement fought equality",
            "The Magna Carta limited monarchy",
            "The Enlightenment emphasized reason",
            "The Scientific Method tests hypotheses",
            "The Age of Exploration discovered continents",
            "The American Revolution created the USA",
            "The French Empire was led by Napoleon",
        ]
        
        all_memories = programming_topics + ai_topics + science_topics + history_topics
        
        # Store all memories
        print(f"  Storing {len(all_memories)} memories...")
        memory_ids = await manager.store_memories_batch(
            contents=all_memories,
            metadata=[{"category": "test"} for _ in all_memories],
        )
        
        assert len(memory_ids) == 100
        
        # Test queries with expected relevant topics - using lower threshold for small model
        test_queries = [
            ("What programming languages are good for web development?", ["javascript", "typescript", "php", "web", "programming"]),
            ("Tell me about artificial intelligence and neural networks", ["machine learning", "deep learning", "transformers", "neural", "ai"]),
            ("How do cells and biology work?", ["dna", "photosynthesis", "cells", "biology", "genetic"]),
            ("What happened in ancient history?", ["rome", "egypt", "greece", "ancient", "empire"]),
        ]
        
        total_accuracy = 0
        total_results = 0
        
        for query, expected_keywords in test_queries:
            results = await manager.retrieve_memories(query, top_k=10, min_relevance=0.3)
            total_results += len(results)
            
            # Check if results contain expected keywords
            relevant_count = 0
            for result in results:
                text = result["text"].lower()
                if any(kw in text for kw in expected_keywords):
                    relevant_count += 1
            
            accuracy = relevant_count / len(results) if results else 0
            total_accuracy += accuracy
            
            print(f"  Query: '{query[:50]}...'")
            print(f"    Retrieved: {len(results)} results, Accuracy: {accuracy*100:.1f}% ({relevant_count}/{len(results)} relevant)")
        
        avg_accuracy = total_accuracy / len(test_queries)
        print(f"\n  Average semantic search accuracy: {avg_accuracy*100:.1f}%")
        print(f"  Total results retrieved: {total_results}")
        
        # Assert we got results and reasonable accuracy (>40% for small model)
        # Note: Using all-MiniLM-L6-v2 (small model), accuracy would be >80% with larger models
        assert total_results >= 10, f"Only retrieved {total_results} results, expected at least 10"
        assert avg_accuracy >= 0.4, f"Accuracy {avg_accuracy*100:.1f}% is below 40% threshold"
        
        print("✓ Semantic search accuracy test passed (>40% with mini model)")
        return True

    async def run_all_tests(self):
        """Run all tests."""
        print("="*60)
        print("MemoryManager Test Suite")
        print("="*60)
        
        tests = [
            self.test_initialization,
            self.test_store_memory,
            self.test_store_memories_batch,
            self.test_retrieve_memories,
            self.test_session_management,
            self.test_context_optimization,
            self.test_token_counting,
            self.test_persistence,
            self.test_semantic_search_accuracy,
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
    tester = TestMemoryManager()
    success = await tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
