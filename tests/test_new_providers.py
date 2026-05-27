"""Tests for new LLM providers: Azure, Fireworks, Perplexity, Replicate."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from legionhercules.llm.azure_provider import AzureOpenAIProvider
from legionhercules.llm.fireworks_provider import FireworksProvider
from legionhercules.llm.perplexity_provider import PerplexityProvider
from legionhercules.llm.replicate_provider import ReplicateProvider
from legionhercules.core.message import Message


class TestAzureOpenAIProvider:
    """Test Azure OpenAI provider."""

    @pytest.mark.asyncio
    async def test_initialization_requires_api_key(self):
        """Test that initialization requires API key."""
        provider = AzureOpenAIProvider(
            model="gpt-4o",
            endpoint="https://test.openai.azure.com"
        )
        with pytest.raises(ValueError, match="Azure OpenAI API key required"):
            await provider.initialize()

    @pytest.mark.asyncio
    async def test_initialization_requires_endpoint(self):
        """Test that initialization requires endpoint."""
        provider = AzureOpenAIProvider(
            model="gpt-4o",
            api_key="test-key"
        )
        with pytest.raises(ValueError, match="Azure OpenAI endpoint required"):
            await provider.initialize()

    @pytest.mark.asyncio
    async def test_initialization_success(self):
        """Test successful initialization."""
        provider = AzureOpenAIProvider(
            model="gpt-4o",
            api_key="test-key",
            endpoint="https://test.openai.azure.com"
        )
        await provider.initialize()
        assert provider._initialized is True
        assert provider._client is not None
        await provider.close()

    @pytest.mark.asyncio
    async def test_chat_method(self):
        """Test chat method with mocked response."""
        provider = AzureOpenAIProvider(
            model="gpt-4o",
            api_key="test-key",
            endpoint="https://test.openai.azure.com"
        )
        
        # Mock the HTTP client
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello!"}, "finish_reason": "stop"}],
            "model": "gpt-4o",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }
        
        with patch.object(provider, '_client') as mock_client:
            mock_client.post = AsyncMock(return_value=mock_response)
            
            messages = [Message.user("Hi")]
            response = await provider.chat(messages)
            
            assert response.content == "Hello!"
            assert response.metadata["model"] == "gpt-4o"
            assert response.usage is not None

    @pytest.mark.asyncio
    async def test_generate_method(self):
        """Test generate method."""
        provider = AzureOpenAIProvider(
            model="gpt-4o",
            api_key="test-key",
            endpoint="https://test.openai.azure.com"
        )
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Generated text"}, "finish_reason": "stop"}],
            "model": "gpt-4o",
        }
        
        with patch.object(provider, '_client') as mock_client:
            mock_client.post = AsyncMock(return_value=mock_response)
            
            response = await provider.generate("Test prompt")
            assert response.content == "Generated text"

    @pytest.mark.asyncio
    async def test_list_models(self):
        """Test list_models returns expected models."""
        provider = AzureOpenAIProvider(
            model="gpt-4o",
            api_key="test-key",
            endpoint="https://test.openai.azure.com"
        )
        models = await provider.list_models()
        assert "gpt-4o" in models
        assert "gpt-4" in models
        assert "gpt-35-turbo" in models


class TestFireworksProvider:
    """Test Fireworks AI provider."""

    @pytest.mark.asyncio
    async def test_initialization_requires_api_key(self):
        """Test that initialization requires API key."""
        provider = FireworksProvider(model="llama-v3p1-70b")
        with pytest.raises(ValueError, match="Fireworks API key required"):
            await provider.initialize()

    @pytest.mark.asyncio
    async def test_initialization_success(self):
        """Test successful initialization."""
        provider = FireworksProvider(
            model="accounts/fireworks/models/llama-v3p1-70b-instruct",
            api_key="test-key"
        )
        await provider.initialize()
        assert provider._initialized is True
        await provider.close()

    @pytest.mark.asyncio
    async def test_chat_method(self):
        """Test chat method with mocked response."""
        provider = FireworksProvider(
            model="accounts/fireworks/models/llama-v3p1-70b-instruct",
            api_key="test-key"
        )
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Fireworks response!"}, "finish_reason": "stop"}],
            "model": "llama-v3p1-70b-instruct",
            "usage": {"prompt_tokens": 5, "completion_tokens": 3}
        }
        
        with patch.object(provider, '_client') as mock_client:
            mock_client.post = AsyncMock(return_value=mock_response)
            
            messages = [Message.user("Hello")]
            response = await provider.chat(messages)
            
            assert response.content == "Fireworks response!"
            assert response.metadata["model"] == "llama-v3p1-70b-instruct"

    @pytest.mark.asyncio
    async def test_list_models_fallback(self):
        """Test list_models returns fallback models on error."""
        provider = FireworksProvider(
            model="llama-v3p1-70b",
            api_key="test-key"
        )
        await provider.initialize()
        
        # Mock client to raise exception
        with patch.object(provider, '_client') as mock_client:
            mock_client.get = AsyncMock(side_effect=Exception("API Error"))
            models = await provider.list_models()
            
            assert len(models) > 0
            assert "accounts/fireworks/models/llama-v3p1-70b-instruct" in models


class TestPerplexityProvider:
    """Test Perplexity AI provider."""

    @pytest.mark.asyncio
    async def test_initialization_requires_api_key(self):
        """Test that initialization requires API key."""
        provider = PerplexityProvider(model="sonar-pro")
        with pytest.raises(ValueError, match="Perplexity API key required"):
            await provider.initialize()

    @pytest.mark.asyncio
    async def test_initialization_success(self):
        """Test successful initialization."""
        provider = PerplexityProvider(
            model="sonar-pro",
            api_key="test-key"
        )
        await provider.initialize()
        assert provider._initialized is True
        await provider.close()

    @pytest.mark.asyncio
    async def test_chat_with_citations(self):
        """Test chat method with citations."""
        provider = PerplexityProvider(
            model="sonar-pro",
            api_key="test-key"
        )
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Answer with sources"}, "finish_reason": "stop"}],
            "model": "sonar-pro",
            "citations": ["https://example.com/source1", "https://example.com/source2"],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8}
        }
        
        with patch.object(provider, '_client') as mock_client:
            mock_client.post = AsyncMock(return_value=mock_response)
            
            messages = [Message.user("What is AI?")]
            response = await provider.chat(messages)
            
            assert response.content == "Answer with sources"
            assert response.metadata.get("citations") is not None
            assert len(response.metadata["citations"]) == 2
            assert response.metadata.get("has_search_results") is True

    @pytest.mark.asyncio
    async def test_search_method(self):
        """Test search method."""
        provider = PerplexityProvider(
            model="sonar-pro",
            api_key="test-key"
        )
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Search results"}, "finish_reason": "stop"}],
            "model": "sonar-pro",
            "citations": ["https://example.com"],
        }
        
        with patch.object(provider, '_client') as mock_client:
            mock_client.post = AsyncMock(return_value=mock_response)
            
            response = await provider.search("Python programming")
            assert response.content == "Search results"

    @pytest.mark.asyncio
    async def test_list_models(self):
        """Test list_models returns Perplexity models."""
        provider = PerplexityProvider(
            model="sonar-pro",
            api_key="test-key"
        )
        models = await provider.list_models()
        assert "sonar" in models
        assert "sonar-pro" in models
        assert "sonar-reasoning" in models


class TestReplicateProvider:
    """Test Replicate provider."""

    @pytest.mark.asyncio
    async def test_initialization_requires_api_key(self):
        """Test that initialization requires API key."""
        provider = ReplicateProvider(model="meta/llama-3-70b")
        with pytest.raises(ValueError, match="Replicate API token required"):
            await provider.initialize()

    @pytest.mark.asyncio
    async def test_initialization_success(self):
        """Test successful initialization."""
        provider = ReplicateProvider(
            model="meta/meta-llama-3-70b-instruct",
            api_key="r8_test_token"
        )
        await provider.initialize()
        assert provider._initialized is True
        await provider.close()

    @pytest.mark.asyncio
    async def test_chat_success_immediate(self):
        """Test chat with immediate success."""
        provider = ReplicateProvider(
            model="meta/meta-llama-3-70b-instruct",
            api_key="r8_test_token"
        )
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "test-prediction",
            "status": "succeeded",
            "output": "Replicate response",
            "metrics": {"predict_time": 1.5}
        }
        
        with patch.object(provider, '_client') as mock_client:
            mock_client.post = AsyncMock(return_value=mock_response)
            
            messages = [Message.user("Hello")]
            response = await provider.chat(messages)
            
            assert response.content == "Replicate response"
            assert response.metadata["status"] == "succeeded"

    @pytest.mark.asyncio
    async def test_chat_with_polling(self):
        """Test chat requiring polling."""
        provider = ReplicateProvider(
            model="meta/meta-llama-3-70b-instruct",
            api_key="r8_test_token"
        )
        
        # First response: prediction started
        mock_post_response = MagicMock()
        mock_post_response.status_code = 200
        mock_post_response.json.return_value = {
            "id": "test-prediction",
            "status": "starting",
        }
        
        # Poll responses
        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {
            "id": "test-prediction",
            "status": "succeeded",
            "output": ["Polled", " response"],
            "metrics": {"predict_time": 2.0}
        }
        
        with patch.object(provider, '_client') as mock_client:
            mock_client.post = AsyncMock(return_value=mock_post_response)
            mock_client.get = AsyncMock(return_value=mock_get_response)
            
            messages = [Message.user("Hello")]
            response = await provider.chat(messages)
            
            assert response.content == "Polled response"

    @pytest.mark.asyncio
    async def test_list_models(self):
        """Test list_models returns popular models."""
        provider = ReplicateProvider(
            model="meta/meta-llama-3-70b-instruct",
            api_key="r8_test_token"
        )
        models = await provider.list_models()
        assert "meta/llama-2-70b-chat" in models
        assert "meta/llama-3-70b-instruct" in models
        assert "mistralai/mixtral-8x7b-instruct" in models


class TestProviderErrorHandling:
    """Test error handling across all providers."""

    @pytest.mark.asyncio
    async def test_azure_error_handling(self):
        """Test Azure provider handles API errors gracefully."""
        provider = AzureOpenAIProvider(
            model="gpt-4o",
            api_key="test-key",
            endpoint="https://test.openai.azure.com"
        )
        await provider.initialize()
        
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.raise_for_status.side_effect = Exception("Rate limited")
        
        with patch.object(provider, '_client') as mock_client:
            mock_client.post = AsyncMock(return_value=mock_response)
            
            messages = [Message.user("Hi")]
            response = await provider.chat(messages)
            
            assert "Error" in response.content
            assert "error" in response.metadata

    @pytest.mark.asyncio
    async def test_fireworks_error_handling(self):
        """Test Fireworks provider handles API errors gracefully."""
        provider = FireworksProvider(
            model="llama-v3p1-70b",
            api_key="test-key"
        )
        await provider.initialize()
        
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = Exception("Server error")
        
        with patch.object(provider, '_client') as mock_client:
            mock_client.post = AsyncMock(return_value=mock_response)
            
            messages = [Message.user("Hi")]
            response = await provider.chat(messages)
            
            assert "Error" in response.content

    @pytest.mark.asyncio
    async def test_perplexity_error_handling(self):
        """Test Perplexity provider handles API errors gracefully."""
        provider = PerplexityProvider(
            model="sonar-pro",
            api_key="test-key"
        )
        await provider.initialize()
        
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = Exception("Unauthorized")
        
        with patch.object(provider, '_client') as mock_client:
            mock_client.post = AsyncMock(return_value=mock_response)
            
            messages = [Message.user("Hi")]
            response = await provider.chat(messages)
            
            assert "Error" in response.content

    @pytest.mark.asyncio
    async def test_replicate_error_handling(self):
        """Test Replicate provider handles API errors gracefully."""
        provider = ReplicateProvider(
            model="meta/meta-llama-3-70b-instruct",
            api_key="r8_test_token"
        )
        await provider.initialize()
        
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.raise_for_status.side_effect = Exception("Bad request")
        
        with patch.object(provider, '_client') as mock_client:
            mock_client.post = AsyncMock(return_value=mock_response)
            
            messages = [Message.user("Hi")]
            response = await provider.chat(messages)
            
            assert "Error" in response.content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
