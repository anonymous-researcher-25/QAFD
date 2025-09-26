import sys

if sys.version_info < (3, 9):
    from typing import AsyncIterator
else:
    from collections.abc import AsyncIterator

import pipmaster as pm
if not pm.is_installed("openai"):
    pm.install("openai")

from openai import (
    AsyncOpenAI,
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

import numpy as np
from typing import Union, Dict, Any
from .utils import logger


# LLM Configuration
LLM_CONFIGS = {
    "llama": {
        "api_url": "http://105.144.47.80:8001/v1",
        "model": "llama70b",
        "api_key": "dummy",
        "provider": "local",
        "model_name": "llama"
    },
    "qwen": {
        "api_url": "http://105.144.47.80:8001/v1",
        "model": "llm_base_model",
        "api_key": "dummy",
        "provider": "vllm",
        "model_name": "qwen"
    }
}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)
async def _local_model_if_cache(
    model_config: Dict[str, Any],
    prompt: str,
    system_prompt: str = None,
    history_messages: list = [],
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    """Complete a prompt using local model API with caching support.
    
    Args:
        model_config: Configuration dictionary for the model
        prompt: The prompt to complete
        system_prompt: Optional system prompt
        history_messages: Optional list of previous messages
        **kwargs: Additional keyword arguments
        
    Returns:
        The completed text or an async iterator of text chunks if streaming
    """
    stream = kwargs.get("stream", False)
    
    # Extract configuration
    api_url = model_config["api_url"]
    model = model_config["model"]
    api_key = model_config["api_key"]
    
    # Remove special kwargs
    kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)
    kwargs.pop("max_tokens", None)  # Let the model decide
    kwargs.pop("stream", None)  # Remove stream to avoid duplicate parameter
    
    # Create OpenAI-compatible client
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "QAFD_RAG/1.0",
    }
    if api_key and api_key != "dummy":
        headers["Authorization"] = f"Bearer {api_key}"
    
    openai_client = AsyncOpenAI(
        base_url=api_url,
        api_key=api_key if api_key != "dummy" else "sk-dummy",
        default_headers=headers,
        timeout=300.0  # 5 minutes timeout
    )
    
    try:
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})
        
        logger.debug(f"===== Local Model Query =====")
        logger.debug(f"Model: {model}")
        logger.debug(f"API URL: {api_url}")
        logger.debug(f"Prompt: {prompt}")
        
        # Make the API call
        response = await openai_client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream,
            **kwargs
        )
        
        if stream:
            async def inner():
                try:
                    async for chunk in response:
                        if chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                except Exception as e:
                    logger.error(f"Error in stream response: {str(e)}")
                    raise
                finally:
                    try:
                        await openai_client.close()
                        logger.debug("Successfully closed local model client for streaming")
                    except Exception as close_error:
                        logger.warning(f"Failed to close local model client: {close_error}")
            
            return inner()
        else:
            content = response.choices[0].message.content
            await openai_client.close()
            return content
            
    except APIConnectionError as e:
        logger.error(f"Local Model API Connection Error: {e}")
        await openai_client.close()
        raise
    except RateLimitError as e:
        logger.error(f"Local Model API Rate Limit Error: {e}")
        await openai_client.close()
        raise
    except APITimeoutError as e:
        logger.error(f"Local Model API Timeout Error: {e}")
        await openai_client.close()
        raise
    except Exception as e:
        logger.error(f"Local Model API Error: {e}")
        try:
            await openai_client.close()
        except:
            pass
        raise


async def local_model_complete(
    prompt: str,
    system_prompt: str = None,
    history_messages: list = [],
    keyword_extraction: bool = False,
    model_name: str = "llama",  # Default to llama
    **kwargs
) -> Union[str, AsyncIterator[str]]:
    """Complete a prompt using local model.
    
    Args:
        prompt: The prompt to complete
        system_prompt: Optional system prompt
        history_messages: Optional list of previous messages
        keyword_extraction: Whether to extract keywords (not supported for local models)
        model_name: Name of the model to use ("qwen" or "llama")
        **kwargs: Additional keyword arguments
        
    Returns:
        The completed text or an async iterator of text chunks if streaming
    """
    if model_name not in LLM_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(LLM_CONFIGS.keys())}")
    
    model_config = LLM_CONFIGS[model_name]
    
    return await _local_model_if_cache(
        model_config,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs
    )


async def qwen_complete(
    prompt: str,
    system_prompt: str = None,
    history_messages: list = [],
    keyword_extraction: bool = False,
    **kwargs
) -> Union[str, AsyncIterator[str]]:
    """Complete a prompt using qwen local model."""
    return await local_model_complete(
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        keyword_extraction=keyword_extraction,
        model_name="qwen",
        **kwargs
    )


async def llama_complete(
    prompt: str,
    system_prompt: str = None,
    history_messages: list = [],
    keyword_extraction: bool = False,
    **kwargs
) -> Union[str, AsyncIterator[str]]:
    """Complete a prompt using Llama local model."""
    return await local_model_complete(
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        keyword_extraction=keyword_extraction,
        model_name="llama",
        **kwargs
    )



if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Test the local model
        result = await llama_complete("How are you?")
        print(f"Llama response: {result}")
        
        result = await qwen_complete("What is 2+2?")
        print(f"qwen response: {result}")
    
    asyncio.run(main()) 