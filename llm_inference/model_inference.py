# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""
Model inference utilities for different API providers.
Supports OpenAI, Together, Anthropic, Google, Mistral, Azure, etc.
"""

import os
import json
import time
import logging
from typing import Dict, Any
from openai import OpenAI
import tiktoken

logger = logging.getLogger(__name__)


class ModelInference:
    """Unified interface for calling different LLM APIs."""

    def __init__(self):
        """Initialize the model inference with API keys."""
        # Load API keys from environment
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.together_api_key = os.getenv("TOGETHER_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.mistral_api_key = os.getenv("MISTRAL_API_KEY")
        self.azure_api_key = os.getenv("AZURE_API_KEY")
        self.azure_endpoint = os.getenv("AZURE_ENDPOINT")
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        self.perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
        self.replicate_api_key = os.getenv("REPLICATE_API_KEY")

        # AWS credentials
        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

        self.gpt2_enc = tiktoken.get_encoding("gpt2")

    def infer(
        self, model_name: str, prompt: str, max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Run inference on the specified model.

        Args:
            model_name: Name of the model (e.g., "WizardLM/WizardLM-13B-V1.2")
            prompt: Input prompt for the model
            max_retries: Maximum number of retries on failure

        Returns:
            Dictionary containing response, token usage, and metadata
        """

        # Determine API provider based on model name
        provider = self._get_provider(model_name)

        for attempt in range(max_retries):
            try:
                if provider == "openai":
                    return self._call_openai(model_name, prompt)
                elif provider == "together":
                    return self._call_together(model_name, prompt)
                elif provider == "anthropic":
                    return self._call_anthropic(model_name, prompt)
                elif provider == "google":
                    return self._call_google(model_name, prompt)
                elif provider == "mistral":
                    return self._call_mistral(model_name, prompt)
                elif provider == "azure":
                    return self._call_azure(model_name, prompt)
                elif provider == "deepseek":
                    return self._call_deepseek(model_name, prompt)
                elif provider == "perplexity":
                    return self._call_perplexity(model_name, prompt)
                elif provider == "openrouter":
                    return self._call_openrouter(model_name, prompt)
                elif provider == "replicate":
                    return self._call_replicate(model_name, prompt)
                elif provider == "aws":
                    return self._call_aws(model_name, prompt)
                elif provider == "xai":
                    return self._call_xai(model_name, prompt)
                elif provider == "zhipu":
                    return self._call_zhipu(model_name, prompt)
                else:
                    # Default to Together API for most open-source models
                    return self._call_together(model_name, prompt)

            except Exception as e:
                if attempt == max_retries - 1:
                    return {
                        "response": "",
                        "error": str(e),
                        "success": False,
                        "token_usage": {
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "total_tokens": 0,
                        },
                        "provider": provider,
                        "model_used": model_name,
                    }

                backoff_time = 2**attempt
                time.sleep(backoff_time)  # Exponential backoff

        # Fallback failure result if all retries somehow did not return
        return {
            "response": "",
            "error": "Inference did not return a result",
            "success": False,
            "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "provider": provider if "provider" in locals() else "unknown",
            "model_used": model_name,
        }

    def _get_provider(self, model_name: str) -> str:
        """Determine the API provider based on model name."""

        # Model to provider mapping - you can add more models here
        model_to_provider = {
            # OpenAI models
            "gpt-3.5-turbo": "openai",
            "gpt-3.5-turbo-1106": "openai",
            "gpt-4": "openai",
            "gpt-4-turbo": "openai",
            "gpt-4.1": "openai",
            "gpt-4.1-mini": "openai",
            "gpt-4.1-nano": "openai",
            "gpt-4o": "openai",
            "gpt-4o-mini": "openai",
            "gpt-4-1106-preview": "openai",
            "o4-mini": "openai",
            "gpt-5-chat-latest": "openai",
            "gpt-5-mini": "openai",
            "gpt-5-nano": "openai",
            # Anthropic models
            "claude-3-haiku-20240307": "anthropic",
            "claude-3-7-sonnet-20250219": "anthropic",
            # Google models
            "gemini-2.0-flash-001": "google",
            "gemini-2.5-flash": "google",
            "gemini-2.5-pro": "google",
            # Mistral models
            "mistral-medium": "mistral",
            "codestral-latest": "mistral",
            "open-mixtral-8x7b": "mistral",
            "mistral-large-latest": "mistral",
            "mistral-medium-latest": "mistral",
            "mistral-small-latest": "mistral",
            "open-mistral-7b": "mistral",
            "open-mistral-nemo": "mistral",
            # DeepSeek models
            "deepseek-coder": "deepseek",
            # Together AI models
            "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": "together",
            "meta-llama/Meta-Llama-3-70B-Instruct-Turbo": "together",
            "meta-llama/Llama-3-70b-chat-hf": "together",
            # OpenRouter
            "mistralai/mixtral-8x7b-instruct": "openrouter",
            "mistralai/mistral-7b-instruct": "openrouter",
            "meta-llama/llama-3-8b-instruct": "openrouter",
            "anthropic/claude-3.5-sonnet": "openrouter",
            "Qwen/QwQ-32B": "openrouter",
            # Replicate
            "meta/codellama-34b-instruct": "replicate",
            # AWS Bedrock
            "llama-3-1-8b-instruct": "aws",
            "llama-3-2-1b-instruct": "aws",
            "llama-3-2-3b-instruct": "aws",
            "llama-3-3-70b-instruct": "aws",
            "llama-3-1-405b-instruct": "aws",
            # Zhipu
            "glm-4-air": "zhipu",
            "glm-4-flash": "zhipu",
            "glm-4-plus": "zhipu",
        }

        # Check if exact model name is in mapping
        if model_name in model_to_provider:
            return model_to_provider[model_name]
        else:
            raise ValueError(f"Model {model_name} not found in model_to_provider")

    def _call_xai(self, model_name: str, prompt: str) -> Dict[str, Any]:
        """Call XAI API."""

        from xai_sdk import Client
        from xai_sdk.chat import user, system

        client = Client(
            api_key=os.getenv("XAI_API_KEY"),
            timeout=3600,  # Override default timeout with longer timeout for reasoning models
        )

        chat = client.chat.create(model=model_name)
        chat.append(system("You are Grok, a highly intelligent, helpful AI assistant."))
        chat.append(user(prompt))

        response = chat.sample()

        return {
            "response": response.content,
            "success": True,
            "token_usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens
                + response.usage.output_tokens,
            },
            "model_used": model_name,
            "provider": "xai",
        }

    def _call_zhipu(self, model_name: str, prompt: str) -> Dict[str, Any]:
        from zhipuai import ZhipuAI

        client = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"))

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )

        return {
            "response": response.choices[0].message.content,
            "success": True,
            "token_usage": {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            "model_used": model_name,
            "provider": "zhipu",
        }

    def _call_replicate(self, model_name: str, prompt: str) -> Dict[str, Any]:
        """Call Replicate API."""
        import replicate

        client = replicate.Client(api_token=self.replicate_api_key)

        if model_name == "meta/codellama-34b-instruct":
            model_name += (
                ":eeb928567781f4e90d2aba57a51baef235de53f907c214a4ab42adabf5bb9736"
            )

        response = client.run(model_name, input={"prompt": prompt})

        # Collect the generator output into a string
        response_text = ""
        for item in response:
            response_text += str(item)

        input_tokens = len(self.gpt2_enc.encode(prompt))
        output_tokens = len(self.gpt2_enc.encode(response_text))
        total_tokens = input_tokens + output_tokens

        return {
            "response": response_text,
            "success": True,
            "token_usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            },
            "model_used": model_name,
            "provider": "replicate",
        }

    def _call_openrouter(self, model_name: str, prompt: str) -> Dict[str, Any]:
        """Call OpenRouter API."""
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
        )

        response = client.chat.completions.create(
            model=model_name, messages=[{"role": "user", "content": prompt}]
        )

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) if usage is not None else 0
        completion_tokens = (
            getattr(usage, "completion_tokens", 0) if usage is not None else 0
        )
        total_tokens = (
            getattr(usage, "total_tokens", 0)
            if usage is not None
            else input_tokens + completion_tokens
        )

        return {
            "response": response.choices[0].message.content,
            "success": True,
            "token_usage": {
                "input_tokens": input_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            "model_used": model_name,
            "provider": "openrouter",
        }

    def _call_openai(self, model_name: str, prompt: str) -> Dict[str, Any]:
        """Call OpenAI API."""
        import openai

        client = openai.OpenAI(api_key=self.openai_api_key)

        response = client.chat.completions.create(
            model=model_name.replace("openai/", ""),
            messages=[{"role": "user", "content": prompt}],
        )

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) if usage is not None else 0
        completion_tokens = (
            getattr(usage, "completion_tokens", 0) if usage is not None else 0
        )
        total_tokens = (
            getattr(usage, "total_tokens", 0)
            if usage is not None
            else input_tokens + completion_tokens
        )

        return {
            "response": response.choices[0].message.content,
            "success": True,
            "token_usage": {
                "input_tokens": input_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            "model_used": model_name,
            "provider": "openai",
        }

    def _call_together(self, model_name: str, prompt: str) -> Dict[str, Any]:
        """Call Together AI API."""
        import together

        client = together.Together(api_key=self.together_api_key)

        # Clean up model name for Together API
        clean_model_name = model_name.replace("togetherai/", "").replace(
            "together/", ""
        )

        response = client.chat.completions.create(
            model=clean_model_name, messages=[{"role": "user", "content": prompt}]
        )

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) if usage is not None else 0
        completion_tokens = (
            getattr(usage, "completion_tokens", 0) if usage is not None else 0
        )
        total_tokens = (
            getattr(usage, "total_tokens", 0)
            if usage is not None
            else input_tokens + completion_tokens
        )

        return {
            "response": response.choices[0].message.content,
            "success": True,
            "token_usage": {
                "input_tokens": input_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            "model_used": model_name,
            "provider": "together",
        }

    def _call_anthropic(self, model_name: str, prompt: str) -> Dict[str, Any]:
        """Call Anthropic API."""
        import anthropic

        client = anthropic.Anthropic(api_key=self.anthropic_api_key)

        clean_model_name = model_name.replace("anthropic/", "")

        response = client.messages.create(
            model=clean_model_name,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", 0) if usage is not None else 0
        output_tokens = getattr(usage, "output_tokens", 0) if usage is not None else 0
        total_tokens = input_tokens + output_tokens

        content0 = response.content[0]
        text = getattr(content0, "text", str(content0))
        return {
            "response": text,
            "success": True,
            "token_usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            },
            "model_used": model_name,
            "provider": "anthropic",
        }

    def _call_google(self, model_name: str, prompt: str) -> Dict[str, Any]:
        """Call Google AI API."""
        import google.generativeai as genai

        genai.configure(api_key=self.google_api_key)

        clean_model_name = model_name.replace("google/", "")
        model = genai.GenerativeModel(clean_model_name)

        response = model.generate_content(prompt)

        # Google doesn't provide detailed token usage in the free tier
        # Estimate tokens roughly
        input_tokens = len(prompt.split()) * 1.3
        output_tokens = len(response.text.split()) * 1.3 if response.text else 0

        return {
            "response": response.text,
            "success": True,
            "token_usage": {
                "input_tokens": int(input_tokens),
                "output_tokens": int(output_tokens),
                "total_tokens": int(input_tokens + output_tokens),
            },
            "model_used": model_name,
            "provider": "google",
        }

    def _call_mistral(self, model_name: str, prompt: str) -> Dict[str, Any]:
        """Call Mistral AI API."""
        from mistralai import Mistral

        client = Mistral(api_key=self.mistral_api_key)

        clean_model_name = model_name.replace("mistral/", "")

        from typing import Any, cast

        response = client.chat.complete(
            model=clean_model_name,
            messages=cast(
                Any,
                [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            ),
            max_tokens=2048,
            temperature=0.7,
        )

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) if usage is not None else 0
        completion_tokens = (
            getattr(usage, "completion_tokens", 0) if usage is not None else 0
        )
        total_tokens = (
            getattr(usage, "total_tokens", 0)
            if usage is not None
            else input_tokens + completion_tokens
        )

        return {
            "response": response.choices[0].message.content,
            "success": True,
            "token_usage": {
                "input_tokens": input_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            "model_used": model_name,
            "provider": "mistral",
        }

    def _call_azure(self, model_name: str, prompt: str) -> Dict[str, Any]:
        """Call Azure OpenAI API."""
        import openai

        client = openai.AzureOpenAI(
            api_key=self.azure_api_key,
            api_version="2023-12-01-preview",
            azure_endpoint=self.azure_endpoint,
        )

        # For Azure, model_name should be the deployment name
        deployment_name = model_name.replace("azure/", "")

        response = client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.7,
        )

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) if usage is not None else 0
        completion_tokens = (
            getattr(usage, "completion_tokens", 0) if usage is not None else 0
        )
        total_tokens = (
            getattr(usage, "total_tokens", 0)
            if usage is not None
            else input_tokens + completion_tokens
        )

        return {
            "response": response.choices[0].message.content,
            "success": True,
            "token_usage": {
                "input_tokens": input_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            "model_used": model_name,
            "provider": "azure",
        }

    def _call_deepseek(self, model_name: str, prompt: str) -> Dict[str, Any]:
        """Call DeepSeek API."""
        import openai

        # DeepSeek uses OpenAI-compatible API
        client = openai.OpenAI(
            api_key=self.deepseek_api_key, base_url="https://api.deepseek.com"
        )

        clean_model_name = model_name.replace("deepseek/", "")

        response = client.chat.completions.create(
            model=clean_model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.7,
        )

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) if usage is not None else 0
        completion_tokens = (
            getattr(usage, "completion_tokens", 0) if usage is not None else 0
        )
        total_tokens = (
            getattr(usage, "total_tokens", 0)
            if usage is not None
            else input_tokens + completion_tokens
        )

        return {
            "response": response.choices[0].message.content,
            "success": True,
            "token_usage": {
                "input_tokens": input_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            "model_used": model_name,
            "provider": "deepseek",
        }

    def _call_perplexity(self, model_name: str, prompt: str) -> Dict[str, Any]:
        """Call Perplexity API."""
        import openai

        # Perplexity uses OpenAI-compatible API
        client = openai.OpenAI(
            api_key=self.perplexity_api_key, base_url="https://api.perplexity.ai"
        )

        clean_model_name = model_name.replace("perplexity/", "")

        response = client.chat.completions.create(
            model=clean_model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.7,
        )

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) if usage is not None else 0
        completion_tokens = (
            getattr(usage, "completion_tokens", 0) if usage is not None else 0
        )
        total_tokens = (
            getattr(usage, "total_tokens", 0)
            if usage is not None
            else input_tokens + completion_tokens
        )

        return {
            "response": response.choices[0].message.content,
            "success": True,
            "token_usage": {
                "input_tokens": input_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            "model_used": model_name,
            "provider": "perplexity",
        }

    def _call_aws(self, model_name: str, prompt: str) -> Dict[str, Any]:
        """Call AWS Bedrock API."""
        import boto3
        from botocore.exceptions import ClientError

        # Map model names to their inference profile ARNs
        model_arn_mapping = {
            "llama-3-1-8b-instruct": "arn:aws:bedrock:us-east-2:287882045629:inference-profile/us.meta.llama3-1-8b-instruct-v1:0",
            "llama-3-2-1b-instruct": "arn:aws:bedrock:us-east-2:287882045629:inference-profile/us.meta.llama3-2-1b-instruct-v1:0",
            "llama-3-2-3b-instruct": "arn:aws:bedrock:us-east-2:287882045629:inference-profile/us.meta.llama3-2-3b-instruct-v1:0",
            "llama-3-3-70b-instruct": "arn:aws:bedrock:us-east-2:287882045629:inference-profile/us.meta.llama3-3-70b-instruct-v1:0",
            "llama-3-1-405b-instruct": "arn:aws:bedrock:us-east-2:287882045629:inference-profile/us.meta.llama3-1-405b-instruct-v1:0",
        }

        # Use bedrock-runtime client with explicit credentials
        runtime_kwargs = {"region_name": "us-east-2"}
        if self.aws_access_key_id and self.aws_secret_access_key:
            runtime_kwargs["aws_access_key_id"] = self.aws_access_key_id
            runtime_kwargs["aws_secret_access_key"] = self.aws_secret_access_key

        runtime = boto3.client("bedrock-runtime", **runtime_kwargs)

        # Get the inference profile ARN
        inference_profile_arn = model_arn_mapping.get(model_name)
        if not inference_profile_arn:
            raise ValueError(f"Model {model_name} not found in AWS Bedrock mapping")

        # Format prompt for Llama 3/3.1 native format
        formatted_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{prompt}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

        body = {
            "prompt": formatted_prompt,
        }

        try:
            response = runtime.invoke_model(
                modelId=inference_profile_arn,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )
            payload = json.loads(response["body"].read())

            # Extract the generated text
            generated_text = payload.get("generation", "")

            # Extract token usage from response headers
            response_headers = response.get("ResponseMetadata", {}).get(
                "HTTPHeaders", {}
            )
            input_tokens = int(
                response_headers.get("x-amzn-bedrock-input-token-count", 0)
            )
            output_tokens = int(
                response_headers.get("x-amzn-bedrock-output-token-count", 0)
            )
            total_tokens = input_tokens + output_tokens

            return {
                "response": generated_text,
                "success": True,
                "token_usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                },
                "model_used": model_name,
                "provider": "aws",
            }

        except ClientError as e:
            return {
                "response": "",
                "error": f"AWS Bedrock error: {str(e)}",
                "success": False,
                "token_usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                },
                "model_used": model_name,
                "provider": "aws",
            }
