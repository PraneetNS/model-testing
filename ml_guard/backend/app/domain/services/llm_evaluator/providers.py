import abc
import requests
import json
import time
from typing import List, Dict, Any, Optional
import structlog

logger = structlog.get_logger(__name__)

class LLMProvider(abc.ABC):
    @abc.abstractmethod
    async def generate(self, prompt: str, max_tokens: int = 500) -> str:
        pass

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo", max_retries: int = 3):
        self.api_key = api_key
        self.model_name = model_name
        self.url = "https://api.openai.com/v1/chat/completions"
        self.max_retries = max_retries

    async def generate(self, prompt: str, max_tokens: int = 500) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": min(max_tokens, 2048) # Enforcement
        }
        
        last_err = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(self.url, headers=headers, json=data, timeout=30)
                if response.status_code == 429: # Rate limit
                    wait_time = (attempt + 1) * 2
                    logger.warning("OpenAI Rate Limited", attempt=attempt, wait=wait_time)
                    time.sleep(wait_time)
                    continue
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                last_err = e
                logger.error("OpenAI attempt failed", attempt=attempt, error=str(e))
                time.sleep(1)
        
        raise last_err

class HuggingFaceProvider(LLMProvider):
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        self.url = f"https://api-inference.huggingface.co/models/{model_name}"

    async def generate(self, prompt: str, max_tokens: int = 500) -> str:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        data = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "return_full_text": False
            }
        }
        
        try:
            response = requests.post(self.url, headers=headers, json=data, timeout=45)
            response.raise_for_status()
            res_json = response.json()
            
            # HF API sometimes returns a list or a dict depending on model type
            if isinstance(res_json, list) and len(res_json) > 0:
                return res_json[0].get("generated_text", "")
            if isinstance(res_json, dict) and "generated_text" in res_json:
                return res_json["generated_text"]
            
            return str(res_json)
        except Exception as e:
            logger.error("HuggingFace generation failed", model=self.model_name, error=str(e))
            raise e

class ExternalAPIProvider(LLMProvider):
    def __init__(self, endpoint_url: str, api_key: Optional[str] = None):
        self.endpoint_url = endpoint_url
        self.api_key = api_key

    async def generate(self, prompt: str, max_tokens: int = 500) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            # Standardized structure for enterprise external APIs
            payload = {
                "prompt": prompt, 
                "max_tokens": max_tokens,
                "stream": False
            }
            response = requests.post(self.endpoint_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            return response.json().get("text", response.json().get("response", ""))
        except Exception as e:
            logger.error("External API generation failed", url=self.endpoint_url, error=str(e))
            raise e

class GroqProvider(LLMProvider):
    def __init__(self, api_key: str, model_name: str = "llama3-8b-8192", max_retries: int = 3):
        self.api_key = api_key
        self.model_name = model_name
        self.url = "https://api.groq.com/openai/v1/chat/completions"
        self.max_retries = max_retries

    async def generate(self, prompt: str, max_tokens: int = 500) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens
        }
        
        last_err = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(self.url, headers=headers, json=data, timeout=30)
                if response.status_code == 429: # Rate limit
                    wait_time = (attempt + 1) * 2
                    logger.warning("Groq Rate Limited", attempt=attempt, wait=wait_time)
                    time.sleep(wait_time)
                    continue
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                last_err = e
                logger.error("Groq attempt failed", attempt=attempt, error=str(e))
                time.sleep(1)
        
        raise last_err

