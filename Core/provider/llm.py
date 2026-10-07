# Copyright (C) 2025-2026 Shu Wang
# SPDX-License-Identifier: Apache-2.0 OR AGPL-3.0-only

from typing import Dict, Optional, Any, List, Union
from abc import ABC, abstractmethod
from openai import OpenAI
import ollama
import logging
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, as_completed
from Core.Common.Memory import Memory
from Core.configs.llm_config import LLMConfig
from Core.utils.utils import get_max_output_tokens
from Core.provider.TokenTracker import TokenTracker
import time
import threading

log = logging.getLogger(__name__)

def is_quota_error(exc: Exception) -> bool:
    """Return True for API quota/rate-limit failures that should stop immediately."""
    text = str(exc).lower()

    markers = (
        "resource_exhausted",
        "quota exceeded",
        "error code: 429",
        "status code: 429",
        "too many requests",
    )

    return any(marker in text for marker in markers)


def is_transient_api_error(exc: Exception) -> bool:
    """
    Return True for temporary API/network failures.

    These may be retried with bounded exponential backoff,
    but must never be silently converted into an empty result.
    """
    text = str(exc).lower()

    markers = (
        "error code: 500",
        "error code: 502",
        "error code: 503",
        "error code: 504",
        "status code: 500",
        "status code: 502",
        "status code: 503",
        "status code: 504",
        "service unavailable",
        "high demand",
        "connection error",
        "connection timeout",
        "timed out",
        "timeout",
        "apiconnectionerror",
        "apitimeouterror",
        "internalservererror",
    )

    return any(marker in text for marker in markers)

class BaseLLMController(ABC):
    @abstractmethod
    def _prepare_messages(
        self, prompt_or_memory: Union[str, Memory], images: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Prepares the message list in the format required by the specific API.
        This must be implemented by each subclass.
        """
        pass

    @abstractmethod
    def get_completion(self, prompt_or_memory: Union[str, Memory]) -> str:
        """Get completion from LLM."""
        pass

    @abstractmethod
    def get_json_completion(
        self,
        prompt_or_memory: Union[str, Memory],
        schema: BaseModel,
        images: Optional[List[str]] = None,
    ) -> dict:
        """Get structured JSON response from LLM using Pydantic schema."""
        pass


class OpenAIController(BaseLLMController):
    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
    ):
        self.model = (
            llm_config.model_name
            if llm_config and hasattr(llm_config, "model_name")
            else "gpt-3.5-turbo"
        )
        self.max_tokens = (
            llm_config.max_tokens
            if llm_config and hasattr(llm_config, "max_tokens")
            else 4000
        )
        self.temperature = (
            llm_config.temperature
            if llm_config and hasattr(llm_config, "temperature")
            else 0.7
        )
        base_url = (
            llm_config.api_base
            if llm_config and hasattr(llm_config, "api_base")
            else None
        )

        # Gemini's OpenAI-compatible endpoint does not accept some
        # Qwen/vLLM-specific request fields used by the original BookRAG setup.
        self.is_gemini = bool(
            base_url and "generativelanguage.googleapis.com" in base_url
        )
        # Gemini free-tier throttling.
        # Observed project limit: 5 requests/minute.
        # 13 seconds gives a small safety margin over the theoretical 12 seconds.
        self.min_request_interval = 13.0 if self.is_gemini else 0.0
        self._last_request_time = 0.0
        self._request_lock = threading.Lock()

        self.frequency_penalty = (
            llm_config.frequency_penalty
            if llm_config and hasattr(llm_config, "frequency_penalty")
            else 0.0
        )
        self.presence_penalty = (
            llm_config.presence_penalty
            if llm_config and hasattr(llm_config, "presence_penalty")
            else 0.0
        )

        if base_url is not None:
            if self.is_gemini:
                self.client = OpenAI(
                    api_key=llm_config.api_key,
                    base_url=base_url,
                    max_retries=0,
                    timeout=60.0,
                )
            else:
                self.client = OpenAI(
                    api_key=llm_config.api_key,
                    base_url=base_url,
                )
        else:
            self.client = OpenAI(api_key=llm_config.api_key)

    def _wait_for_rate_limit(self):
        """Space Gemini requests to stay under the observed free-tier RPM limit."""
        if not self.is_gemini or self.min_request_interval <= 0:
            return

        with self._request_lock:
            now = time.monotonic()
            elapsed = now - self._last_request_time
            wait_time = self.min_request_interval - elapsed

            if wait_time > 0:
                log.info(
                    f"Gemini rate-limit throttle: waiting "
                    f"{wait_time:.1f} seconds before next request."
                )
                time.sleep(wait_time)

            self._last_request_time = time.monotonic()

    def _prepare_messages(
        self, prompt_or_memory: Union[str, Memory], add_system_prompt: bool = True
    ) -> List[Dict[str, Any]]:
        """Prepares messages for the OpenAI API format."""
        if isinstance(prompt_or_memory, Memory):
            res_dict = [msg.to_dict() for msg in prompt_or_memory.storage]
            return res_dict

        messages = []
        if add_system_prompt:
            messages.append(
                {"role": "system", "content": "You are a helpful assistant."}
            )
        messages.append({"role": "user", "content": prompt_or_memory})
        return messages

    def get_completion(
        self, prompt_or_memory: Union[str, Memory], json_response: bool = False
    ) -> str:
        messages = self._prepare_messages(prompt_or_memory)
        parameters = {
            "model": self.model,
            "messages": messages,
        }

        # Preserve BookRAG's original Qwen/vLLM request behavior, while
        # sending a clean OpenAI-compatible payload to Gemini.
        if not self.is_gemini:
            parameters.update(
                {
                    "temperature": self.temperature,
                    "max_tokens": get_max_output_tokens(messages, self.max_tokens),
                    "frequency_penalty": self.frequency_penalty,
                    "presence_penalty": self.presence_penalty,
                    "extra_body": {
                        "chat_template_kwargs": {"enable_thinking": False}
                    },
                }
            )
        if json_response:
            parameters["response_format"] = {"type": "json_object"}

        if self.is_gemini:
            log.info("Sending Gemini completion request...")

        self._wait_for_rate_limit()
        response = self.client.chat.completions.create(**parameters)

        if response.usage:
            tracker = TokenTracker.get_instance()
            tracker.add_usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            )

        return response.choices[0].message.content


    def get_json_completion(
        self,
        prompt_or_memory: Union[str, Memory],
        schema: BaseModel,
        images: Optional[list] = None,
        think_mode: bool = False,
    ) -> dict:
        """
        Get structured JSON response from OpenAI using Pydantic schema.
        :param prompt: str
        :param schema: Pydantic BaseModel
        :param images: Optional[list] = None
        :return: dict
        """
        if isinstance(prompt_or_memory, Memory):
            # If it's a memory object, use its history.
            # We will attach images to the content of the *last* message.
            messages = [msg.to_dict() for msg in prompt_or_memory.storage]
            last_content = messages[-1]["content"]

            content_list = [{"type": "text", "text": last_content}]
            if images:
                for img_url in images:
                    content_list.append(
                        {"type": "image_url", "image_url": {"url": img_url}}
                    )
            messages[-1]["content"] = content_list

        else:  # It's a simple string prompt
            content_list = [{"type": "text", "text": prompt_or_memory}]
            if images:
                for img_url in images:
                    content_list.append(
                        {"type": "image_url", "image_url": {"url": img_url}}
                    )
            messages = [{"role": "user", "content": content_list}]

        parse_parameters = {
            "model": self.model,
            "messages": messages,
            "response_format": schema,
        }

        # chat_template_kwargs is specific to the original Qwen/vLLM path.
        # Gemini structured output uses the OpenAI-compatible parse interface
        # without those extra fields.
        if not self.is_gemini:
            parse_parameters.update(
                {
                    "temperature": self.temperature,
                    "extra_body": {
                        "chat_template_kwargs": {"enable_thinking": think_mode}
                    },
                }
            )

        if self.is_gemini:
            log.info("Sending Gemini structured JSON request...")

        self._wait_for_rate_limit()
        completion = self.client.beta.chat.completions.parse(**parse_parameters)

        if completion.usage:
            tracker = TokenTracker.get_instance()
            tracker.add_usage(
                prompt_tokens=completion.usage.prompt_tokens,
                completion_tokens=completion.usage.completion_tokens,
            )

        message = completion.choices[0].message
        if hasattr(message, "parsed") and message.parsed:
            return message.parsed
        elif hasattr(message, "refusal") and message.refusal:
            return {"refusal": message.refusal}
        else:
            return {}


class OllamaController(BaseLLMController):
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        self.model = (
            llm_config.model_name
            if llm_config and hasattr(llm_config, "model_name")
            else "llama3.1"
        )
        self.max_tokens = (
            llm_config.max_tokens
            if llm_config and hasattr(llm_config, "max_tokens")
            else 4000
        )
        self.temperature = (
            llm_config.temperature
            if llm_config and hasattr(llm_config, "temperature")
            else 0.7
        )
        self.frequency_penalty = (
            llm_config.frequency_penalty
            if llm_config and hasattr(llm_config, "frequency_penalty")
            else 0.0
        )
        self.presence_penalty = (
            llm_config.presence_penalty
            if llm_config and hasattr(llm_config, "presence_penalty")
            else 0.0
        )
        self.api_base = (
            llm_config.api_base
            if llm_config and hasattr(llm_config, "api_base")
            else None
        )
        self.api_key = (
            llm_config.api_key
            if llm_config and hasattr(llm_config, "api_key")
            else None
        )
        self.client = ollama.Client(host=self.api_base)

    def _prepare_messages(
        self, prompt_or_memory: Union[str, Memory], add_system_prompt: bool = True
    ) -> List[Dict[str, Any]]:
        """Prepares messages for the Ollama API format."""
        if isinstance(prompt_or_memory, Memory):
            return prompt_or_memory.get()

        messages = []
        if add_system_prompt:
            messages.append(
                {"role": "system", "content": "You are a helpful assistant."}
            )
        messages.append({"role": "user", "content": prompt_or_memory})
        return messages

    def get_completion(
        self, prompt_or_memory: Union[str, Memory], json_response: bool = False
    ) -> str:
        messages = self._prepare_messages(prompt_or_memory)
        response = self.client.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": self.temperature,
            },
            think=False,
        )

        if response:
            tracker = TokenTracker.get_instance()
            tracker.add_usage(
                prompt_tokens=response.get("prompt_eval_count", 0),
                completion_tokens=response.get("eval_count", 0),
            )

        return response["message"]["content"]

    def get_json_completion(
        self,
        prompt_or_memory: Union[str, Memory],
        schema: BaseModel,
        images: Optional[list] = None,
        think_mode: bool = False,
    ) -> dict:
        """
        Get structured JSON response from Ollama using Pydantic schema.
        :param prompt: str
        :param schema: Pydantic BaseModel
        :param images: Optional[list] = None
        :return: dict
        """
        messages = self._prepare_messages(prompt_or_memory)

        if images:
            # Ollama expects images at the top level of the message dictionary
            messages[-1]["images"] = images
        response = self.client.chat(
            model=self.model,
            messages=messages,
            format=schema.model_json_schema(),  # 关键：传递schema
            options={
                "temperature": self.temperature,
            },
        )

        if response:
            tracker = TokenTracker.get_instance()
            prompt_tokens = response.get("prompt_eval_count", 0)
            completion_tokens = response.get("eval_count", 0)
            try:
                prompt_tokens = int(prompt_tokens)
            except Exception as e:
                logging.error(f"Error converting prompt_tokens: {e}")
                prompt_tokens = 0

            try:
                completion_tokens = int(completion_tokens)
            except Exception as e:
                logging.error(f"Error converting completion_tokens: {e}")
                completion_tokens = 0
            tracker.add_usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        return schema.model_validate_json(response["message"]["content"])


class LLM:
    """LLM-based controller for memory metadata generation"""

    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
    ):
        if llm_config is None:
            raise ValueError("Config must be provided")
        if not isinstance(llm_config, LLMConfig):
            config = LLMConfig(**llm_config)
        else:
            config = llm_config
        self.config: LLMConfig = config
        self.max_workers = config.max_workers
        backend = config.backend
        if backend == "openai":
            self.llm = OpenAIController(llm_config=config)
        elif backend == "ollama":
            self.llm = OllamaController(llm_config=config)
        else:
            raise ValueError("Backend must be one of: 'openai', 'ollama'")

    def get_completion(
        self, prompt: Union[str, Memory], json_response: bool = False
    ) -> str:
        retry = 0
        max_retries = 3
        while retry < max_retries:
            try:
                res = self.llm.get_completion(prompt, json_response)
                if len(res.strip()) == 0:
                    raise ValueError("Empty response from LLM")
                return res
            except Exception as e:
                if is_quota_error(e):
                    log.error(
                        "LLM quota/rate limit reached. "
                        "Stopping immediately instead of retrying."
                    )
                    raise

                if is_transient_api_error(e):
                    retry += 1

                    if retry >= max_retries:
                        log.error(
                            "Transient Gemini/API failure persisted after "
                            "maximum retries. Stopping safely."
                        )
                        raise

                    wait_time = 5 * (2 ** (retry - 1))
                    log.warning(
                        f"Transient Gemini/API error: {e}. "
                        f"Retrying in {wait_time} seconds "
                        f"({retry}/{max_retries})."
                    )
                    time.sleep(wait_time)
                    continue

                print(f"Error getting completion: {e}")
                retry += 1
                time.sleep(1)
                if retry >= max_retries:
                    raise RuntimeError(
                        "Failed to get completion after multiple retries"
                    )
        # If we reach here, it means we failed to get a response after retries
        logging.error("Max retries reached, returning empty response.")
        # Log the error and return an empty string or handle it as needed
        logging.error(f"Error: {e}")
        logging.error("Returning empty response.")
        return ""

    def batch_get_completion(
        self, prompts: List[Union[str, Memory]], json_response: bool = False
    ) -> list:
        """ """
        results = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(self.get_completion, prompt, json_response): idx
                for idx, prompt in enumerate(prompts)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    if is_quota_error(e) or is_transient_api_error(e):
                        log.error(
                            "Quota/rate-limit or persistent transient API failure "
                            "reached during batch completion. "
                            "Cancelling pending requests and stopping."
                        )
                        for pending_future in future_to_idx:
                            if not pending_future.done():
                                pending_future.cancel()
                        raise

                    results[idx] = f"Error: {e}"
        return results

    def get_json_completion(
        self,
        prompt: Union[str, Memory],
        schema: BaseModel,
        images: Optional[list] = None,
        think_mode: bool = False,
    ) -> dict:
        retry = 0
        max_retries = 3
        while retry < max_retries:
            try:
                res = self.llm.get_json_completion(prompt, schema, images, think_mode=think_mode)
                if not res:
                    raise ValueError("Empty response from LLM")
                return res
            except Exception as e:
                if is_quota_error(e):
                    log.error(
                        "LLM quota/rate limit reached. "
                        "Stopping immediately instead of retrying."
                    )
                    raise

                if is_transient_api_error(e):
                    retry += 1

                    if retry >= max_retries:
                        log.error(
                            "Transient Gemini/API failure persisted after "
                            "maximum retries. Stopping safely."
                        )
                        raise

                    wait_time = 5 * (2 ** (retry - 1))
                    log.warning(
                        f"Transient Gemini/API error: {e}. "
                        f"Retrying in {wait_time} seconds "
                        f"({retry}/{max_retries})."
                    )
                    time.sleep(wait_time)
                    continue

                print(f"Error getting JSON completion: {e}")
                retry += 1
                time.sleep(1)
                if retry >= max_retries:
                    raise RuntimeError(
                        "Failed to get JSON completion after multiple retries"
                    )
        logging.error("Max retries reached, returning empty dict.")
        logging.error(f"Error: {e}")
        logging.error("Returning empty dict.")
        return {}


if __name__ == "__main__":
    # Example usage
    print("Testing LLM controller...")

    # from Core.configs.system_config import load_system_config

    # cfg = load_system_config("/home/wangshu/multimodal/GBC-RAG/config/default.yaml")
    # cfg.llm.max_workers = 4
    # controller = LLM(llm_config=cfg.llm)
    # prompt = "respond in 20 words. who are you?"

    # token_tracker = TokenTracker.get_instance()
    # token_tracker.reset()

    # start_time = time.time()
    # response = controller.batch_get_completion(prompts=[prompt] * 8)
    # end_time = time.time()
    # print(response)
    # print(f"Response time: {end_time - start_time:.2f} seconds")

    # batch_cost = token_tracker.record_stage("batch_get_completion")
    # print(f"Batch cost: {batch_cost}")

    # tmp_memory = Memory()
    # tmp_memory.add_batch(
    #     [
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {"role": "user", "content": "What is the capital of France?"},
    #     ]
    # )
    # response = controller.get_completion(tmp_memory)
    # print(response)
    # memory_cost = token_tracker.record_stage("get_completion")
    # print(f"Memory cost: {memory_cost}")
    # token_tracker.print_all_stages()