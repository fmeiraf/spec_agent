"""
OpenAI Client wrapper specifically for GPT-5 models.
"""

from abc import ABC, abstractmethod
from typing import Optional

import litellm


class Client(ABC):
    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def acompletion(
        self,
        messages: list[dict[str, str]] | str,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        **kwargs,
    ) -> str:
        pass


class LiteLLMClient(Client):
    def __init__(self, model: str):
        super().__init__(model)
        self.client = litellm.completion
        self.aclient = litellm.acompletion

    async def acompletion(
        self,
        messages: list[dict[str, str]] | str,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        **kwargs,
    ) -> str:
        try:
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            elif isinstance(messages, dict):
                messages = [messages]

            response = await self.aclient(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                timeout=timeout,
                **kwargs,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Error generating completion: {str(e)}")
