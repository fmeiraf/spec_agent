"""
OpenAI Client wrapper specifically for GPT-5 models.
"""

from abc import ABC, abstractmethod
from typing import Optional

import litellm
from langfuse import observe


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

    def debug_messages(self, messages: list[dict[str, str]] | str) -> None:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, dict):
            messages = [messages]

        from rich import print as rprint

        for message in messages:
            rprint(
                f"""
            role: [bold]{message["role"]}[/bold]

            content: {message["content"]}
            """,
                end="\n\n",
            )

    @observe(name="spec_agent.llm.acompletion")
    async def acompletion(
        self,
        messages: list[dict[str, str]] | str,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        debug: bool = False,
        **kwargs,
    ) -> str:
        try:
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            elif isinstance(messages, dict):
                messages = [messages]

            if debug:
                self.debug_messages(messages)

            response = await self.aclient(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                timeout=timeout,
                **kwargs,
            )

            if kwargs.get("response_format"):
                return kwargs["response_format"].model_validate_json(response.choices[0].message.content)

            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Error generating completion: {str(e)}")
