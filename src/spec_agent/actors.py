import uuid
from abc import ABC, abstractmethod
from typing import Any, Literal, Optional, Set

from pydantic import BaseModel, Field, model_validator

from spec_agent.llm import Client, LiteLLMClient
from spec_agent.spec import Spec, SubTask, SubTaskReview


class ActorAttributes(BaseModel):
    system_prompt: str
    user_prompt: str
    type: str = Field(description="The type of the actor")
    allowed_types: Set[str] = Field(description="The types of the actor that are allowed to be performed by the actor")
    job_description: str = Field(description="The description of the jobs the actor will be performing")
    specialty: str = Field(
        description="Instructions on what situation the actor is specialized in and what situations if should be deployed in or not."
    )

    @model_validator(mode="after")
    def validate_allowed_types(self) -> "ActorConfig":
        if self.type not in self.allowed_types:
            raise ValueError(f"The type {self.type} is not allowed to be performed by the actor")
        return self


class ActorConfig(BaseModel):
    attributes: ActorAttributes  # separating so this could be easily used in LLM calls
    model: str
    actor_type: Literal["reviewer", "worker"]
    temperature: Optional[float] = None
    max_retries: int = 1
    message_history: list[dict[str, str]] = Field(
        default_factory=list, description="Chat history for all the tasks the actor has performed"
    )


class Actor(ABC):
    def __init__(self, config: ActorConfig, llm: Client = LiteLLMClient()):
        self.config = config
        self.id = uuid.uuid4()
        self.message_history = config.message_history
        self.llm = llm(model=config.model)

    def __repr__(self) -> str:
        return f"{self.config.actor_type.capitalize()} {self.__class__.__name__}(id={self.id}) \n {self.config.attributes.model_dump_json(indent=2)}"


# --- Worker registry and registration helper ---
# Maps task type -> Worker class
WORKER_REGISTRY: dict[str, type["Worker"]] = {}


def registers_worker(worker_type: str):
    """Decorator to register a Worker implementation for a given task type."""

    def _wrap(cls: type["Worker"]):
        WORKER_REGISTRY[worker_type] = cls
        return cls

    return _wrap


class Reviewer(Actor):
    def __init__(self, config: ActorConfig, llm: Client = LiteLLMClient()):
        if not config.actor_type == "reviewer":
            raise ValueError("Reviewer must be a reviewer")

        super().__init__(config, llm)

        # Build and keep workers for allowed types that are registered
        self._workers: dict[str, Worker] = {}
        for task_type in self.config.attributes.allowed_types:
            worker_cls = WORKER_REGISTRY.get(task_type)
            if worker_cls is None:
                continue  # silently skip unknown types (or raise if desired)

            # Derive a worker config from reviewer config
            worker_cfg = config.model_copy(deep=True)
            worker_cfg.actor_type = "worker"
            worker_cfg.attributes.type = task_type

            worker_instance = worker_cls(worker_cfg, llm)
            self._workers[task_type] = worker_instance

            # Optional: dynamic attribute e.g., reviewer.worker_code
            setattr(self, f"worker_{task_type}", worker_instance)

    def get_worker(self, task_type: str) -> "Worker":
        worker = self._workers.get(task_type)
        if worker is None:
            known = ", ".join(sorted(self._workers.keys()))
            raise KeyError(f"Unknown task_type '{task_type}'. Known: {known}")
        return worker

    @abstractmethod
    async def handle_first_assignment(self, spec: Spec, **kwargs: Any) -> SubTask:
        pass

    @abstractmethod
    async def review(self, subtask: SubTask, **kwargs: Any) -> SubTaskReview:
        pass


class Worker(Actor):
    def __init__(self, config: ActorConfig, llm: Client = LiteLLMClient()):
        if not config.actor_type == "worker":
            raise ValueError("Worker must be a worker")

        super().__init__(config, llm)

    @abstractmethod
    async def perform_work(self, subtask: SubTask, **kwargs: Any) -> SubTask:
        pass
