import inspect
import uuid
from abc import ABC, abstractmethod
from logging import getLogger
from typing import Any, Callable, List, Optional, Type, Union

from jinja2 import Template
from pydantic import BaseModel, Field

from spec_agent.llm import Client, LiteLLMClient
from spec_agent.prompts import SUPERVISOR_SYSTEM_PROMPT, WORKER_SYSTEM_PROMPT
from spec_agent.spec import Spec, SubTask, SubTaskReview

logger = getLogger(__name__)


class Profile(BaseModel):
    actor_name: str = Field(description="The name of the actor")
    specialty: str = Field(description="A detailed description of what the actor is specialized in")


class ActorConfig(BaseModel):
    model: str
    max_retries: int = 1
    llm_kwargs: dict[str, Any] = Field(default_factory=dict, description="Additional kwargs to pass to the LLM")


def _parse_class_config(nested: Any) -> ActorConfig:
    # Accept ActorConfig, dict, pydantic BaseModel, or a nested class with attributes
    if isinstance(nested, ActorConfig):
        return nested

    if isinstance(nested, dict):
        return ActorConfig(**dict[str, Any](nested))

    if isinstance(nested, BaseModel):
        return ActorConfig(**nested.model_dump())

    if isinstance(nested, type):
        return ActorConfig(**{k: v for k, v in vars(nested).items() if not k.startswith("_") and not callable(v)})

    raise TypeError("Unsupported nested config. Use an ActorConfig, dict, BaseModel, or a nested class.")


def _validate_abstract_methods_have_kwargs(cls: type) -> None:
    """Validate that all abstract methods in a class have **kwargs in their signature.

    Raises TypeError if any abstract method is missing **kwargs.

    This validation runs when a subclass of Actor is defined, ensuring that:
    - Any abstract method still abstract in this class must have **kwargs
    - Any method marked with @abstractmethod in base classes must have **kwargs when implemented
    """
    # Collect methods to check:
    # 1. Methods still abstract in this class
    # 2. Methods that were abstract in base classes and are now implemented in this class
    abstract_methods_to_check = set()

    # 1. Check methods that are still abstract in this class
    if hasattr(cls, "__abstractmethods__"):
        abstract_methods_to_check.update(cls.__abstractmethods__)

    # 2. Check methods from base classes that were abstract and are now defined in cls
    # We look for methods that exist in cls but were abstract in a base class
    # This catches implementations that override abstract methods
    for base in cls.__mro__[1:]:  # Skip cls itself, check base classes
        if hasattr(base, "__abstractmethods__"):
            for method_name in base.__abstractmethods__:
                # Check if cls explicitly defines this method (not just inherits it)
                # Using cls.__dict__ to see if it's overridden in cls
                if method_name in cls.__dict__:
                    abstract_methods_to_check.add(method_name)

    if not abstract_methods_to_check:
        return  # No abstract methods to check

    for method_name in abstract_methods_to_check:
        # Get the method from the class (checking the actual implementation/declaration)
        if not hasattr(cls, method_name):
            # This shouldn't happen, but skip if method not found
            continue

        method = getattr(cls, method_name)
        if not callable(method):
            continue

        # Get the signature
        try:
            sig = inspect.signature(method)
        except (ValueError, TypeError):
            # Some built-in methods might not have inspectable signatures
            continue

        # Check if **kwargs is present in the signature
        has_kwargs = False
        for param_name, param in sig.parameters.items():
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                has_kwargs = True
                break

        if not has_kwargs:
            raise TypeError(
                f"Abstract method '{cls.__name__}.{method_name}' must include '**kwargs' "
                f"in its signature. Current signature: {sig}"
            )

        if method_name == "perform_work":
            if "subtask" not in sig.parameters:
                raise TypeError(
                    f"Abstract method '{cls.__name__}.{method_name}' must include 'subtask' "
                    f"in its signature. Current signature: {sig}"
                )
            if "spec" not in sig.parameters:
                raise TypeError(
                    f"Abstract method '{cls.__name__}.{method_name}' must include 'spec' "
                    f"in its signature. Current signature: {sig}"
                )


class Actor(ABC):
    _class_config: Optional[ActorConfig] = None
    _class_prompt: Optional[Template] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        # Validate that all abstract methods have **kwargs in their signature
        _validate_abstract_methods_have_kwargs(cls)

        nested = getattr(cls, "config", None) or getattr(cls, "Config", None)
        if nested is not None:
            cls._class_config = _parse_class_config(nested)

        nested_prompt = getattr(cls, "prompt", None) or getattr(cls, "Prompt", None)
        if nested_prompt is not None:
            if isinstance(nested_prompt, Template):
                cls._class_prompt = nested_prompt
            else:
                raise TypeError("Prompt must be a jinja2 Template")

    def __init__(
        self,
        config: Optional[ActorConfig] = None,
        llm: Union[Client, Type[Client]] = LiteLLMClient,
        prompt: Optional[Template] = None,
        profile: Optional[Profile] = None,
    ):
        resolved = config or getattr(self.__class__, "_class_config", None)
        if resolved is None:
            raise ValueError(
                "ActorConfig must be provided either via constructor or by defining a nested `config`/`Config`."
            )

        self.config = resolved.model_copy(deep=True)
        self.profile = profile
        self.id = uuid.uuid4()

        # Resolve prompt: constructor > class > default based on type
        resolved_prompt = prompt or getattr(self.__class__, "_class_prompt", None)
        if resolved_prompt is None:
            # Set defaults based on class type
            # Check if this is a Supervisor by looking for Supervisor-specific methods
            if hasattr(self.__class__, "handle_first_assignment") or hasattr(self.__class__, "review"):
                resolved_prompt = SUPERVISOR_SYSTEM_PROMPT
            elif hasattr(self.__class__, "perform_work"):
                resolved_prompt = WORKER_SYSTEM_PROMPT
            else:
                # For base Actor, default to worker prompt
                resolved_prompt = WORKER_SYSTEM_PROMPT

        self.prompt = resolved_prompt

        if isinstance(llm, type):
            self.llm = llm(model=self.config.model)
        else:
            self.llm = llm

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id}) \n {self.profile.model_dump_json(indent=2)}"

    def render_system_prompt(self, **kwargs: Any) -> str:
        return self.prompt.render(**kwargs)


# --- Worker registry and registration helper ---
class WorkerRegistration(BaseModel):
    worker_class: type["Worker"]
    profile: Profile

    class Config:
        arbitrary_types_allowed = True


WORKER_REGISTRY: dict[str, WorkerRegistration] = {}


def register_worker(*profiles: Profile):
    """Decorator to register a Worker implementation with profile information.

    Usage examples:
    - Single registration: @register_worker(Profile(actor_name="TypeA Worker", specialty="..."))
    - Multiple registrations: @register_worker(
          Profile(actor_name="TypeA Worker", specialty="..."),
          Profile(actor_name="TypeB Worker", specialty="...")
      )

    Each profile should have a unique actor_name that will be used as the registration key.
    """

    def _wrap(cls: type["Worker"]):
        for profile in profiles:
            registration = WorkerRegistration(worker_class=cls, profile=profile)
            WORKER_REGISTRY[profile.actor_name] = registration
        return cls

    return _wrap


class Supervisor(Actor):
    def __init__(
        self,
        config: Optional[ActorConfig] = None,
        llm: Union[Client, Type[Client]] = LiteLLMClient,
        prompt: Optional[Template] = None,
        profile: Optional[Profile] = None,
    ):
        super().__init__(config, llm, prompt, profile)

        # Initialize cost tracking
        self.spec_cost = 0.0
        self.total_worker_cost = 0.0

        # Build and keep workers for allowed types that are registered
        self._workers: dict[str, Worker] = {}
        for actor_name, registration in WORKER_REGISTRY.items():
            worker_cls = registration.worker_class
            worker_profile = registration.profile
            self._workers[actor_name] = worker_cls(self.config, self.llm, profile=worker_profile)

    def get_workers_string(self) -> str:
        return "\n".join([repr(worker) for worker in self._workers.values()])

    def list_registered_worker_types(self) -> List[str]:
        return sorted(list[str](self._workers.keys()))

    def get_worker(self, task_type: str) -> "Worker":
        worker = self._workers.get(task_type)
        if worker is None:
            known = ", ".join(sorted(self._workers.keys()))
            logger.error(f"Unknown task_type '{task_type}'. Known: {known}")
            raise KeyError(f"Unknown task_type '{task_type}'. Known: {known}")
        return worker

    def get_worker_job_function(self, task_type: str) -> Callable:
        worker = self.get_worker(task_type)
        return worker.perform_work

    def merge_task_to_spec(self, subtask: SubTask, merge_context: bool = True) -> bool:
        try:
            if not self.spec:
                raise ValueError("Spec not initialized")
            self.spec = self.spec.merge_review(subtask, merge_context=merge_context)
            return True
        except Exception as e:
            logger.error(f"Error merging task to spec: {e}")
            return False

    @abstractmethod
    async def handle_first_assignment(self, **kwargs: Any) -> List[SubTask]:
        pass

    @abstractmethod
    async def review(self, **kwargs: Any) -> SubTaskReview:
        pass


class Worker(Actor):
    def __init__(
        self,
        config: Optional[ActorConfig] = None,
        llm: Union[Client, Type[Client]] = LiteLLMClient,
        prompt: Optional[Template] = None,
        profile: Optional[Profile] = None,
    ):
        super().__init__(config, llm, prompt, profile)

    @abstractmethod
    async def perform_work(self, subtask: SubTask, spec: Spec, **kwargs: Any) -> SubTask:
        pass
