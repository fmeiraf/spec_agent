from logging import getLogger
from typing import Any, List, Optional

from pydantic import BaseModel

from spec_agent.actors import SubTask, Supervisor
from spec_agent.scheduler import run_scheduler
from spec_agent.spec import Spec

logger = getLogger(__name__)


class SpecResult(BaseModel):
    """Result of completing a spec, including the spec and cost breakdown."""

    spec: Spec
    total_cost: float
    supervisor_cost: float
    worker_cost: float

    class Config:
        arbitrary_types_allowed = True


class SpecAgent:
    def __init__(
        self,
        supervisor: Supervisor,
        worker_pool_size: int = 3,
        timeout: Optional[int] = None,
    ):
        self.supervisor = supervisor
        self.worker_pool_size = worker_pool_size
        self.timeout = timeout

    def _initialize(
        self, spec: Spec, spec_output_format: BaseModel, task_output_format: BaseModel, goal_output: Any, **kwargs: Any
    ) -> None:
        # Set the output format on the spec itself for validation
        spec.spec_output_format = spec_output_format
        spec.final_result = goal_output
        self.supervisor.spec = spec
        self.supervisor.spec_output_format = spec_output_format
        self.supervisor.task_output_format = task_output_format
        self.supervisor.spec_cost = 0.0
        self.supervisor.total_worker_cost = 0.0

        for key, value in kwargs.items():
            setattr(self.supervisor, key, value)

        for worker in self.supervisor._workers.values():
            worker.config.llm_kwargs = kwargs.get("llm_kwargs", {})
            for key, value in kwargs.items():
                setattr(worker, key, value)

    async def complete_spec(
        self,
        spec: Spec,
        spec_output_format: BaseModel,
        task_output_format: BaseModel,
        goal_output: Any,
        max_rounds: int = 10,
        **kwargs: Any,
    ) -> SpecResult:
        self._initialize(spec, spec_output_format, task_output_format, goal_output, **kwargs)
        initial_tasks: List[SubTask] = await self.supervisor.handle_first_assignment(
            spec=spec, spec_output_format=spec_output_format, task_output_format=task_output_format, **kwargs
        )

        await run_scheduler(initial=initial_tasks, supervisor=self.supervisor, max_rounds=max_rounds)

        # Calculate total costs
        supervisor_cost = getattr(self.supervisor, "spec_cost", 0.0)
        worker_cost = getattr(self.supervisor, "total_worker_cost", 0.0)
        total_cost = supervisor_cost + worker_cost

        return SpecResult(
            spec=self.supervisor.spec,
            total_cost=total_cost,
            supervisor_cost=supervisor_cost,
            worker_cost=worker_cost,
        )
