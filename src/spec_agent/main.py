from logging import getLogger
from typing import Any, List, Optional

from pydantic import BaseModel

from spec_agent.actors import SubTask, Supervisor
from spec_agent.scheduler import run_scheduler
from spec_agent.spec import Spec

logger = getLogger(__name__)


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
        self, spec: Spec, spec_output_format: BaseModel, task_output_format: BaseModel, **kwargs: Any
    ) -> None:
        self.supervisor.spec = spec
        self.supervisor.spec_output_format = spec_output_format
        self.supervisor.task_output_format = task_output_format

        for key, value in kwargs.items():
            setattr(self.supervisor, key, value)

        for worker in self.supervisor._workers.values():
            worker.task_output_format = task_output_format
            worker.spec_output_format = spec_output_format
            worker.spec = spec
            worker.config.llm_kwargs = kwargs.get("llm_kwargs", {})

            for key, value in kwargs.items():
                setattr(worker, key, value)

    async def complete_spec(
        self,
        spec: Spec,
        spec_output_format: BaseModel,
        task_output_format: BaseModel,
        **kwargs: Any,
    ) -> Spec:
        self._initialize(spec, spec_output_format, task_output_format, **kwargs)
        initial_tasks: List[SubTask] = await self.supervisor.handle_first_assignment(
            spec=spec, spec_output_format=spec_output_format, task_output_format=task_output_format, **kwargs
        )

        final_result = await run_scheduler(initial=initial_tasks, supervisor=self.supervisor)
        return final_result
