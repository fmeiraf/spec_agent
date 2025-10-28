from copy import deepcopy
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class SubTask(BaseModel):
    id: str
    description: str
    context: Dict[str, Any]  # this will probably hold the message history for the subtask
    result: Any | None = None


class SubTaskReview(BaseModel):
    subtask: SubTask
    is_approved: bool
    review_reason: str
    followup_subtasks: List[SubTask] = Field(default_factory=list)


class Spec(BaseModel):
    id: str
    description: str
    version: int = 1
    schema_version: int = 1
    all_subtasks: Dict[str, SubTask] = Field(default_factory=dict)
    final_result: Any | None = None

    def clone(self) -> "Spec":
        return Spec(
            id=self.id,
            version=self.version,
            schema_version=self.schema_version,
            description=deepcopy(self.description),
            all_subtasks=deepcopy(self.all_subtasks),
        )

    def merge_subtask(self, subtask: SubTask, *, merge_context: bool = True) -> "Spec":
        """
        Return a new Spec with the given subtask merged in.

        - Adds the subtask if not present.
        - If present, updates description/result and merges context when requested.
        - Increments version only when an actual change occurs.
        """
        existing = self.all_subtasks.get(subtask.id)

        # Fast path: not present yet
        if existing is None:
            new_spec = self.clone()
            new_spec.all_subtasks[subtask.id] = deepcopy(subtask)
            new_spec.version += 1
            return new_spec

        # Build merged subtask
        merged = SubTask(
            id=existing.id,
            description=subtask.description if subtask.description != existing.description else existing.description,
            result=subtask.result if subtask.result != existing.result else existing.result,
            context=({**existing.context, **subtask.context} if merge_context else deepcopy(subtask.context)),
        )

        # No change detected
        if merged == existing:
            return self

        # Apply change on a cloned spec and bump version
        new_spec = self.clone()
        new_spec.all_subtasks[subtask.id] = merged
        new_spec.version += 1
        return new_spec
