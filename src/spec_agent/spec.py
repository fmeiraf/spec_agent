import uuid
from copy import deepcopy
from typing import Any, Dict, List, Literal, Set

from pydantic import BaseModel, Field, model_validator


class SubItem(BaseModel):
    """
    A subitem is a single isolated item that is part of the spec.
    It can be a subtask, a document, a code snippet, etc.
    """

    type: str
    version: int = Field(default=1)
    description: str
    acceptance_criteria: str
    context: List[Dict[str, Any]] = Field(
        default_factory=list
    )  # this will probably hold the message history for the subtask
    task_output_format: str = Field(
        description="The output format of the task. This is a JSON schema that describes the output of the task."
    )


class SubTaskDetails(BaseModel):
    task_description: str
    actor_name: str


class SubTask(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    subitem: SubItem
    task_details: SubTaskDetails
    status: Literal["pending", "in_progress", "completed", "failed"] = Field(default="pending")
    task_error: str | None = None
    task_result: Any | None = None
    task_context: List[Dict[str, Any]] = Field(
        default_factory=list
    )  # this will probably hold the message history for the subtask


class SubTaskReview(BaseModel):
    subtask: SubTask
    is_approved: bool
    review_message: str


class Spec(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    acceptance_criteria: str
    subitems_type: Set[str]
    all_subitems: Dict[str, SubItem] = Field(default_factory=dict)
    status: Literal["pending", "in_progress", "completed", "failed"] = Field(default="pending")
    version: int = Field(default=1)
    output_format: str = Field(
        description="The output format of the spec. This is a JSON schema that describes the output of the spec."
    )
    final_result: Any | None = Field(default=None)

    @model_validator(mode="after")
    def validate_subitems_type(self) -> "Spec":
        # Ensure all subitems have allowed types
        invalid = [key for key, item in self.all_subitems.items() if item.type not in self.subitems_type]
        if invalid:
            raise ValueError(f"Subitems with invalid type: {invalid}")
        return self

    def clone(self) -> "Spec":
        return Spec(
            id=self.id,
            title=self.title,
            description=self.description,
            acceptance_criteria=self.acceptance_criteria,
            subitems_type=self.subitems_type,
            version=self.version,
            all_subitems=deepcopy(self.all_subitems),
            status=self.status,
            final_result=deepcopy(self.final_result),
        )

    def merge_subitem(self, item_id: str, subitem: SubItem, *, merge_context: bool = True) -> "Spec":
        """
        Return a new Spec with the given subitem merged in.

        - Adds the subitem if not present.
        - If present, updates type/description/result and merges context when requested.
        - Increments version only when an actual change occurs.
        """
        existing = self.all_subitems.get(item_id)

        # Fast path: not present yet
        if existing is None:
            new_spec = self.clone()
            new_spec.all_subitems[item_id] = deepcopy(subitem)
            new_spec.version += 1
            return new_spec

        # Build merged subitem
        merged = SubItem(
            type=subitem.type if subitem.type != existing.type else existing.type,
            description=(subitem.description if subitem.description != existing.description else existing.description),
            result=subitem.result if subitem.result != existing.result else existing.result,
            context=({**existing.context, **subitem.context} if merge_context else deepcopy(subitem.context)),
        )

        # No change detected
        if merged == existing:
            return self

        # Apply change on a cloned spec and bump version
        new_spec = self.clone()
        new_spec.all_subitems[item_id] = merged
        new_spec.version += 1
        return new_spec

    def merge_review(self, review: SubTaskReview, *, merge_context: bool = True) -> "Spec":
        """
        Merge a SubTaskReview into the Spec by updating the linked SubItem state.

        - Identifies the target subitem by the linked subtask id
        - Updates type/description when they change
        - Merges context from existing subitem, the subitem inside the task, and the task context
        - Records review metadata (is_approved, review_message) in context
        - Updates result only when the review is approved
        - Increments version only on actual changes
        """
        subtask = review.subtask
        item_id = subtask.id
        task_item = subtask.subitem

        existing = self.all_subitems.get(item_id)

        # Create path: no existing subitem for this id
        if existing is None:
            # Build new subitem incorporating task subitem + task context + review info
            base_context = deepcopy(task_item.context)
            if merge_context:
                merged_context = {**base_context, **deepcopy(subtask.task_context)}
            else:
                merged_context = deepcopy(subtask.task_context)

            merged_context["is_approved"] = review.is_approved
            merged_context["review_message"] = review.review_message

            new_item = SubItem(
                type=task_item.type,
                description=task_item.description,
                context=merged_context,
                result=deepcopy(subtask.task_result) if review.is_approved else deepcopy(task_item.result),
            )

            new_spec = self.clone()
            new_spec.all_subitems[item_id] = new_item
            new_spec.version += 1
            return new_spec

        # Update path: merge with existing subitem
        if merge_context:
            merged_context = {
                **deepcopy(existing.context),
                **deepcopy(task_item.context),
                **deepcopy(subtask.task_context),
            }
        else:
            merged_context = deepcopy(subtask.task_context)

        merged_context["is_approved"] = review.is_approved
        merged_context["review_message"] = review.review_message

        merged_item = SubItem(
            type=task_item.type if task_item.type != existing.type else existing.type,
            description=(
                task_item.description if task_item.description != existing.description else existing.description
            ),
            context=merged_context,
            result=(
                deepcopy(subtask.task_result)
                if review.is_approved and subtask.task_result != existing.result
                else existing.result
            ),
        )

        if merged_item == existing:
            return self

        new_spec = self.clone()
        new_spec.all_subitems[item_id] = merged_item
        new_spec.version += 1
        return new_spec
