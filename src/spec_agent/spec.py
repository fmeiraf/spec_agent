import uuid
from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional, Set, Type

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
    supervisor_diagnosis: str
    assigned_actor_name: str
    assigned_subitem_type: str


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
    final_result: Any | None = Field(default=None)
    spec_output_format: Optional[Type[BaseModel]] = Field(default=None, exclude=True)
    task_output_format: Optional[Type[BaseModel]] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

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
            spec_output_format=self.spec_output_format,
        )

    def merge_subitem(self, subtask: SubTask, *, merge_context: bool = True) -> "Spec":
        """
        Return a new Spec with the given subitem merged in.

        - Adds the subitem if not present.
        - If present, updates type/description/result and merges context when requested.
        - Increments version only when an actual change occurs.
        """
        existing = self.all_subitems.get(subtask.subitem.type)

        # Fast path: not present yet
        if existing is None:
            raise ValueError(f"Subitem with id {subtask.subitem.type} not found")

        # Build merged subitem
        merged = SubItem(
            type=subtask.subitem.type if subtask.subitem.type != existing.type else existing.type,
            description=(
                subtask.subitem.description
                if subtask.subitem.description != existing.description
                else existing.description
            ),
            context=(
                {**existing.context, **subtask.subitem.context} if merge_context else deepcopy(subtask.subitem.context)
            ),
            task_output_format=subtask.subitem.task_output_format
            if subtask.subitem.task_output_format != existing.task_output_format
            else existing.task_output_format,
            acceptance_criteria=subtask.subitem.acceptance_criteria
            if subtask.subitem.acceptance_criteria != existing.acceptance_criteria
            else existing.acceptance_criteria,
        )

        # No change detected
        if merged == existing:
            return self

        # Apply change on a cloned spec and bump version
        self.all_subitems[subtask.subitem.type] = merged
        self.version += 1
        return self

    def set_final_result(self, result: Any) -> "Spec":
        """
        Set and validate the final result against the spec output format.
        Returns a new Spec with the validated result.

        - If spec_output_format is set, validates the result against it
        - If result is a dict, parses it using the output format
        - If result is already an instance of the output format, uses it directly
        - If no output format is set, stores the result as-is
        """
        if self.spec_output_format is not None:
            # Validate and parse the result
            if isinstance(result, dict):
                validated_result = self.spec_output_format(**result)
            elif isinstance(result, self.spec_output_format):
                validated_result = result
            else:
                raise ValueError(
                    f"Result must be a dict or instance of {self.spec_output_format.__name__}, "
                    f"got {type(result).__name__}"
                )

            # Store as dict for serialization
            final_value = validated_result.model_dump() if hasattr(validated_result, "model_dump") else validated_result
        else:
            # No validation if format not specified
            final_value = result

        self.final_result = final_value
        return self

    def get_final_result_as_model(self) -> BaseModel | None:
        """
        Get the final result as a validated Pydantic model instance.

        Returns:
            - The final_result as an instance of spec_output_format
            - None if final_result is None

        Raises:
            ValueError if spec_output_format is not set
        """
        if self.final_result is None:
            return None
        if self.spec_output_format is None:
            raise ValueError("Cannot convert to model: spec_output_format not set")
        if isinstance(self.final_result, self.spec_output_format):
            return self.final_result
        return self.spec_output_format(**self.final_result)

    def merge_review(
        self, subtask: SubTask, final_result: Optional[Any] = None, *, merge_context: bool = True
    ) -> "Spec":
        """
        Merge a SubTaskReview into the Spec by updating the linked SubItem state.

        - Identifies the target subitem by the linked subtask id
        - Updates type/description when they change
        - Merges context from existing subitem, the subitem inside the task, and the task context
        - Records review metadata (is_approved, review_message) in context
        - Updates result only when the review is approved
        - Validates the final result against spec_output_format if set
        - Increments version only on actual changes
        """
        if final_result is None and subtask.task_result is None:
            raise ValueError("No final result or task result provided")

        existing = self.all_subitems.get(subtask.subitem.type)

        # Fast path: not present yet
        if existing is None:
            raise ValueError(f"Subitem with id {subtask.subitem.type} not found")

        merged = SubItem(
            type=subtask.subitem.type if subtask.subitem.type != existing.type else existing.type,
            description=(
                subtask.subitem.description
                if subtask.subitem.description != existing.description
                else existing.description
            ),
            context=subtask.subitem.context if merge_context else deepcopy(subtask.subitem.context),
            task_output_format=subtask.subitem.task_output_format
            if subtask.subitem.task_output_format != existing.task_output_format
            else existing.task_output_format,
            acceptance_criteria=subtask.subitem.acceptance_criteria
            if subtask.subitem.acceptance_criteria != existing.acceptance_criteria
            else existing.acceptance_criteria,
        )

        self.all_subitems[subtask.subitem.type] = merged

        if final_result is None:
            final_result = subtask.task_result

        self.set_final_result(final_result)
        self.version += 1
        return self
