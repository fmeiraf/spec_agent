# Spec Agent

**Spec Agent** is an experimetal inference strategy that aims to use an async environment to simulate a _supervisor > worker_ situation with real time task review and follow up assigment.

In summary this strategy works in 2 phases:

- **Initial assignment**: a supervisor agent starts a multi-step process assigning one or multiple tasks to a specialized pool of workers (that work in parallel)
- **Review and follow-up**: as the workers finish their tasks, the results are pushed to a review queue where the supervisor will review them _sequentially_ - this is to avoid merging issues. Reviews could be approved, partially approved, and rejected. Reviews can potentially lead to new tasks assigned to workers - these will be executed in parallel by the pool of workers when a worker is available.

The phases above are pretty much "glued" together using a `Spec` object that carry a lot of relevant information used in all the prompts and message exchanges between supervisor and workers.

The main motivators for me to try this approach arises from production challegens like:

- Increase efficiency on inference for multi step tasks
- Create context windows that allow the agent to be focused on a very specific part of the process
- Enable flexibilty to craft specific worflows for isolated parts of a multi step task
- Leverage async environments to create more dynamic exchange between agents

The things that excite me on this:

- The possibility of creating multiple different workflows customizing the actors actions
- Capability of keeping context focused and isolated for improtant parts of the task

Things I see as limitations:

- The review phase will very rapidly become a bottleneck - as it works sequentially. The merging phase is something complex for multi agent systems and depending on the task could be a big limitation
- Dedicated agents can be very shalow on their actions without correct context and prompting - so it does require iteration for good results

## Introduction

Spec Agent addresses the challenge of completing complex, multi-dimensional specifications using LLMs. Instead of asking a single model to handle everything at once, Spec Agent:

- **Decomposes** complex specs into focused sub-items, each with clear acceptance criteria
- **Coordinates** specialized workers (each with their own specialty) to tackle specific aspects
- **Reviews** work iteratively, ensuring quality while allowing for refinement
- **Merges** results intelligently, combining worker outputs into a cohesive final result
- **Tracks** costs transparently, providing visibility into both supervisor and worker expenses

This approach leads to higher quality outputs, better specialization, and more reliable completion of complex tasks. The framework is designed to be flexible, allowing you to define custom workers, supervisors, and specifications tailored to your specific domain.

## Architecture

Spec Agent follows a **supervisor-worker architecture**:

```
┌─────────────────────────────────────────┐
│           Supervisor                    │
│  - Creates initial task assignments     │
│  - Reviews worker results               │
│  - Merges approved changes              │
│  - Generates follow-up tasks            │
└──────────────┬──────────────────────────┘
               │
               │ Assigns tasks
               │
    ┌──────────┴──────────┐
    │                     │
┌───▼────┐         ┌──────▼─────┐
│Worker 1│         │  Worker 2  │
│Specialty│         │  Specialty  │
│   A     │         │     B       │
└───┬────┘         └──────┬─────┘
    │                     │
    └──────────┬──────────┘
               │
               │ Returns results
               │
    ┌──────────▼──────────┐
    │   Review Queue      │
    │  (Sequential)       │
    └─────────────────────┘
```

### Key Components

#### Spec

A **Spec** defines what needs to be accomplished. It contains:

- **Title & Description**: High-level overview of the goal
- **Acceptance Criteria**: Overall requirements for completion
- **SubItems**: Individual aspects that must be addressed, each with:
  - Type identifier
  - Description
  - Acceptance criteria
  - Expected output format

#### Supervisor

The **Supervisor** orchestrates the entire process:

- Analyzes the spec and creates initial task assignments
- Reviews worker results against acceptance criteria
- Merges approved changes into the final result
- Generates follow-up tasks when needed
- Tracks costs and manages the workflow

#### Workers

**Workers** are specialized agents registered with specific profiles:

- Each worker has a **specialty** (what they're good at)
- Workers receive tasks matching their specialty
- They perform work and return results in a specified format
- Multiple workers can operate in parallel

#### Scheduler

The **Scheduler** manages the execution flow:

- Assigns tasks to workers in parallel (up to `max_workers`)
- Collects completed tasks into a review queue
- Ensures reviews happen sequentially (maintaining order)
- Continues until all criteria are met or `max_rounds` is reached

## Installation

```bash
# Install the package
pip install spec-agent

# Or with uv
uv pip install spec-agent
```

## Quick Start

Here's a minimal example based on the data visualization use case:

```python
import asyncio
from spec_agent.actors import ActorConfig, Profile, Supervisor, Worker, register_worker
from spec_agent.main import SpecAgent
from spec_agent.spec import Spec, SubItem, TaskList, SubTaskReview
from pydantic import BaseModel, Field

# 1. Define output formats
class SpecOutputFormat(BaseModel):
    plot_function: str = Field(description="The plot function that takes a pandas dataframe and returns a svg string.")

class TaskOutputFormat(BaseModel):
    plot_function_diff: str = Field(description="The diff of the plot function.")

# 2. Create a specialized worker
@register_worker(
    Profile(actor_name="axis_and_grid", specialty="A worker that improves the axes and grid of a plot function.")
)
class DataVizWorker(Worker):
    config = ActorConfig(model="xai/grok-4-fast-reasoning")

    async def perform_work(self, subtask: SubTask, spec: Spec, **kwargs):
        # Perform work based on subtask
        # Return updated SubTask with results
        ...

# 3. Create a supervisor
class DataVizSupervisor(Supervisor):
    config = ActorConfig(model="xai/grok-4-fast-reasoning")

    async def handle_first_assignment(self, **kwargs):
        # Generate initial tasks from spec
        ...

    async def review(self, subtask: SubTask, **kwargs):
        # Review worker results and return follow-up tasks if needed
        ...

# 4. Define your spec
VizSpec = Spec(
    title="Improve the visualization",
    description="Given a current plot function, improve the plot so it meets styling and layout criteria",
    acceptance_criteria="A plot function that takes a pandas dataframe and returns a svg string...",
    subitems_type={"axis_and_grid", "font_and_text_clarity", ...},
    all_subitems={
        "axis_and_grid": SubItem(
            type="axis_and_grid",
            description="Improve the axes and grid of the plot",
            acceptance_criteria="Creates clean, unobtrusive axis and grid styling...",
            task_output_format=json.dumps(TaskOutputFormat.model_json_schema()),
        ),
        ...
    },
    spec_output_format=SpecOutputFormat,
    task_output_format=TaskOutputFormat,
)

# 5. Run the agent
async def main():
    agent = SpecAgent(supervisor=DataVizSupervisor())
    completed_spec = await agent.complete_spec(
        spec=VizSpec,
        spec_output_format=SpecOutputFormat,
        task_output_format=TaskOutputFormat,
        goal_output=SpecOutputFormat(plot_function=CURRENT_PLOT_FUNCTION),
        max_rounds=25
    )

    print(f"Total Cost: ${completed_spec.total_cost:.4f}")
    print(f"Final Result: {completed_spec.spec.final_result}")

asyncio.run(main())
```

## Concepts in Detail

### Spec Structure

A `Spec` is the blueprint for your task:

```python
Spec(
    title="Your task title",
    description="What needs to be accomplished",
    acceptance_criteria="How you'll know it's done",
    subitems_type={"type1", "type2", ...},  # Set of allowed subitem types
    all_subitems={
        "type1": SubItem(
            type="type1",
            description="What this subitem addresses",
            acceptance_criteria="Specific criteria for this subitem",
            task_output_format=json.dumps(YourTaskOutputFormat.model_json_schema()),
        ),
        ...
    },
    spec_output_format=YourSpecOutputFormat,
    task_output_format=YourTaskOutputFormat,
)
```

### Worker Registration

Workers are registered using the `@register_worker` decorator:

```python
@register_worker(
    Profile(actor_name="worker_type_1", specialty="What this worker specializes in"),
    Profile(actor_name="worker_type_2", specialty="Another specialty"),
)
class MyWorker(Worker):
    config = ActorConfig(model="xai/grok-4-fast-reasoning")

    async def perform_work(self, subtask: SubTask, spec: Spec, **kwargs) -> SubTask:
        # Your work logic here
        # Return updated SubTask with task_result, task_cost, etc.
        ...
```

You can register multiple profiles for a single worker class if it handles multiple specialties.

### Supervisor Implementation

Supervisors must implement two methods:

```python
class MySupervisor(Supervisor):
    config = ActorConfig(model="xai/grok-4-fast-reasoning")

    async def handle_first_assignment(self, **kwargs) -> List[SubTask]:
        """Generate initial tasks from the spec."""
        # Analyze spec and create tasks
        # Return list of SubTask objects
        ...

    async def review(self, subtask: SubTask, **kwargs) -> List[SubTask]:
        """Review worker results and return follow-up tasks if needed."""
        # Review the subtask
        # If approved/partially_approved, merge results
        # If rejected/partially_approved, return new tasks
        # Return empty list if no follow-ups needed
        ...
```

### Task Lifecycle

1. **Initial Assignment**: Supervisor analyzes spec and creates initial `SubTask` objects
2. **Worker Execution**: Workers receive tasks matching their specialty and perform work in parallel
3. **Review**: Completed tasks are queued for sequential review by the supervisor
4. **Merge**: Approved results are merged into the spec's `final_result`
5. **Iteration**: Follow-up tasks are created if acceptance criteria aren't fully met
6. **Completion**: Process continues until all criteria are met or `max_rounds` is reached

## Example: Data Visualization Improvement

See `examples/create_data_visualization.py` for a complete example that:

- Defines 7 specialized workers for different visualization aspects:

  - Axis and grid styling
  - Font and text clarity
  - Figure size and layout
  - Color palette and contrast
  - Lines and markers
  - Legend and annotations
  - Faceting and small multiples

- Uses a supervisor that:

  - Creates initial tasks based on the current plot function
  - Reviews improvements against detailed acceptance criteria
  - Merges approved changes using LLM-based code merging
  - Generates follow-up tasks when needed

- Tracks costs separately for supervisor and workers

## Configuration

### Actor Configuration

Workers and supervisors use `ActorConfig`:

```python
ActorConfig(
    model="xai/grok-4-fast-reasoning",  # LLM model identifier
    max_retries=1,                       # Retry attempts
    llm_kwargs={                         # Additional LLM parameters
        "temperature": 0.7,
        "max_tokens": 2000,
    }
)
```

### Agent Configuration

```python
agent = SpecAgent(
    supervisor=MySupervisor(),
    worker_pool_size=3,      # Max parallel workers
    timeout=None,            # Optional timeout in seconds
)

result = await agent.complete_spec(
    spec=MySpec,
    spec_output_format=SpecOutputFormat,
    task_output_format=TaskOutputFormat,
    goal_output=InitialOutput,
    max_rounds=10,          # Max iterations
    # Additional kwargs passed to supervisor and workers
    data_frame=my_data,
)
```

## Cost Tracking

Spec Agent tracks costs automatically:

```python
result = await agent.complete_spec(...)

print(f"Total Cost: ${result.total_cost:.4f}")
print(f"Supervisor Cost: ${result.supervisor_cost:.4f}")
print(f"Worker Cost: ${result.worker_cost:.4f}")
```

Costs are accumulated:

- **Supervisor cost**: From task assignment, review, and merge operations
- **Worker cost**: From all worker `perform_work` calls
- **Total cost**: Sum of supervisor and worker costs

## Advanced Usage

### Custom Prompts

Override default prompts by providing Jinja2 templates:

```python
from jinja2 import Template

class MyWorker(Worker):
    prompt = Template("""
    Your custom prompt here.
    Use {{ variable }} for template variables.
    """)
```

### Context Management

Tasks maintain context through `task_context`:

```python
subtask = SubTask(
    ...,
    task_context=[
        {"role": "user", "content": "Previous context"},
        {"role": "assistant", "content": "Previous response"},
    ]
)
```

Workers can use this context to maintain conversation history.

### Result Merging

The supervisor handles merging approved results:

```python
# In supervisor.review()
if review_result.content.review_result in ["approved", "partially_approved"]:
    # Merge the result
    self.spec = self.spec.merge_review(
        subtask,
        final_result=merged_result,
    )
```

The `merge_review` method updates the spec's `final_result` and increments the version.

## Best Practices

1. **Clear Acceptance Criteria**: Define specific, measurable acceptance criteria for each subitem
2. **Specialized Workers**: Create focused workers with clear specialties
3. **Iterative Refinement**: Expect multiple rounds of review and refinement (it depends on how many initial workers you assign tasks to, but with 5 workers, 25 rounds will probably be close to have assigments reviewed, including follow ups)
4. **Cost Monitoring**: Monitor costs, especially with long-running specs
5. **Error Handling**: Implement robust error handling in worker and supervisor methods (because we are using TaskGroup for async management, errors on the actors execution will stop the whole group of execution - if you don't catch the exceptions)
6. **Output Formats**: Use Pydantic models for structured output validation
