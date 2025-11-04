import asyncio
from asyncio import TaskGroup
from typing import Iterable

from spec_agent.actors import Supervisor
from spec_agent.spec import Spec, SubTask
from spec_agent.ui import SchedulerUI, console


async def run_scheduler(
    initial: Iterable[SubTask], supervisor: Supervisor, max_workers: int = 8, max_rounds: int = 10
) -> None:
    review_queue: asyncio.Queue[SubTask] = asyncio.Queue()
    pending_reviews: dict[str, SubTask] = {}  # Track tasks waiting for review
    in_flight = 0
    rounds = 0
    worker_sem = asyncio.Semaphore(max_workers)

    # Track total worker costs
    if not hasattr(supervisor, "total_worker_cost"):
        supervisor.total_worker_cost = 0.0

    def log_pending_reviews():
        """Log current tasks waiting for review."""
        console.print(SchedulerUI.pending_reviews_card(pending_reviews))

    async def spawn(tg: TaskGroup, task: SubTask, spec: Spec):
        nonlocal in_flight, rounds
        await worker_sem.acquire()
        in_flight += 1
        rounds += 1
        console.print(SchedulerUI.worker_card(task.id, task.subitem.type))

        async def runner():
            nonlocal in_flight
            try:
                result_task = await supervisor.get_worker_job_function(task.subitem.type)(subtask=task, spec=spec)

                # Accumulate worker cost
                supervisor.total_worker_cost += result_task.task_cost

                # Add to review queue and track it
                await review_queue.put(result_task)
                pending_reviews[result_task.id] = result_task

                # Log when worker finishes and task is waiting for review
                console.print(SchedulerUI.worker_complete_card(result_task.id, result_task.subitem.type))
                log_pending_reviews()
            finally:
                in_flight -= 1
                worker_sem.release()

        tg.create_task(runner())

    async with TaskGroup() as tg:
        # seed first round
        for task in initial:
            await spawn(tg, task, supervisor.spec)

        # Single consumer: strictly sequential reviews
        while True:
            # If nothing in flight, also drain queue and exit when empty
            if (in_flight == 0 and review_queue.empty()) or rounds >= max_rounds:
                break

            result_task = await review_queue.get()  # blocks; ensures order of handling

            # Remove from pending reviews tracking
            pending_reviews.pop(result_task.id, None)

            console.print(SchedulerUI.review_card(result_task.id))
            # supervisor.review returns List[SubTask] and tracks its own costs internally
            followups = await supervisor.review(result_task)  # strictly sequential

            followup_count = len(followups) if followups else None
            console.print(SchedulerUI.review_complete_card(result_task.id, followup_count))

            for task in followups:
                if rounds < max_rounds:
                    await spawn(tg, task, supervisor.spec)
