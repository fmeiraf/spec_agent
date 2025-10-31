import asyncio
import random
from asyncio import TaskGroup
from typing import Iterable, List

from spec_agent.actors import Supervisor
from spec_agent.spec import Spec, SubTask


async def run_scheduler(
    initial: Iterable[SubTask], supervisor: Supervisor, max_workers: int = 8, max_rounds: int = 10
) -> None:
    review_queue: asyncio.Queue[SubTask] = asyncio.Queue()
    in_flight = 0
    rounds = 0
    worker_sem = asyncio.Semaphore(max_workers)

    async def spawn(tg: TaskGroup, task: SubTask, spec: Spec):
        nonlocal in_flight, rounds
        await worker_sem.acquire()
        in_flight += 1
        rounds += 1

        async def runner():
            nonlocal in_flight
            try:
                result_task = await supervisor.get_worker_job_function(task.subitem.type)(subtask=task, spec=spec)
                await review_queue.put(result_task)
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
            followups = await supervisor.review(result_task)  # strictly sequential

            for task in followups:
                if rounds < max_rounds:
                    await spawn(tg, task, supervisor.spec)


# --- demo ---
if __name__ == "__main__":

    async def do_async_work(task: SubTask, *, max_delay: float = 1.5) -> SubTask:
        # Simulate variable latency + outcome
        await asyncio.sleep(random.uniform(0.1, max_delay))
        success = random.random() > 0.2  # 80% pass
        round_num = int(task.context.get("round", 0))
        # Return task with updated result payload
        return SubTask(
            id=task.id,
            description=task.description,
            context=task.context,
            result={"ok": success, "round": round_num},
        )

    async def professor_review(task_result: SubTask) -> List[SubTask]:
        """
        Handle results sequentially (this function is called by exactly one consumer).
        It may decide to spawn follow-up tasks (new async jobs).
        """
        round_num = int(task_result.context.get("round", 0))
        sid = task_result.context.get("id")
        ok = bool(task_result.result.get("ok")) if isinstance(task_result.result, dict) else bool(task_result.result)
        # Synchronous/ordered logic: print in order for demonstration
        print(f"[REVIEW] round={round_num} id={sid} -> ok={ok}")

        followups: List[SubTask] = []
        if not ok and round_num < 3:
            # Ask the same student for a revision (new async job)
            new_ctx = dict(task_result.context)
            new_ctx["round"] = round_num + 1
            followups.append(
                SubTask(
                    id=task_result.id,
                    description=task_result.description,
                    context=new_ctx,
                    result=None,
                )
            )
        return followups

    initial = [
        SubTask(id=f"task-{i}", description=f"task-{i}", context={"id": i, "round": 0}, result=None) for i in range(5)
    ]

    asyncio.run(run_scheduler(initial, do_async_work, professor_review))
