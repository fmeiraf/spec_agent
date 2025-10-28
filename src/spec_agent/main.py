from logging import getLogger
from typing import Optional

from spec_agent.actors import Reviewer
from spec_agent.spec import Spec

logger = getLogger(__name__)


class SpecAgent:
    def __init__(self, reviewer: Reviewer, timeout: Optional[int] = None):
        self.reviewer = reviewer
        self.timeout = timeout

    async def complete_spec(
        self,
        spec: Spec,
        **kwargs,
    ) -> Spec:
        pass
