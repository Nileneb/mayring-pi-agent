from __future__ import annotations

import asyncio
from typing import Callable

from a2a.helpers import new_task, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    add_a2a_routes_to_fastapi,
    create_agent_card_routes,
    create_jsonrpc_routes,
)
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    TaskState,
)
from a2a.utils import DEFAULT_RPC_URL, TransportProtocol

from mayring_pi_agent.pi import run_task_with_memory

_SKILLS = [
    AgentSkill(
        id="task",
        name="Free-form task",
        description="Run any free-form task grounded in live cloud memory (Mayring).",
        tags=["mayring", "memory", "ollama"],
    ),
    AgentSkill(
        id="categorize",
        name="Mayring categorization",
        description="Categorize text into the Mayring codebook using goal-anchored mixed-method analysis.",
        tags=["mayring", "categorization"],
    ),
    AgentSkill(
        id="judge",
        name="Relevance judgement",
        description="Judge relevance / quality of a candidate against memory context.",
        tags=["mayring", "judge", "reranker"],
    ),
]


def build_agent_card(base_url: str, model: str, version: str) -> AgentCard:
    url = base_url.rstrip("/") + "/"
    return AgentCard(
        name="MayringCoder Pi-Agent",
        description=(
            f"Memory-grounded Ollama agent ({model}) with live cloud-memory search. "
            "Wraps run_task_with_memory over the A2A protocol."
        ),
        version=version,
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        supported_interfaces=[
            AgentInterface(url=url, protocol_binding=TransportProtocol.JSONRPC)
        ],
        skills=_SKILLS,
    )


class PiAgentExecutor(AgentExecutor):
    """A2A AgentExecutor that wraps run_task_with_memory.

    The A2A context_id is threaded into run_task_with_memory as session_id so the
    cloud-memory recency-lane keeps the live session thread in the agent's context.
    """

    def __init__(
        self,
        model: str,
        ollama_url: str,
        repo_slug: str | None = None,
        num_predict: int | None = None,
        runner: Callable[..., str] = run_task_with_memory,
    ):
        self._model = model
        self._ollama_url = ollama_url
        self._repo_slug = repo_slug
        self._num_predict = num_predict
        self._runner = runner

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        text = context.get_user_input()
        if context.current_task is None:
            await event_queue.enqueue_event(
                new_task(context.task_id, context.context_id, TaskState.TASK_STATE_SUBMITTED)
            )
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.start_work()

        result = await asyncio.to_thread(
            self._runner,
            task=text,
            ollama_url=self._ollama_url,
            model=self._model,
            repo_slug=self._repo_slug,
            session_id=context.context_id,
            num_predict=self._num_predict,
        )

        await updater.complete(updater.new_agent_message([new_text_part(result)]))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.cancel()


def register_a2a(
    app,
    *,
    base_url: str,
    model: str,
    ollama_url: str,
    version: str = "0.1.4",
    repo_slug: str | None = None,
    num_predict: int | None = None,
    runner: Callable[..., str] = run_task_with_memory,
) -> AgentCard:
    """Mount A2A agent-card + JSON-RPC routes onto an existing FastAPI app."""
    card = build_agent_card(base_url=base_url, model=model, version=version)
    executor = PiAgentExecutor(
        model=model,
        ollama_url=ollama_url,
        repo_slug=repo_slug,
        num_predict=num_predict,
        runner=runner,
    )
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=create_agent_card_routes(card),
        jsonrpc_routes=create_jsonrpc_routes(handler, DEFAULT_RPC_URL),
    )
    return card
