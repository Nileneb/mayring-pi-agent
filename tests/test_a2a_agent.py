import asyncio


from mayring_pi_agent.a2a_agent import PiAgentExecutor, build_agent_card


class _RecordingQueue:
    def __init__(self):
        self.events = []

    async def enqueue_event(self, event):
        self.events.append(event)


class _FakeContext:
    def __init__(self, text):
        self._text = text
        self.task_id = "task-1"
        self.context_id = "ctx-1"
        self.current_task = None

    def get_user_input(self, delimiter="\n"):
        return self._text


def _all_text(events):
    return "\n".join(str(e) for e in events)


def test_agent_card_advertises_skills_and_interface():
    card = build_agent_card(base_url="http://localhost:8080", model="gemma4:e4b", version="0.1.4")

    assert card.name
    assert card.version == "0.1.4"
    skill_ids = {s.id for s in card.skills}
    assert {"task", "categorize", "judge"}.issubset(skill_ids)
    assert card.supported_interfaces, "card must advertise at least one interface"
    iface = card.supported_interfaces[0]
    assert iface.url == "http://localhost:8080/"


def test_register_a2a_serves_agent_card_on_fastapi():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from mayring_pi_agent.a2a_agent import register_a2a

    app = FastAPI()
    register_a2a(
        app,
        base_url="http://testserver",
        model="gemma4:e4b",
        ollama_url="http://localhost:11434",
        runner=lambda **kw: "stub",
    )
    client = TestClient(app)
    resp = client.get("/.well-known/agent-card.json")
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "MayringCoder Pi-Agent"
    assert any(s["id"] == "task" for s in body["skills"])


def test_executor_runs_task_with_user_input_and_emits_response():
    calls = {}

    def fake_runner(task, ollama_url, model, **kwargs):
        calls["task"] = task
        calls["ollama_url"] = ollama_url
        calls["model"] = model
        return "Das aktuelle Thema ist der A2A-Wrapper."

    executor = PiAgentExecutor(
        model="gemma4:e4b",
        ollama_url="http://localhost:11434",
        runner=fake_runner,
    )
    queue = _RecordingQueue()
    ctx = _FakeContext("Worüber reden wir gerade?")

    asyncio.run(executor.execute(ctx, queue))

    assert calls["task"] == "Worüber reden wir gerade?"
    assert calls["model"] == "gemma4:e4b"
    assert calls["ollama_url"] == "http://localhost:11434"
    assert "A2A-Wrapper" in _all_text(queue.events)
