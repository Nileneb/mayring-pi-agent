"""Pi-Agent HTTP server — FastAPI wrapper on port 8091."""
from __future__ import annotations

import os
import re

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError as exc:
    raise ImportError("pip install fastapi uvicorn") from exc

from src.agents.pi import run_task_with_memory

app = FastAPI(title="MayringCoder Pi-Agent", version="1.0.0")

_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:0.6b")
_PORT = int(os.getenv("PI_PORT", "8091"))


class TaskRequest(BaseModel):
    task: str
    repo_slug: str | None = None
    system_prompt: str | None = None
    max_tool_calls: int = 5
    timeout: float = 180.0


class TaskResponse(BaseModel):
    result: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/task", response_model=TaskResponse)
async def task(req: TaskRequest) -> TaskResponse:
    repo_slug = req.repo_slug or ""
    safe_repo_slug: str | None = None
    if repo_slug and ".." not in repo_slug and "/" not in repo_slug and "\\" not in repo_slug:
        if re.fullmatch(r"[A-Za-z0-9._-]+", repo_slug):
            safe_repo_slug = repo_slug

    try:
        result = run_task_with_memory(
            task=req.task,
            ollama_url=_OLLAMA_URL,
            model=_MODEL,
            repo_slug=safe_repo_slug,
            system_prompt=req.system_prompt,
            max_tool_calls=req.max_tool_calls,
            timeout=req.timeout,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return TaskResponse(result=result)


def main() -> None:
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=_PORT)


if __name__ == "__main__":
    main()
