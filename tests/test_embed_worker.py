from mayring_pi_agent import pi_worker


def test_embed_capability_added_when_enabled(monkeypatch):
    monkeypatch.setenv("PI_EMBED_ENABLED", "true")
    monkeypatch.setenv("PI_WORKER_CAPABILITIES", "local-gpu")
    assert "embed" in pi_worker._capabilities()


def test_embed_capability_absent_by_default(monkeypatch):
    monkeypatch.delenv("PI_EMBED_ENABLED", raising=False)
    monkeypatch.setenv("PI_WORKER_CAPABILITIES", "local-gpu")
    assert "embed" not in pi_worker._capabilities()


def test_embed_once_claims_computes_completes():
    posts = []

    def post_fn(path, body):
        posts.append((path, body))
        if path == "/embed_pool/claim":
            return {"job": {"embed_id": "emb_1", "text": "hello", "model": "bge-m3"}}
        if path == "/embed_pool/complete":
            return {"status": "verified", "verdict": "agreement"}
        return {}

    def embed_fn(text, model):
        assert text == "hello" and model == "bge-m3"
        return [0.1, 0.2, 0.3]

    status = pi_worker._embed_once(
        post_fn=post_fn, embed_fn=embed_fn,
        ollama_url="http://localhost:11434", model="bge-m3")
    assert status == "dispatched"
    paths = [p for p, _ in posts]
    assert paths == ["/embed_pool/claim", "/embed_pool/complete"]
    assert posts[1][1] == {"embed_id": "emb_1", "vector": [0.1, 0.2, 0.3]}


def test_embed_once_empty_when_no_job():
    def post_fn(path, body):
        return {"job": None}

    status = pi_worker._embed_once(
        post_fn=post_fn, embed_fn=lambda t, m: [0.0],
        ollama_url="http://localhost:11434", model="bge-m3")
    assert status == "empty"
