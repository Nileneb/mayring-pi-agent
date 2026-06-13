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
        post_fn=post_fn, embed_fn=embed_fn, model="bge-m3")
    assert status == "dispatched"
    paths = [p for p, _ in posts]
    assert paths == ["/embed_pool/claim", "/embed_pool/complete"]
    assert posts[1][1] == {"embed_id": "emb_1", "vector": [0.1, 0.2, 0.3]}


def test_embed_once_empty_when_no_job():
    def post_fn(path, body):
        return {"job": None}

    status = pi_worker._embed_once(
        post_fn=post_fn, embed_fn=lambda t, m: [0.0], model="bge-m3")
    assert status == "empty"


def test_golden_once_claims_and_completes():
    posts = []

    def post_fn(path, body):
        posts.append((path, body))
        if path == "/embed_pool/golden/claim":
            return {"job": {"embed_id": "emb_g", "text": "probe", "model": "bge-m3"}}
        return {}

    status = pi_worker._golden_once(
        post_fn=post_fn, embed_fn=lambda t, m: [0.6, 0.8], model="bge-m3")
    assert status == "dispatched"
    assert [p for p, _ in posts] == ["/embed_pool/golden/claim", "/embed_pool/golden/complete"]
    assert posts[1][1] == {"embed_id": "emb_g", "vector": [0.6, 0.8]}


def test_golden_once_empty_when_no_job():
    assert pi_worker._golden_once(
        post_fn=lambda p, b: {"job": None}, embed_fn=lambda t, m: [0.0], model="bge-m3") == "empty"


def test_gpu_idle_true_when_idle():
    calls = []

    def get_fn(path):
        calls.append(path)
        return {"idle": True, "idle_seconds": 300.0}

    assert pi_worker._gpu_idle(get_fn=get_fn, idle_for=120) is True
    assert calls == ["/stats/gpu-idle?idle_for=120"]


def test_gpu_idle_false_when_busy():
    assert pi_worker._gpu_idle(
        get_fn=lambda p: {"idle": False, "idle_seconds": 5.0}, idle_for=120) is False


def test_gpu_idle_none_on_probe_failure():
    # endpoint unreachable / 403 without admin scope -> caller defers (real work priority)
    assert pi_worker._gpu_idle(get_fn=lambda p: None, idle_for=120) is None
