import json

import httpx
import httpx2
import pytest

from router.jev import DEFAULT_INSTRUCTIONS, QUESTION_KEY, FakeJev, TypeSafeJev
from router.llm import FakeLLM, OllamaClient, OpenAICompatClient

PARAMS = dict(max_output_tokens=600, temperature=0, seed=7)


def test_ollama_client_request_and_parse():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"model": "qwen3:8b", "message": {"content": "ANSWER: B"}, "prompt_eval_count": 40, "eval_count": 5},
        )

    client = OllamaClient("http://ollama.test", transport=httpx.MockTransport(handler), **PARAMS)
    r = client.complete("qwen3:8b", "hello")

    assert seen["path"] == "/api/chat"
    assert seen["body"]["think"] is False
    assert seen["body"]["stream"] is False
    assert seen["body"]["options"] == {"temperature": 0, "seed": 7, "num_predict": 600}
    assert seen["body"]["messages"] == [{"role": "user", "content": "hello"}]
    assert (r.text, r.input_tokens, r.output_tokens) == ("ANSWER: B", 40, 5)


def test_openai_compat_client_request_and_parse():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "vendor/m",
                "choices": [{"message": {"content": "ANSWER: 42"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            },
        )

    client = OpenAICompatClient(
        "http://llm.test/api/v1", "test-key", transport=httpx.MockTransport(handler), **PARAMS
    )
    r = client.complete("vendor/m", "hello")

    assert seen["auth"] == "Bearer test-key"
    assert seen["path"] == "/api/v1/chat/completions"
    assert seen["body"]["max_tokens"] == 600 and seen["body"]["temperature"] == 0
    assert (r.text, r.input_tokens, r.output_tokens) == ("ANSWER: 42", 12, 3)


def test_typesafe_jev_request_and_parse():
    seen = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx2.Response(
            200,
            json={
                "model": "jev-1.14.0",
                "answers": {QUESTION_KEY: {"type": "noul", "noul": 0.83}},
                "usage": {"input_tokens": 57, "output_tokens": 1},
            },
        )

    jev = TypeSafeJev(
        "jev-1.14.0",
        api_key="test-key",
        base_url="http://jev.test",
        max_retries=0,
        timeout_s=2.0,
        transport=httpx2.MockTransport(handler),
    )
    r = jev.ask("What is 2+2?")

    assert seen["path"].endswith("/v1/systemone")
    assert seen["body"]["model"] == "jev-1.14.0"
    assert seen["body"]["state"] == {"prompt": "What is 2+2?"}
    assert seen["body"]["questions"][QUESTION_KEY]["type"] == "noul"
    assert seen["body"]["questions"][QUESTION_KEY]["instructions"] == DEFAULT_INSTRUCTIONS
    assert r.p_small_ok == 0.83
    assert (r.input_tokens, r.model) == (57, "jev-1.14.0")


def test_fake_jev_supports_several_phrasings():
    r = FakeJev(lambda prompt: 0.25).ask("x", {"a": "one", "b": "two"})
    assert r.probs == {"a": 0.25, "b": 0.25}


def test_fake_llm_records_calls():
    llm = FakeLLM(lambda model, prompt: f"{model} says ANSWER: A")
    r = llm.complete("small", "question text")
    assert r.text == "small says ANSWER: A"
    assert llm.calls == [("small", "question text")]
    assert r.input_tokens == 2 and r.output_tokens == 4


def test_ollama_client_retries_a_stalled_request():
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) < 3:
            raise httpx.ReadTimeout("stalled", request=request)
        return httpx.Response(200, json={"message": {"content": "ANSWER: A"}, "prompt_eval_count": 1, "eval_count": 1})

    client = OllamaClient("http://ollama.test", transport=httpx.MockTransport(handler), max_retries=2, retry_wait_s=0, **PARAMS)
    assert client.complete("m", "x").text == "ANSWER: A"
    assert len(attempts) == 3


def test_ollama_client_gives_up_after_max_retries():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("stalled", request=request)

    client = OllamaClient("http://ollama.test", transport=httpx.MockTransport(handler), max_retries=1, retry_wait_s=0, **PARAMS)
    with pytest.raises(httpx.ReadTimeout):
        client.complete("m", "x")


def test_ollama_client_retries_server_errors_but_not_client_errors():
    attempts = []

    def flaky(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            return httpx.Response(500, json={"error": "runner crashed"})
        return httpx.Response(200, json={"message": {"content": "ANSWER: A"}, "prompt_eval_count": 1, "eval_count": 1})

    client = OllamaClient("http://ollama.test", transport=httpx.MockTransport(flaky), retry_wait_s=0, **PARAMS)
    assert client.complete("m", "x").text == "ANSWER: A"
    assert len(attempts) == 2

    calls = []

    def missing(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(404, json={"error": "model not found"})

    client = OllamaClient("http://ollama.test", transport=httpx.MockTransport(missing), retry_wait_s=0, **PARAMS)
    with pytest.raises(httpx.HTTPStatusError):
        client.complete("m", "x")
    assert len(calls) == 1


def test_ollama_degenerate_output_is_an_empty_answer_not_a_crash():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(500, json={"error": "prediction aborted, token repeat limit reached"})

    client = OllamaClient("http://ollama.test", transport=httpx.MockTransport(handler), retry_wait_s=0, **PARAMS)
    r = client.complete("m", "x" * 35)
    assert len(calls) == 1  # deterministic failure, so no retry
    assert r.text == "" and r.error == "token repeat limit reached"
    assert r.input_tokens == 10 and r.output_tokens == 600  # billed as a run to the cap
