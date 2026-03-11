import json
import tempfile
import unittest

from fastapi.testclient import TestClient

import proxy


class FakeResponse:
    def __init__(self, status_code=200, body=b"", chunks=None):
        self.status_code = status_code
        self._body = body
        self._chunks = list(chunks or [])
        self.closed = False

    async def aread(self):
        return self._body

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk

    async def aclose(self):
        self.closed = True


class FakeHttpClient:
    def __init__(self):
        self.requests = []
        self.responses = []

    def queue_response(self, response):
        self.responses.append(response)

    def build_request(self, method, url, json=None, headers=None):
        req = {"method": method, "url": url, "json": json, "headers": headers or {}}
        self.requests.append(req)
        return req

    async def send(self, request, stream=False):
        request["stream"] = stream
        if not self.responses:
            raise AssertionError("No fake response queued")
        return self.responses.pop(0)

    async def aclose(self):
        return None


class ProxyFormatTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        proxy.CACHE_DB_PATH = f"{self.tempdir.name}/sig_cache.db"
        proxy._db_conn = None
        proxy._token_cache["token"] = ""
        proxy._token_cache["ts"] = 0.0
        proxy.VERTEX_AI_PROJECT = "test-project"
        proxy.VERTEX_AI_REGION = "us-west1"
        proxy.ANTHROPIC_API_KEY = "test-key"
        self.fake_http = FakeHttpClient()
        proxy.http_client = self.fake_http
        self.client = TestClient(proxy.app)
        self.client.__enter__()
        proxy.http_client = self.fake_http

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.tempdir.cleanup()

    def test_vertex_non_stream_returns_json_and_preserves_user_blocks(self):
        upstream = {
            "id": "chatcmpl-vertex",
            "object": "chat.completion",
            "created": 123,
            "model": "google/gemini-3.1-flash-lite-preview",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        self.fake_http.queue_response(FakeResponse(body=json.dumps(upstream).encode()))

        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "google/gemini-3.1-flash-lite-preview",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
                    ],
                }],
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("application/json"))
        self.assertEqual(response.json()["id"], "chatcmpl-vertex")
        sent_body = self.fake_http.requests[0]["json"]
        self.assertFalse(self.fake_http.requests[0]["stream"])
        self.assertIsInstance(sent_body["messages"][1]["content"], list)
        self.assertEqual(sent_body["messages"][1]["content"][1]["type"], "image_url")

    def test_anthropic_non_stream_maps_required_tool_choice_and_length_finish_reason(self):
        upstream = {
            "id": "msg_123",
            "content": [{"type": "text", "text": "partial"}],
            "stop_reason": "max_tokens",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        self.fake_http.queue_response(FakeResponse(body=json.dumps(upstream).encode()))

        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "anthropic/claude-sonnet-4-5",
                "stream": False,
                "tool_choice": "required",
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }],
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["object"], "chat.completion")
        self.assertIn("created", body)
        self.assertEqual(body["choices"][0]["finish_reason"], "length")
        sent_body = self.fake_http.requests[0]["json"]
        self.assertEqual(sent_body["tool_choice"], {"type": "any"})

    def test_anthropic_stream_emits_openai_style_chunks(self):
        chunks = [
            b'data: {"type":"message_start","message":{"id":"msg_abc"}}\n\n',
            b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hel"}}\n\n',
            b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"lo"}}\n\n',
            b'data: {"type":"message_delta","delta":{"stop_reason":"max_tokens"}}\n\n',
            b'data: {"type":"message_stop"}\n\n',
        ]
        self.fake_http.queue_response(FakeResponse(chunks=chunks))

        with self.client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "anthropic/claude-sonnet-4-5",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        ) as response:
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
            text = "".join(response.iter_text())

        self.assertIn('"object": "chat.completion.chunk"', text)
        self.assertIn('"role": "assistant"', text)
        self.assertIn('"model": "claude-sonnet-4-5"', text)
        self.assertIn('"finish_reason": "length"', text)
        self.assertIn("data: [DONE]", text)


if __name__ == "__main__":
    unittest.main()
