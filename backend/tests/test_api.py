"""End-to-end and edge-case tests for the PSB Legal Assistant backend API."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_assistant_port
from app.core.config import Settings
from app.core.exceptions import NotFoundError
from app.integrations.base import AssistantReply, ChatTurn
from app.integrations.llm_client import SYSTEM_PROMPT
from app.main import create_app
from app.repositories.chat_repo import DEFAULT_CHAT_TITLE
from app.repositories.document_repo import DocumentRepository
from app.schemas.chat import SourceCitation
from tests.conftest import FakeAssistant, SampleCorpus


class TestHealthEndpoint:
    """Tests for `GET /api/health`."""

    def test_health_ok_when_llm_configured(self, client: TestClient) -> None:
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json() == {
            "status": "ok",
            "database": "ok",
            "assistant_backend": "direct_llm",
            "llm_configured": True,
        }

    @pytest.mark.parametrize("api_key", ["", "your-api-key-here"])
    def test_health_reports_unconfigured_llm(
        self,
        db_url: str,
        data_dir: Path,
        open_real_llm_client,
        api_key: str,
    ) -> None:
        settings = Settings(
            llm_base_url="https://mock-llm.local/v1",
            llm_api_key=api_key,
            database_url=db_url,
            data_dir=data_dir,
        )
        with open_real_llm_client(settings, transport=None) as (real_client, _):
            response = real_client.get("/api/health")
            assert response.status_code == 200
            payload = response.json()
            assert payload["status"] == "ok"
            assert payload["database"] == "ok"
            assert payload["assistant_backend"] == "direct_llm"
            assert payload["llm_configured"] is False


class TestChatCrud:
    """Tests for creating, listing/searching, and deleting chat sessions."""

    def test_create_chat_without_body_and_with_custom_title(
        self, client: TestClient
    ) -> None:
        empty_resp = client.post("/api/chats")
        assert empty_resp.status_code == 201
        empty_chat = empty_resp.json()
        assert empty_chat["id"]
        assert empty_chat["title"] == DEFAULT_CHAT_TITLE
        assert empty_chat["created_at"]
        assert empty_chat["updated_at"]

        empty_json_resp = client.post("/api/chats", json={})
        assert empty_json_resp.status_code == 201
        assert empty_json_resp.json()["title"] == DEFAULT_CHAT_TITLE

        blank_resp = client.post("/api/chats", json={"title": "   "})
        assert blank_resp.status_code == 201
        assert blank_resp.json()["title"] == DEFAULT_CHAT_TITLE

        custom_resp = client.post("/api/chats", json={"title": "  Трудовой договор  "})
        assert custom_resp.status_code == 201
        custom_chat = custom_resp.json()
        assert custom_chat["title"] == "Трудовой договор"
        assert custom_chat["id"] != empty_chat["id"]

    def test_create_chat_rejects_overlong_title(self, client: TestClient) -> None:
        response = client.post("/api/chats", json={"title": "А" * 256})
        assert response.status_code == 422

    def test_list_chats_and_case_insensitive_cyrillic_filter(
        self, client: TestClient
    ) -> None:
        c1 = client.post("/api/chats", json={"title": "Трудовой Договор"}).json()
        c2 = client.post("/api/chats", json={"title": "Положение о закупках"}).json()
        c3 = client.post("/api/chats", json={"title": "NDA Contract Review"}).json()

        all_resp = client.get("/api/chats")
        assert all_resp.status_code == 200
        all_data = all_resp.json()
        assert all_data["total"] == 3
        assert {item["id"] for item in all_data["items"]} == {
            c1["id"],
            c2["id"],
            c3["id"],
        }

        cyr_lower = client.get("/api/chats", params={"q": "трудовой"})
        assert cyr_lower.status_code == 200
        assert cyr_lower.json()["total"] == 1
        assert cyr_lower.json()["items"][0]["id"] == c1["id"]

        cyr_upper = client.get("/api/chats", params={"q": "ЗАКУПК"})
        assert cyr_upper.status_code == 200
        assert cyr_upper.json()["total"] == 1
        assert cyr_upper.json()["items"][0]["id"] == c2["id"]

        latin_resp = client.get("/api/chats", params={"q": "contract"})
        assert latin_resp.status_code == 200
        assert latin_resp.json()["total"] == 1
        assert latin_resp.json()["items"][0]["id"] == c3["id"]

        no_match = client.get("/api/chats", params={"q": "несуществующий"})
        assert no_match.status_code == 200
        assert no_match.json() == {"items": [], "total": 0}

        blank_query = client.get("/api/chats", params={"q": "   "})
        assert blank_query.status_code == 200
        assert blank_query.json()["total"] == 3

    def test_delete_chat_and_subsequent_404s(self, client: TestClient) -> None:
        chat_id = client.post("/api/chats", json={"title": "Временный чат"}).json()["id"]
        send_resp = client.post(
            f"/api/chats/{chat_id}/messages",
            json={"content": "Тестовый вопрос"},
        )
        assert send_resp.status_code == 200

        del_resp = client.delete(f"/api/chats/{chat_id}")
        assert del_resp.status_code == 204
        assert del_resp.content == b""

        assert client.get(f"/api/chats/{chat_id}/messages").status_code == 404
        assert (
            client.post(
                f"/api/chats/{chat_id}/messages",
                json={"content": "После удаления"},
            ).status_code
            == 404
        )
        assert client.delete(f"/api/chats/{chat_id}").status_code == 404

        listed_ids = [item["id"] for item in client.get("/api/chats").json()["items"]]
        assert chat_id not in listed_ids

    def test_unknown_chat_id_returns_404(
        self, client: TestClient, fake_assistant: FakeAssistant
    ) -> None:
        unknown_id = "00000000-0000-0000-0000-000000000000"

        get_resp = client.get(f"/api/chats/{unknown_id}/messages")
        assert get_resp.status_code == 404
        assert "detail" in get_resp.json()

        post_resp = client.post(
            f"/api/chats/{unknown_id}/messages",
            json={"content": "Привет"},
        )
        assert post_resp.status_code == 404
        assert "detail" in post_resp.json()
        assert fake_assistant.calls == []

        del_resp = client.delete(f"/api/chats/{unknown_id}")
        assert del_resp.status_code == 404


class TestMultiTurnDialogAndPersistence:
    """Tests for message persistence, history context passing, validation, and SQLite durability."""

    def test_multi_turn_dialog_auto_title_and_metadata(
        self, client: TestClient, fake_assistant: FakeAssistant
    ) -> None:
        chat = client.post("/api/chats").json()
        chat_id = chat["id"]
        assert chat["title"] == DEFAULT_CHAT_TITLE

        turn1_resp = client.post(
            f"/api/chats/{chat_id}/messages",
            json={"content": "  Какой срок действия доверенности по ГК РФ?  "},
        )
        assert turn1_resp.status_code == 200
        turn1 = turn1_resp.json()
        assert turn1["chat_id"] == chat_id
        assert turn1["user_message"]["role"] == "user"
        assert (
            turn1["user_message"]["content"]
            == "Какой срок действия доверенности по ГК РФ?"
        )
        assert turn1["assistant_message"]["role"] == "assistant"
        assert (
            turn1["assistant_message"]["content"]
            == "Правовой ответ на запрос: Какой срок действия доверенности по ГК РФ?"
        )
        assert turn1["assistant_message"]["intent"] == "legal_qa"
        assert turn1["assistant_message"]["refused"] is False
        assert turn1["assistant_message"]["clarification_needed"] is False
        assert len(turn1["assistant_message"]["sources"]) == 1
        assert (
            turn1["assistant_message"]["sources"][0]["document_id"] == "compliance_reg"
        )

        # Auto-title updated from first user message
        chats_after_first = client.get("/api/chats").json()["items"]
        assert chats_after_first[0]["id"] == chat_id
        assert (
            chats_after_first[0]["title"]
            == "Какой срок действия доверенности по ГК РФ?"
        )

        # Second turn receives full prior conversation history and persists custom flags
        fake_assistant.queued_replies.append(
            AssistantReply(
                content="Уточните, пожалуйста, вид доверенности.",
                sources=[
                    SourceCitation(
                        document_id="credit_rules",
                        document_title="Правила кредитования",
                        excerpt="Доверенность оформляется письменно.",
                    )
                ],
                refused=True,
                intent="clarify_scope",
                clarification_needed=True,
            )
        )
        turn2_resp = client.post(
            f"/api/chats/{chat_id}/messages",
            json={"content": "А если срок в документе не указан?"},
        )
        assert turn2_resp.status_code == 200
        turn2 = turn2_resp.json()
        assert turn2["assistant_message"]["refused"] is True
        assert turn2["assistant_message"]["intent"] == "clarify_scope"
        assert turn2["assistant_message"]["clarification_needed"] is True
        assert len(turn2["assistant_message"]["sources"]) == 1

        # Verify exact ChatTurn history passed to AssistantPort on each turn
        assert len(fake_assistant.calls) == 2
        assert fake_assistant.calls[0] == [
            ChatTurn(
                role="user",
                content="Какой срок действия доверенности по ГК РФ?",
            )
        ]
        assert fake_assistant.calls[1] == [
            ChatTurn(
                role="user",
                content="Какой срок действия доверенности по ГК РФ?",
            ),
            ChatTurn(
                role="assistant",
                content="Правовой ответ на запрос: Какой срок действия доверенности по ГК РФ?",
            ),
            ChatTurn(
                role="user",
                content="А если срок в документе не указан?",
            ),
        ]

        # Verify chronological history endpoint
        history_resp = client.get(f"/api/chats/{chat_id}/messages")
        assert history_resp.status_code == 200
        history_data = history_resp.json()
        assert history_data["chat_id"] == chat_id
        messages = history_data["messages"]
        assert [m["role"] for m in messages] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        assert [m["content"] for m in messages] == [
            "Какой срок действия доверенности по ГК РФ?",
            "Правовой ответ на запрос: Какой срок действия доверенности по ГК РФ?",
            "А если срок в документе не указан?",
            "Уточните, пожалуйста, вид доверенности.",
        ]
        assert messages[1]["sources"][0]["provision_id"] == "p-1.1"
        assert messages[3]["refused"] is True
        assert messages[3]["clarification_needed"] is True
        assert messages[3]["intent"] == "clarify_scope"

    def test_custom_title_is_preserved_and_long_auto_title_is_truncated(
        self, client: TestClient
    ) -> None:
        custom_chat_id = client.post(
            "/api/chats", json={"title": "Закрепленный заголовок"}
        ).json()["id"]
        client.post(
            f"/api/chats/{custom_chat_id}/messages",
            json={"content": "Первый вопрос пользователя"},
        )

        auto_chat_id = client.post("/api/chats").json()["id"]
        long_question = "Очень длинный юридический вопрос про порядок согласования договора поставки в банке"
        client.post(
            f"/api/chats/{auto_chat_id}/messages",
            json={"content": long_question},
        )

        chats = {item["id"]: item for item in client.get("/api/chats").json()["items"]}
        assert chats[custom_chat_id]["title"] == "Закрепленный заголовок"
        assert chats[auto_chat_id]["title"].endswith("...")
        assert len(chats[auto_chat_id]["title"]) <= 63

    @pytest.mark.parametrize(
        "bad_content",
        ["", "   ", "\n\t  \r\n", "Я" * 10_001],
    )
    def test_invalid_message_content_rejected_422(
        self,
        client: TestClient,
        fake_assistant: FakeAssistant,
        bad_content: str,
    ) -> None:
        chat_id = client.post("/api/chats").json()["id"]
        response = client.post(
            f"/api/chats/{chat_id}/messages",
            json={"content": bad_content},
        )
        assert response.status_code == 422
        assert fake_assistant.calls == []
        assert client.get(f"/api/chats/{chat_id}/messages").json()["messages"] == []

    def test_history_limit_caps_assistant_context_without_truncating_db(
        self,
        db_url: str,
        data_dir: Path,
        fake_assistant: FakeAssistant,
    ) -> None:
        limited_settings = Settings(
            llm_base_url="https://mock-llm.local/v1",
            llm_api_key="sk-test-fake-key",
            llm_history_limit=2,
            database_url=db_url,
            data_dir=data_dir,
        )
        limited_app = create_app(limited_settings)
        limited_app.dependency_overrides[get_assistant_port] = lambda: fake_assistant

        with TestClient(limited_app) as limited_client:
            chat_id = limited_client.post("/api/chats").json()["id"]
            limited_client.post(
                f"/api/chats/{chat_id}/messages", json={"content": "Вопрос 1"}
            )
            limited_client.post(
                f"/api/chats/{chat_id}/messages", json={"content": "Вопрос 2"}
            )

            assert len(fake_assistant.calls[1]) == 2
            assert fake_assistant.calls[1] == [
                ChatTurn(role="assistant", content="Правовой ответ на запрос: Вопрос 1"),
                ChatTurn(role="user", content="Вопрос 2"),
            ]
            stored = limited_client.get(f"/api/chats/{chat_id}/messages").json()[
                "messages"
            ]
            assert len(stored) == 4

    def test_sqlite_persistence_across_app_recreation(
        self,
        test_settings: Settings,
        fake_assistant: FakeAssistant,
    ) -> None:
        app1 = create_app(test_settings)
        app1.dependency_overrides[get_assistant_port] = lambda: fake_assistant
        with TestClient(app1) as client1:
            chat_id = client1.post("/api/chats").json()["id"]
            send_resp = client1.post(
                f"/api/chats/{chat_id}/messages",
                json={"content": "Сохранится ли история после перезапуска?"},
            )
            assert send_resp.status_code == 200

        # Re-create application from scratch against the same SQLite database file
        app2 = create_app(test_settings)
        app2.dependency_overrides[get_assistant_port] = lambda: FakeAssistant()
        with TestClient(app2) as client2:
            chats = client2.get("/api/chats").json()["items"]
            assert len(chats) == 1
            assert chats[0]["id"] == chat_id
            assert chats[0]["title"] == "Сохранится ли история после перезапуска?"

            messages = client2.get(f"/api/chats/{chat_id}/messages").json()["messages"]
            assert len(messages) == 2
            assert messages[0]["role"] == "user"
            assert messages[0]["content"] == "Сохранится ли история после перезапуска?"
            assert messages[1]["role"] == "assistant"
            assert messages[1]["intent"] == "legal_qa"
            assert len(messages[1]["sources"]) == 1


class TestDirectLLMBackendErrorHandling:
    """Tests exercising real `DirectLLMBackend` via `httpx.MockTransport` without fake overrides."""

    @pytest.mark.parametrize("api_key", ["", "your-api-key-here"])
    def test_missing_or_placeholder_api_key_returns_503_and_saves_user_message(
        self,
        db_url: str,
        data_dir: Path,
        open_real_llm_client,
        api_key: str,
    ) -> None:
        settings = Settings(
            llm_base_url="https://mock-llm.local/v1",
            llm_api_key=api_key,
            database_url=db_url,
            data_dir=data_dir,
        )
        with open_real_llm_client(settings, transport=None) as (real_client, _):
            chat_id = real_client.post("/api/chats").json()["id"]
            resp = real_client.post(
                f"/api/chats/{chat_id}/messages",
                json={"content": "Вопрос при отсутствии ключа"},
            )
            assert resp.status_code == 503
            assert "detail" in resp.json()

            messages = real_client.get(f"/api/chats/{chat_id}/messages").json()[
                "messages"
            ]
            assert len(messages) == 1
            assert messages[0]["role"] == "user"
            assert messages[0]["content"] == "Вопрос при отсутствии ключа"

    @pytest.mark.parametrize("status_code", [500, 429])
    def test_upstream_retryable_failure_returns_502_without_key_leakage(
        self,
        db_url: str,
        data_dir: Path,
        open_real_llm_client,
        caplog: pytest.LogCaptureFixture,
        status_code: int,
    ) -> None:
        secret_key = "sk-top-secret-psb-key-987654321"
        settings = Settings(
            llm_base_url="https://mock-llm.local/v1",
            llm_api_key=secret_key,
            database_url=db_url,
            data_dir=data_dir,
        )
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            return httpx.Response(
                status_code,
                json={"error": {"message": f"Upstream failure {status_code}"}},
            )

        caplog.set_level(logging.INFO)
        transport = httpx.MockTransport(handler)
        with open_real_llm_client(settings, transport=transport) as (real_client, _):
            chat_id = real_client.post("/api/chats").json()["id"]
            resp = real_client.post(
                f"/api/chats/{chat_id}/messages",
                json={"content": "Запрос со сбоем апстрима"},
            )

            assert attempts == 2
            assert resp.status_code == 502
            assert resp.json() == {"detail": "Upstream LLM request failed"}
            assert secret_key not in resp.text
            assert secret_key not in caplog.text

            messages = real_client.get(f"/api/chats/{chat_id}/messages").json()[
                "messages"
            ]
            assert len(messages) == 1
            assert messages[0]["role"] == "user"

    def test_upstream_non_retryable_4xx_and_invalid_json_return_502(
        self,
        db_url: str,
        data_dir: Path,
        open_real_llm_client,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        secret_key = "sk-top-secret-client-err-key"
        settings = Settings(
            llm_base_url="https://mock-llm.local/v1",
            llm_api_key=secret_key,
            database_url=db_url,
            data_dir=data_dir,
        )
        attempts = 0

        def handler_401(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            return httpx.Response(401, json={"error": "Unauthorized"})

        caplog.set_level(logging.INFO)
        with open_real_llm_client(
            settings, transport=httpx.MockTransport(handler_401)
        ) as (real_client, _):
            chat_id = real_client.post("/api/chats").json()["id"]
            resp = real_client.post(
                f"/api/chats/{chat_id}/messages",
                json={"content": "Ошибка 401"},
            )
            assert attempts == 1
            assert resp.status_code == 502
            assert secret_key not in resp.text
            assert secret_key not in caplog.text

        def handler_bad_json(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"not-valid-json")

        with open_real_llm_client(
            settings, transport=httpx.MockTransport(handler_bad_json)
        ) as (real_client, _):
            chat_id = real_client.post("/api/chats").json()["id"]
            resp = real_client.post(
                f"/api/chats/{chat_id}/messages",
                json={"content": "Невалидный JSON"},
            )
            assert resp.status_code == 502

    def test_upstream_timeout_returns_504_without_key_leakage(
        self,
        db_url: str,
        data_dir: Path,
        open_real_llm_client,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        secret_key = "sk-timeout-secret-key-000111"
        settings = Settings(
            llm_base_url="https://mock-llm.local/v1",
            llm_api_key=secret_key,
            llm_timeout_seconds=1.0,
            database_url=db_url,
            data_dir=data_dir,
        )
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            raise httpx.ReadTimeout("upstream read timed out", request=request)

        caplog.set_level(logging.INFO)
        with open_real_llm_client(
            settings, transport=httpx.MockTransport(handler)
        ) as (real_client, _):
            chat_id = real_client.post("/api/chats").json()["id"]
            resp = real_client.post(
                f"/api/chats/{chat_id}/messages",
                json={"content": "Запрос с таймаутом"},
            )

            assert attempts == 2
            assert resp.status_code == 504
            assert resp.json() == {"detail": "Upstream LLM request timed out"}
            assert secret_key not in resp.text
            assert secret_key not in caplog.text

    @pytest.mark.parametrize("transient_status", [500, 429])
    def test_retry_succeeds_on_second_attempt_after_transient_error(
        self,
        db_url: str,
        data_dir: Path,
        open_real_llm_client,
        transient_status: int,
    ) -> None:
        secret_key = "sk-retry-success-secret"
        settings = Settings(
            llm_base_url="https://mock-llm.local/v1/",
            llm_api_key=secret_key,
            llm_model="deepseek/test-model",
            llm_temperature=0.0,
            llm_max_tokens=256,
            database_url=db_url,
            data_dir=data_dir,
        )
        captured_requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_requests.append(request)
            if len(captured_requests) == 1:
                return httpx.Response(transient_status, json={"error": "transient"})
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "  Ответ успешно получен после повтора.  ",
                            }
                        }
                    ]
                },
            )

        with open_real_llm_client(
            settings, transport=httpx.MockTransport(handler)
        ) as (real_client, _):
            chat_id = real_client.post("/api/chats").json()["id"]
            resp = real_client.post(
                f"/api/chats/{chat_id}/messages",
                json={"content": "Повтори запрос при сбое"},
            )

            assert resp.status_code == 200
            data = resp.json()
            assert (
                data["assistant_message"]["content"]
                == "Ответ успешно получен после повтора."
            )
            assert len(captured_requests) == 2

            req = captured_requests[1]
            assert req.method == "POST"
            assert str(req.url) == "https://mock-llm.local/v1/chat/completions"
            assert req.headers["Authorization"] == f"Bearer {secret_key}"
            body = json.loads(req.content.decode("utf-8"))
            assert body["model"] == "deepseek/test-model"
            assert body["temperature"] == 0.0
            assert body["max_tokens"] == 256
            assert body["messages"] == [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Повтори запрос при сбое"},
            ]


class TestDocumentsApi:
    """Tests for `GET /api/documents`, manifest collision resolution, `.docx`, and traversal protection."""

    def test_list_documents_handles_extensions_and_manifest_id_collisions(
        self, client: TestClient, sample_corpus: SampleCorpus
    ) -> None:
        resp = client.get("/api/documents")
        assert resp.status_code == 200
        payload = resp.json()
        items = payload["items"]
        assert payload["total"] == 4
        assert len(items) == 4

        doc_ids = [item["document_id"] for item in items]
        assert len(set(doc_ids)) == 4

        by_filename = {item["filename"]: item for item in items}
        assert set(by_filename.keys()) == {
            "policies/credit_v1.txt",
            "policies/credit_v2.md",
            "regulation.docx",
            "unlisted_note.txt",
        }

        # Two files with distinct source_hash sharing "credit_rules" in manifest get deterministic unique IDs
        expected_v1_id = (
            f"{sample_corpus.colliding_base_id}__"
            f"{sample_corpus.digests['policies/credit_v1.txt'][:8]}"
        )
        expected_v2_id = (
            f"{sample_corpus.colliding_base_id}__"
            f"{sample_corpus.digests['policies/credit_v2.md'][:8]}"
        )
        assert expected_v1_id != expected_v2_id

        v1 = by_filename["policies/credit_v1.txt"]
        assert v1["document_id"] == expected_v1_id
        assert v1["version_id"] == "v1"
        assert v1["title"] == "Правила кредитования МСБ (редакция 1)"
        assert v1["amends_document_id"] is None

        v2 = by_filename["policies/credit_v2.md"]
        assert v2["document_id"] == expected_v2_id
        assert v2["version_id"] == "v2"
        assert v2["title"] == "Правила кредитования МСБ (редакция 2)"
        assert v2["amends_document_id"] == sample_corpus.colliding_base_id

        docx_item = by_filename["regulation.docx"]
        assert docx_item["document_id"] == "compliance_reg"
        assert docx_item["version_id"] == "v1"
        assert docx_item["title"] == "Внутренний регламент комплаенса"

        note_item = by_filename["unlisted_note.txt"]
        assert note_item["document_id"] == "unlisted_note"
        assert note_item["version_id"] is None

        # Determinism check across repeated listing calls
        resp_again = client.get("/api/documents")
        assert resp_again.status_code == 200
        assert resp_again.json() == payload

    def test_get_document_detail_for_all_listed_documents_including_docx(
        self, client: TestClient
    ) -> None:
        listed = client.get("/api/documents").json()["items"]
        by_filename = {item["filename"]: item for item in listed}

        for summary in listed:
            detail_resp = client.get(f"/api/documents/{summary['document_id']}")
            assert detail_resp.status_code == 200
            detail = detail_resp.json()
            assert detail["document_id"] == summary["document_id"]
            assert detail["version_id"] == summary["version_id"]
            assert detail["title"] == summary["title"]
            assert detail["filename"] == summary["filename"]
            assert detail["amends_document_id"] == summary["amends_document_id"]
            assert detail["char_count"] == len(detail["content"])
            assert detail["char_count"] > 0

        docx_detail = client.get(
            f"/api/documents/{by_filename['regulation.docx']['document_id']}"
        ).json()
        assert "Внутренний регламент комплаенса" in docx_detail["content"]
        assert "Проверка контрагента обязательна." in docx_detail["content"]

        v1_detail = client.get(
            f"/api/documents/{by_filename['policies/credit_v1.txt']['document_id']}"
        ).json()
        v2_detail = client.get(
            f"/api/documents/{by_filename['policies/credit_v2.md']['document_id']}"
        ).json()
        assert "письменной форме" in v1_detail["content"]
        assert "60 месяцев" in v2_detail["content"]

    @pytest.mark.parametrize(
        "malicious_or_missing_id",
        [
            "nonexistent-document-id",
            "../etc/passwd",
            "..%2F..%2Fetc%2Fpasswd",
            "foo\\bar",
            "foo%5Cbar",
            "..%5C..%5Cetc%5Cpasswd",
            "..",
            "bad..id",
            "bad%00id",
        ],
    )
    def test_get_document_404_and_path_traversal_protection(
        self, client: TestClient, malicious_or_missing_id: str
    ) -> None:
        response = client.get(f"/api/documents/{malicious_or_missing_id}")
        assert response.status_code == 404
        assert "root:" not in response.text

    @pytest.mark.parametrize(
        "raw_traversal_id",
        [
            "",
            "../etc/passwd",
            "../../etc/passwd",
            "foo\\bar",
            "..\\..\\etc\\passwd",
            "..",
            "bad..id",
            "bad\x00id",
        ],
    )
    def test_document_repository_rejects_invalid_and_traversal_ids(
        self, data_dir: Path, raw_traversal_id: str
    ) -> None:
        repo = DocumentRepository(data_dir)
        with pytest.raises(NotFoundError):
            repo.get_by_id(raw_traversal_id)
