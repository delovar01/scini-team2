"""Проверка адаптеров моделей без реальных ключей: локальный HTTP-сервер вместо API.

    cd experiment
    python -m unittest test_run -v
"""

import json
import os
import ssl
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock
from urllib.parse import parse_qs

import run


class FakeAPI(BaseHTTPRequestHandler):
    """Изображает OAuth Сбера и /chat/completions. Все запросы складывает в server.log."""

    def log_message(self, *args):                     # не шуметь в консоль
        pass

    def _reply(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("utf-8")
        srv = self.server
        srv.log.append({"path": self.path, "headers": dict(self.headers), "body": raw})
        if self.path == "/oauth":
            srv.tokens_issued += 1
            self._reply(200, {"access_token": f"tok-{srv.tokens_issued}",
                              "expires_at": srv.expires_at})
        elif self.path == "/v1/chat/completions":
            auth = self.headers.get("Authorization", "")
            if srv.reject_tokens and auth in srv.reject_tokens:
                self._reply(401, {"message": "Unauthorized"})
                return
            model = json.loads(raw)["model"]
            self._reply(200, {"choices": [{"message": {"content": f"ответ {model}"}}]})
        else:
            self._reply(404, {})


class APITest(unittest.TestCase):
    def setUp(self):
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeAPI)
        self.srv.log, self.srv.tokens_issued, self.srv.reject_tokens = [], 0, set()
        self.srv.expires_at = 0
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.srv.server_address[1]}"
        self.env = mock.patch.dict(os.environ, {}, clear=False)
        self.env.start()
        for key in list(os.environ):
            if key.startswith(("MODEL_", "GIGACHAT_")):
                del os.environ[key]

    def tearDown(self):
        self.env.stop()
        self.srv.shutdown()
        self.srv.server_close()

    MSG = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]

    # ------------------------------------------------------------ openai

    def test_openai_default_kind_and_headers(self):
        os.environ.update(MODEL_1_BASE_URL=f"{self.base}/v1", MODEL_1_ID="m-free",
                          MODEL_1_API_KEY="k1")
        model = run.make_models("1")[0]
        self.assertIsInstance(model, run.OpenAICompatModel)
        self.assertEqual(model.chat(self.MSG), "ответ m-free")
        h = self.srv.log[0]["headers"]
        self.assertEqual(h["Authorization"], "Bearer k1")
        self.assertNotIn("OpenAI-Project", h)

    def test_openai_yandex_style(self):
        os.environ.update(MODEL_2_KIND="openai", MODEL_2_BASE_URL=f"{self.base}/v1",
                          MODEL_2_ID="gpt://b1gfolder/yandexgpt-lite/latest",
                          MODEL_2_API_KEY="k2", MODEL_2_AUTH_SCHEME="Api-Key",
                          MODEL_2_PROJECT="b1gfolder")
        model = run.make_models("2")[0]
        self.assertEqual(model.chat(self.MSG), "ответ gpt://b1gfolder/yandexgpt-lite/latest")
        h = self.srv.log[0]["headers"]
        self.assertEqual(h["Authorization"], "Api-Key k2")
        self.assertEqual(h["Openai-Project"], "b1gfolder")

    def test_openai_error_becomes_error_string(self):
        os.environ.update(MODEL_1_BASE_URL=f"{self.base}/nope", MODEL_1_ID="m")
        model = run.make_models("1")[0]
        model.retry_base = 0
        self.assertTrue(model.chat(self.MSG).startswith("[ERROR]"))
        self.assertEqual(len(self.srv.log), 3)          # три попытки

    def test_unknown_kind(self):
        os.environ.update(MODEL_1_KIND="claude", MODEL_1_BASE_URL="x", MODEL_1_ID="m")
        with self.assertRaises(SystemExit):
            run.make_models("1")

    # ------------------------------------------------------------ gigachat

    def giga(self, **extra):
        os.environ.update(MODEL_3_KIND="gigachat", MODEL_3_ID="GigaChat-2",
                          MODEL_3_API_KEY="QVVUSEtFWQ==", MODEL_3_BASE_URL=f"{self.base}/v1",
                          GIGACHAT_AUTH_URL=f"{self.base}/oauth", **extra)
        return run.make_models("3")[0]

    def test_gigachat_token_request_and_cache(self):
        model = self.giga()
        self.assertIsInstance(model, run.GigaChatModel)
        self.assertEqual(model.chat(self.MSG), "ответ GigaChat-2")
        self.assertEqual(model.chat(self.MSG), "ответ GigaChat-2")
        oauth = [r for r in self.srv.log if r["path"] == "/oauth"]
        chats = [r for r in self.srv.log if r["path"] == "/v1/chat/completions"]
        self.assertEqual(len(oauth), 1, "токен должен браться один раз и кэшироваться")
        self.assertEqual(len(chats), 2)
        h = oauth[0]["headers"]
        self.assertEqual(h["Authorization"], "Basic QVVUSEtFWQ==")
        self.assertRegex(h["Rquid"], r"^[0-9a-f-]{36}$")
        self.assertEqual(parse_qs(oauth[0]["body"]), {"scope": ["GIGACHAT_API_PERS"]})
        self.assertEqual(chats[0]["headers"]["Authorization"], "Bearer tok-1")

    def test_gigachat_token_refresh_after_30_min(self):
        model = self.giga()
        now = 1_800_000_000.0
        with mock.patch.object(run.time, "time", return_value=now):
            model.chat(self.MSG)
        with mock.patch.object(run.time, "time", return_value=now + 29 * 60 + 30):
            model.chat(self.MSG)                          # за 30 с до конца — уже обновляем
        self.assertEqual(self.srv.tokens_issued, 2)

    def test_gigachat_expires_at_in_ms(self):
        model = self.giga()
        now = 1_800_000_000.0
        self.srv.expires_at = int((now + 120) * 1000)    # сервер дал токен на 2 минуты, в мс
        with mock.patch.object(run.time, "time", return_value=now):
            model.chat(self.MSG)
        self.assertAlmostEqual(model._token_until, now + 120, delta=1)

    def test_gigachat_401_refreshes_token_once(self):
        model = self.giga()
        model.chat(self.MSG)
        self.srv.reject_tokens = {"Bearer tok-1"}        # сервер отозвал токен раньше срока
        self.assertEqual(model.chat(self.MSG), "ответ GigaChat-2")
        self.assertEqual(self.srv.tokens_issued, 2)

    def test_gigachat_requires_key(self):
        os.environ.update(MODEL_3_KIND="gigachat", MODEL_3_ID="GigaChat-2")
        with self.assertRaises(SystemExit):
            run.make_models("3")

    def test_gigachat_ssl_options(self):
        self.assertEqual(self.giga().ssl_context.verify_mode, ssl.CERT_REQUIRED)
        off = self.giga(GIGACHAT_VERIFY_SSL="0").ssl_context
        self.assertEqual(off.verify_mode, ssl.CERT_NONE)
        self.assertFalse(off.check_hostname)
        os.environ["GIGACHAT_VERIFY_SSL"] = "1"
        with self.assertRaises((FileNotFoundError, OSError)):
            self.giga(GIGACHAT_CA_BUNDLE="нет-такого-файла.crt")

    def test_gigachat_default_endpoints(self):
        os.environ.update(MODEL_3_KIND="gigachat", MODEL_3_ID="GigaChat-2", MODEL_3_API_KEY="x")
        model = run.make_models("3")[0]
        self.assertEqual(model.base_url, "https://api.giga.chat/v1")
        self.assertEqual(model.auth_url, "https://ngw.devices.sberbank.ru:9443/api/v2/oauth")


if __name__ == "__main__":
    unittest.main()
