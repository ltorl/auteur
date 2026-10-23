"""A stand-in for openai-oauth: OpenAI-compatible, scripted director replies."""

from __future__ import annotations

import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

TREATMENT = {
    "title": "The Last Lamplighter",
    "logline": "In a city gone electric, one old lamplighter lights his final street.",
    "synopsis": "Elias walks his route at dusk. The gas lamps are being replaced. He lights the last one and "
    "a child watches the flame bloom.",
    "genre": "drama",
    "tone": "bittersweet",
    "style": {
        "visual_style": "1950s Kodachrome realism",
        "palette": "amber, teal shadows",
        "lighting": "dusk, practical flames",
        "camera_language": "slow, observational",
        "texture": "35mm grain, soft halation",
        "sound": "quiet street ambience, hiss of gas, sparse piano",
    },
    "characters": [
        {"id": "Elias", "name": "Elias", "role": "lamplighter",
         "look": "70s, tall, grey beard, navy wool coat, brass pole", "voice": "low, gentle"},
        {"id": "nell", "name": "Nell", "role": "child", "look": "8, red scarf, curious eyes", "voice": "bright"},
    ],
    "locations": [{"id": "street", "name": "Cobbled street", "look": "wet cobbles, iron lamps"}],
}


def shot(i, chars, transition="cut", camera="slow push in"):
    return {
        "id": f"S{i:02d}", "scene": 1, "title": f"Beat {i}", "duration": 3.0, "shot_size": "MS",
        "camera": camera, "lens": "35mm", "characters": chars, "location": "street",
        "action": "Elias raises the pole.", "dialogue": [{"speaker": "Elias", "line": "One more."}] if i == 1 else [],
        "sound": "gas hiss", "keyframe_prompt": f"Elias at lamp {i}", "transition": transition,
        "motion_prompt": f"Elias lifts the brass pole toward lamp {i}. The camera slowly pushes in. "
        "Gas hisses softly. Elias says, \"One more.\"",
    }


SHOTS = {"shots": [shot(1, ["elias"]), shot(2, ["nell"], "dissolve", "pan left"), shot(3, ["elias", "nell"])]}


def png(color=(200, 120, 40), size=(64, 40)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class Handler(BaseHTTPRequestHandler):
    server_version = "fake-openai-oauth"

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            ids = ["gpt-5.4-mini", "gpt-5.6-sol", "gpt-5.6-codex", "gpt-image-2"]
            return self._json({"object": "list", "data": [{"id": i, "object": "model", "owned_by": "x"} for i in ids]})
        self._json({"error": "nope"}, 404)

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        raw = self.rfile.read(n)
        self.server.calls.append((self.path, raw if b"chat" in self.path.encode() else raw[:2000]))
        if self.path.endswith("/images/generations") or self.path.endswith("/images/edits"):
            return self._json({"created": 0, "data": [{"b64_json": png()}]})
        if self.path.endswith("/chat/completions"):
            req = json.loads(raw)
            user = req["messages"][-1]["content"]
            if "Develop a treatment" in user:
                reply = TREATMENT
            elif "Break this treatment" in user:
                self.server.plans += 1
                # first plan attempt is malformed to exercise the repair loop
                reply = {"shots": "oops"} if self.server.plans == 1 else SHOTS
            elif "Their note" in user:
                s = json.loads(json.dumps(SHOTS))
                s["shots"][1]["camera"] = "static"
                s["shots"][1]["motion_prompt"] = "Nell watches, perfectly still."
                reply = {"reply": "Locked off S02 — stillness sells her awe.", "treatment": None, "shots": s["shots"]}
            elif "invalid" in user:
                reply = SHOTS
            else:
                reply = {"reply": "ok"}
            return self._stream(req["model"], json.dumps(reply))
        self._json({"error": "nope"}, 404)

    def _stream(self, model, text):
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        for i in range(0, len(text), 50):
            chunk = {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": model,
                     "choices": [{"index": 0, "delta": {"content": text[i:i + 50]}, "finish_reason": None}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        end = {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": model,
               "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        self.wfile.write(f"data: {json.dumps(end)}\n\ndata: [DONE]\n\n".encode())


def serve():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    srv.calls, srv.plans = [], 0
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/v1"
