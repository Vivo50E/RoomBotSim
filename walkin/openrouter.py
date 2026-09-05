"""LLM / VLM client. One function: ask_json(model, prompt, images) -> dict.

Works against OpenRouter by default and any OpenAI-compatible /chat/completions endpoint
(DeepSeek, vLLM, Ollama, Together, ...). Configure with:
  LLM_BASE_URL  (default https://openrouter.ai/api/v1)   -> POST {base}/chat/completions
  LLM_API_KEY   (falls back to OPENROUTER_API_KEY)
  VLM_MODEL     (default qwen/qwen3.7-flash)             perception model (vision)
  BRAIN_MODEL   (default = VLM_MODEL)                    chat model for the crowd brain + command parser
"""
import os, json, base64, time, logging, requests

log = logging.getLogger("llm")


def base_url():
    return os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")


def api_key():
    return os.environ.get("LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or ""


def vlm_model():
    return os.environ.get("VLM_MODEL", "qwen/qwen3.7-flash")


def brain_model():
    return os.environ.get("BRAIN_MODEL") or vlm_model()


def has_key():
    return bool(api_key())


def _headers():
    return {"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000", "X-Title": "walk-in"}


def image_part(jpeg_path):
    b64 = base64.b64encode(open(jpeg_path, "rb").read()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


def text_part(s):
    return {"type": "text", "text": s}


def chat(model, messages, max_tokens=1500, temperature=0.0, json_mode=True, timeout=45):
    if not has_key():
        raise RuntimeError("no LLM_API_KEY / OPENROUTER_API_KEY set")
    url = base_url() + "/chat/completions"
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    if "openrouter.ai" in url:
        body["reasoning"] = {"enabled": False}          # VERIFY: OpenRouter unified reasoning param
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    last = None
    for attempt in range(3):
        try:
            r = requests.post(url, headers=_headers(), json=body, timeout=timeout)
        except requests.RequestException as e:
            last = str(e); time.sleep(1.0 * (attempt + 1)); continue
        if r.status_code == 400 and attempt == 0:
            body.pop("reasoning", None); body.pop("response_format", None)
            c = messages[-1]["content"]
            if isinstance(c, list) and c and c[0].get("type") == "text":
                c[0]["text"] = "Answer immediately with the JSON only, no thinking.\n\n" + c[0]["text"]
            elif isinstance(c, str):
                messages[-1]["content"] = "Answer immediately with the JSON only, no thinking.\n\n" + c
            continue
        if r.status_code in (429, 500, 502, 503):
            time.sleep(1.5 * (attempt + 1)); last = r.text[:300]; continue
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        content = (msg.get("content") or "").strip()
        if not content:
            last = "empty content"; continue
        return content
    raise RuntimeError(f"llm failed: {last}")


def parse_json(text):
    t = text.strip()
    fence = chr(96) * 3
    if t.startswith(fence):
        t = t.split("\n", 1)[1] if "\n" in t else t
        t = t.rsplit(fence, 1)[0]
    s, e = t.find("{"), t.rfind("}")
    if s < 0 or e < 0:
        raise ValueError("no json object in response")
    return json.loads(t[s:e + 1])


def ask_json(model, prompt, images=(), system=None, **kw):
    """prompt: str. images: list of jpeg paths. Returns parsed dict. Raises on failure."""
    content = [text_part(prompt)] + [image_part(p) for p in images]
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
    raw = chat(model, messages, **kw)
    try:
        return parse_json(raw)
    except Exception:
        log.warning("json parse failed, raw=%r", raw[:500]); raise
