from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import sqlite3
import requests
import json
import uuid
import os
import logging
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("CHAT_DB_PATH", "chat.db")
OLLAMA_CHAT_URL = os.getenv("OLLAMA_URL", "http://localhost:11434") + "/api/chat"

logging.basicConfig(
    format='{"ts":"%(asctime)s","level":"%(levelname)s","component":"chat_server","message":"%(message)s"}',
    level=logging.INFO,
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger("chat_server")

SYSTEM_PROMPT = """You are an expert software engineer. You write production-quality code and respond like a senior colleague, not a tutorial author.

Rules:
- Write complete, runnable code. Modern idioms: Python (type hints, pathlib, f-strings); JS (const/let, async/await).
- If the request has real ambiguity, ask ONE clarifying question before writing code. Don't ask about trivia.
- Comments only where intent is non-obvious. Never restate what the code does in English after the code block.
- No "In this function we..." summaries. No "This will output..." explanations. The code speaks for itself.
- No filler ("great question", "I hope this helps", "certainly!", "let me know if...").
- When debugging, identify the actual cause before suggesting fixes.

Example of the response style I want:

USER: Write a function that checks if a string is a palindrome.
ASSISTANT: One ambiguity: case-sensitive and whitespace-sensitive, or normalized? Assuming normalized (typical interpretation).

```python
def is_palindrome(s: str) -> bool:
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]
```

End of example. Match that density."""


# ---------- Database ----------

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
        """)

def create_session(title: str = "New chat") -> str:
    session_id = str(uuid.uuid4())
    with get_db() as conn:
        conn.execute("INSERT INTO sessions (id, title) VALUES (?, ?)", (session_id, title))
    return session_id

def list_sessions():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]

def delete_session(session_id: str):
    with get_db() as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

def rename_session(session_id: str, title: str):
    with get_db() as conn:
        conn.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, session_id))

def touch_session(session_id: str):
    with get_db() as conn:
        conn.execute(
            "UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (session_id,),
        )

def session_message_count(session_id: str) -> int:
    with get_db() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    return row["c"] if row else 0

def save_message(session_id: str, role: str, content: str):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )

def load_messages(session_id: str):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


# ---------- Title generation ----------

def generate_title(model: str, user_msg: str, assistant_msg: str) -> str:
    """Ask the LLM for a short 3-6 word title summarizing the conversation."""
    prompt = (
        "Summarize the following exchange as a very short title (3-6 words, no quotes, no punctuation at the end). "
        "Just output the title, nothing else.\n\n"
        f"User: {user_msg}\n\nAssistant: {assistant_msg[:500]}"
    )
    try:
        res = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=30,
        )
        res.raise_for_status()
        data = res.json()
        title = data.get("message", {}).get("content", "").strip()
        # Clean up: strip quotes, trailing punctuation, limit length
        title = title.strip('"\'.,! ').split("\n")[0]
        return title[:60] if title else "New chat"
    except Exception:
        return user_msg.strip().replace("\n", " ")[:50] or "New chat"


# ---------- App ----------

init_db()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Serve frontend ----------
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

@app.get("/")
def serve_index():
    return FileResponse("index.html")

app.mount("/static", StaticFiles(directory="."), name="static")


class ChatRequest(BaseModel):
    session_id: str
    messages: list
    model: str = "qwen2.5-coder:14b"


@app.post("/sessions")
def new_session():
    sid = create_session()
    logger.info(f"session created: {sid}")
    return {"session_id": sid}


@app.get("/sessions")
def get_sessions():
    return list_sessions()


@app.delete("/sessions/{session_id}")
def remove_session(session_id: str):
    delete_session(session_id)
    logger.info(f"session deleted: {session_id}")
    return {"ok": True}


@app.get("/sessions/{session_id}/messages")
def get_session_messages(session_id: str):
    return load_messages(session_id)


@app.post("/chat")
def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    user_message = req.messages[-1]["content"]
    is_first_message = session_message_count(req.session_id) == 0
    logger.info(f"chat session={req.session_id} model={req.model} msg_len={len(user_message)}")

    messages = req.messages
    if messages[0].get("role") != "system":
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    def generate():
        full_reply = ""
        try:
            with requests.post(
                OLLAMA_CHAT_URL,
                json={"model": req.model, "messages": messages, "stream": True},
                stream=True,
            ) as res:
                res.raise_for_status()
                for line in res.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line.decode("utf-8"))
                    content = data.get("message", {}).get("content", "")
                    if content:
                        full_reply += content
                        yield content
                    if data.get("done"):
                        break
        except Exception as e:
            logger.error(f"stream error session={req.session_id}: {e}")
            yield f"\n[Error: {str(e)}]"
            return

        save_message(req.session_id, "user", user_message)
        save_message(req.session_id, "assistant", full_reply)
        touch_session(req.session_id)

        # Generate smart title from first exchange
        if is_first_message and full_reply:
            title = generate_title(req.model, user_message, full_reply)
            rename_session(req.session_id, title)

    return StreamingResponse(generate(), media_type="text/plain")