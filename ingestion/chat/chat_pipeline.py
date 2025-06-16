import time
from transformers import pipeline
import redis
import torch
import httpx

### ─── PHASE 1: SPAM FILTERING ───────────────────────────────────────
spam_filter = pipeline(
    "text-classification",
    model="cja5553/xlm-roberta-Twitter-spam-classification",
    device=0 if torch.cuda.is_available() else -1
)

def is_spam(message: str, threshold: float = 0.8) -> bool:
    r = spam_filter(message)[0]
    return r["label"].lower() == "spam" and r["score"] > threshold


### ─── PHASE 2: SENTIMENT-BASED SUMMARY ─────────────────────────────
OLLAMA_URL = "http://127.0.0.1:11434/v1/completions"

# Number of recent messages to include in each summary window
top_k_msgs = 20

def generate_summary(messages: list[str], timeout: float = 30.0) -> str:
    if not messages:
        return "No messages to summarize."

    # Build a prompt listing the recent chat messages and asking for overall sentiment
    joined = "\n".join(f"- {m}" for m in messages)
    prompt = (
        "Here are the most recent chat messages:\n" +
        f"{joined}\n\n" +
        "Based on these messages, summarize the overall sentiment or feeling in one concise sentence."
    )

    payload = {
        "model": "mistral",
        "prompt": prompt,
        "n": 1,
        "max_tokens": 32,
        "temperature": 0.5
    }
    try:
        resp = httpx.post(OLLAMA_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get("choices", [])[0].get("text", "").strip() or "[Empty response]"
    except Exception as e:
        return f"[Error] Summarization failed: {e}"


### ─── REDIS FOR MESSAGE BUFFER ─────────────────────────────────────
r = redis.Redis(host="localhost", port=6379, db=0)
redis_key = "chat:recent"

# Push each non-spam message into a capped Redis list
def store_message(message: str, timestamp: float) -> None:
    if is_spam(message):
        return
    # use a Redis list for FIFO
    r.lpush(redis_key, f"{int(timestamp)}|{message}")
    r.ltrim(redis_key, 0, top_k_msgs - 1)

# Retrieve the list of recent messages (sorted oldest→newest)
def get_recent_messages() -> list[str]:
    raw = r.lrange(redis_key, 0, top_k_msgs - 1)
    msgs = []
    for entry in raw[::-1]:  # reverse to oldest first
        try:
            _, msg = entry.decode().split("|", 1)
            msgs.append(msg)
        except Exception:
            continue
    return msgs


if __name__ == "__main__":
    import sys
    print("Streaming chat. Type one line per message:")
    last = time.time()
    interval = 10  # seconds between summaries

    while True:
        line = sys.stdin.readline().strip()
        if not line:
            continue
        now = time.time()
        store_message(line, now)

        if now - last >= interval:
            recent = get_recent_messages()
            summary = generate_summary(recent)
            print("\n*** LIVE SUMMARY ***")
            print(summary)
            print()
            last = now
