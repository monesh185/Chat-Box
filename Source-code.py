# --------------------------- 0. Dependencies ---------------------------
import sys
import json
import threading
from dataclasses import dataclass
from typing import List, Tuple

try:
    from sentence_transformers import SentenceTransformer, util
    import torch
    from flask import Flask, request, jsonify
except ImportError:
    sys.exit(
        "❌  Missing deps. Install once with:\n"
        "    pip install sentence-transformers flask torch --quiet"
    )

# --------------------------- 1. FAQ “dataset” ---------------------------
FAQ: List[Tuple[str, str]] = [
    (
        "How do I reset my password?",
        "Click **Forgot password** on the sign-in page. We’ll email you a secure link.",
    ),
    (
        "Where can I track my order?",
        "Open **Your Account → Orders** to see real-time shipping status.",
    ),
    (
        "My package arrived damaged.",
        "So sorry! Choose **Problem with order → Damaged item** and we’ll ship a free replacement.",
    ),
    (
        "How long is the warranty?",
        "All products carry a **12-month warranty** from the delivery date.",
    ),
    (
        "I need a copy of my invoice.",
        "In the order details page, click **Download invoice (PDF)**.",
    ),
    (
        "How do I reach human support?",
        "Email support@example.com or call **+1-800-555-1234** (24 × 7).",
    ),
]

# --------------------------- 2. Training phase ---------------------------
@dataclass
class KB:
    model: SentenceTransformer
    q_texts: List[str]
    a_texts: List[str]
    q_embs: "torch.Tensor"


def build_kb(pairs: List[Tuple[str, str]]) -> KB:
    print("🛠️  Encoding FAQ questions …")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    q, a = zip(*pairs)
    q_embs = model.encode(list(q), convert_to_tensor=True, show_progress_bar=False)
    return KB(model=model, q_texts=list(q), a_texts=list(a), q_embs=q_embs)


kb = build_kb(FAQ)

# --------------------------- 3. Query helper ---------------------------
def answer(query: str) -> str:
    q_vec = kb.model.encode(query, convert_to_tensor=True)
    sims = util.cos_sim(q_vec, kb.q_embs)[0]
    best_idx = int(torch.argmax(sims))
    if sims[best_idx] < 0.25:
        return (
            "Sorry, I’m not sure about that. "
            "Could you re-phrase or email support@example.com?"
        )
    return kb.a_texts[best_idx]


# --------------------------- 4. Optional Flask API ---------------------------
app = Flask(__name__)


@app.route("/chat", methods=["POST"])
def chat_route():
    data = request.get_json(force=True)
    msg = data.get("msg", "")
    return jsonify({"answer": answer(msg)})


def run_flask():
    app.run(port=8000, threaded=True)


# --------------------------- 5. Command-line REPL ---------------------------
def repl():
    print("\n🤖  SupportBot ready!  Ask a question or type 'quit'\n")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            user = "quit"
        if user.lower() in {"quit", "exit"}:
            print("Bot: Bye! 👋")
            break
        print("Bot:", answer(user), "\n")


# --------------------------- 6. Main entry ---------------------------
if __name__ == "__main__":
    # Launch Flask in a background thread so you can cURL while chatting
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    repl()
