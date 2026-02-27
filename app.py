# import streamlit as st
# import json
# import numpy as np
# import random
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity

# # =====================================================
# # 1. Load and Cache the Model
# # =====================================================
# @st.cache_resource
# def load_model():
#     return SentenceTransformer("all-MiniLM-L6-v2")

# model = load_model()

# # =====================================================
# # 2. Load and Cache the Dataset (UTF-8 SAFE)
# # =====================================================
# @st.cache_data
# def load_data():
#     with open("DataSet.json", "r", encoding="utf-8", errors="replace") as f:
#         data = json.load(f)

#     sentences = []
#     intent_map = []

#     for intent in data["intents"]:
#         for pattern in intent["patterns"]:
#             sentences.append(pattern)
#             intent_map.append(intent)

#     embeddings = model.encode(sentences)
#     return intent_map, embeddings

# intent_map, dataset_embeddings = load_data()

# # =====================================================
# # 3. Session Memory
# # =====================================================
# if "messages" not in st.session_state:
#     st.session_state.messages = [
#         {
#             "role": "assistant",
#             "content": "Hello there! I'm here to talk about how computers learn. What do you want to know?",
#         }
#     ]

# if "last_intent" not in st.session_state:
#     st.session_state.last_intent = None

# if "last_confidence" not in st.session_state:
#     st.session_state.last_confidence = 0.0

# # =====================================================
# # 4. Phrase Lists
# # =====================================================
# follow_up_phrases = [
#     "explain more", "more detail", "more details", "elaborate",
#     "expand", "tell me more", "detailed explanation", "go deeper",
#     "give me an example", "what does that mean",
#     "in detail", "detailed response", "give me in detail",
#     "give me details", "explain in detail", "tell me exactly",
#     "give me more", "explain it better", "make it clearer",
#     "can you explain", "how does that work",
#     "explain a bit more", "can you elaborate",
#     "can you expand on that", "can you give me an example",
#     "can you explain that in more detail",
#     "can you explain that better", "can you explain that clearer",
# ]

# equation_phrases = [
#     "equation", "formula", "mathematical form",
#     "write equation", "write formula",
#     "give equation", "show equation",
#     "equation of", "formula of", "mathematical representation of",
#     "give me the equation of ", "what's the equation of", "what is the equation of ",
#     "give me mathematical equation of ", "can you show the equation of ", "can you give me the formula of "
# ]

# # =====================================================
# # 5. Equation Formatter
# # =====================================================
# def format_equation(eq_obj):
#     if isinstance(eq_obj, dict):
#         output = f"**Equation:**\n\n{eq_obj['equation']}\n\n**Explaination:**\n"
#         for term, meaning in eq_obj["terms"].items():
#             output += f"- **{term}**: {meaning}\n"
#         return output
#     return str(eq_obj)

# # =====================================================
# # 6. Chatbot Brain
# # =====================================================
# def get_bot_response(user_input, threshold=0.55):
#     lower_input = user_input.lower().strip()
#     last_intent = st.session_state.last_intent

#     # Detect equation intent
#     is_equation_request = any(p in lower_input for p in equation_phrases)

#     # -------------------------------------------------
#     # 1️⃣ Very short follow-up (e.g., "give equation")
#     # -------------------------------------------------
#     if last_intent and is_equation_request and len(lower_input.split()) <= 3:
#         if "equation_explaination" in last_intent:
#             eq = random.choice(last_intent["equation_explaination"])
#             return format_equation(eq)

#     # -------------------------------------------------
#     # 2️⃣ Standard Intent Matching
#     # -------------------------------------------------
#     user_embedding = model.encode([user_input])
#     similarities = cosine_similarity(user_embedding, dataset_embeddings)[0]

#     best_match_idx = np.argmax(similarities)
#     best_score = similarities[best_match_idx]

#     # -------------------------------------------------
#     # 3️⃣ Confident match
#     # -------------------------------------------------
#     if best_score >= threshold:
#         matched_intent = intent_map[best_match_idx]

#         # Update memory FIRST
#         st.session_state.last_intent = matched_intent
#         st.session_state.last_confidence = best_score

#         # 🔥 Equation override
#         if is_equation_request and "equation_explaination" in matched_intent:
#             eq = random.choice(matched_intent["equation_explaination"])
#             return format_equation(eq)

#         # Same-topic follow-up → detailed
#         if (
#             last_intent
#             and matched_intent["tag"] == last_intent["tag"]
#             and best_score < 0.85
#         ):
#             if "detailed_responses" in matched_intent:
#                 return random.choice(matched_intent["detailed_responses"])

#         return random.choice(matched_intent.get("responses", ["Okay."]))

#     # -------------------------------------------------
#     # 4️⃣ Relaxed follow-up handling
#     # -------------------------------------------------
#     if last_intent:
#         # equation follow-up even if similarity low
#         if is_equation_request and "equation_explaination" in last_intent:
#             eq = random.choice(last_intent["equation_explaination"])
#             return format_equation(eq)

#         # normal follow-up
#         if any(p in lower_input for p in follow_up_phrases):
#             if "detailed_responses" in last_intent:
#                 return random.choice(last_intent["detailed_responses"])

#     # -------------------------------------------------
#     # 5️⃣ Fallback
#     # -------------------------------------------------
#     return "Hmm, I'm not quite sure I understand. Are you asking about how computers learn, or something else?"

# # =====================================================
# # 7. Streamlit UI
# # =====================================================
# st.title("🤖 ELI5 Machine Learning Bot")
# st.markdown("Ask me anything about how computers learn!")

# # Display history
# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])

# # Input box
# if prompt := st.chat_input("Type your question here..."):

#     # user message
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     # bot response
#     response = get_bot_response(prompt)

#     st.session_state.messages.append({"role": "assistant", "content": response})
#     with st.chat_message("assistant"):
#         st.markdown(response)
# import streamlit as st
# import requests

# # -----------------------------
# # CONFIG
# # -----------------------------
# OLLAMA_URL = "http://localhost:11434/api/generate"
# MODEL_NAME = "llama3"

# ELI5_PHRASES = [
#     "eli5", "explain like i'm 5", "explain like i am 5",
#     "simple explanation", "in simple terms"
# ]

# DETAIL_PHRASES = [
#     "explain more", "more detail", "in detail",
#     "elaborate", "deep dive"
# ]

# # -----------------------------
# # MODE DETECTION
# # -----------------------------
# def detect_mode(user_input: str):
#     text = user_input.lower()

#     if any(p in text for p in ELI5_PHRASES):
#         return "eli5"
#     if any(p in text for p in DETAIL_PHRASES):
#         return "detail"
#     return "normal"


# # -----------------------------
# # SYSTEM PROMPT
# # -----------------------------
# def build_system_prompt(mode: str):
#     if mode == "eli5":
#         return (
#             "Explain in very simple terms like teaching a 5-year-old. "
#             "Use short sentences and easy examples."
#         )
#     elif mode == "detail":
#         return (
#             "Give a detailed but clear explanation with examples. "
#             "Keep it structured and educational."
#         )
#     else:
#         return (
#             "You are a helpful educational assistant. "
#             "Give clear and correct explanations."
#         )


# # -----------------------------
# # OLLAMA CALL (WITH CONTEXT)
# # -----------------------------
# def query_ollama(messages, mode):
#     system_prompt = build_system_prompt(mode)

#     payload = {
#         "model": MODEL_NAME,
#         "prompt": system_prompt + "\n\n" + messages,
#         "stream": False
#     }

#     response = requests.post(OLLAMA_URL, json=payload)

#     if response.status_code == 200:
#         return response.json()["response"]
#     else:
#         return "⚠️ Error talking to Ollama."


# # -----------------------------
# # STREAMLIT UI
# # -----------------------------
# st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
# st.title("🤖 Ollama Chatbot")

# # Session memory
# if "history" not in st.session_state:
#     st.session_state.history = []

# # Chat input
# user_input = st.chat_input("Ask something...")

# if user_input:
#     # Save user message
#     st.session_state.history.append(("user", user_input))

#     # Detect mode
#     mode = detect_mode(user_input)

#     # Build conversation context
#     conversation_text = ""
#     for role, msg in st.session_state.history:
#         if role == "user":
#             conversation_text += f"User: {msg}\n"
#         else:
#             conversation_text += f"Assistant: {msg}\n"

#     # Get response
#     bot_reply = query_ollama(conversation_text, mode)

#     # Save bot reply
#     st.session_state.history.append(("assistant", bot_reply))

# # -----------------------------
# # DISPLAY CHAT
# # -----------------------------
# for role, msg in st.session_state.history:
#     with st.chat_message(role):
#         st.write(msg)
import streamlit as st
import json
import random
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# ⚙️ CONFIG
# =====================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"
THRESHOLD = 0.55

# =====================================================
# 🚀 LOAD MODEL (cached for speed)
# =====================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_embedding_model()

# =====================================================
# 📚 LOAD DATASET
# =====================================================

@st.cache_resource
def load_dataset():
    with open("DataSet.json", "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)

    sentences = []
    intent_map = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            sentences.append(pattern)
            intent_map.append(intent)

    embeddings = model.encode(sentences)
    return sentences, intent_map, embeddings

sentences, intent_map, dataset_embeddings = load_dataset()

# =====================================================
# 🧠 SESSION MEMORY
# =====================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "last_intent" not in st.session_state:
    st.session_state.last_intent = None

if "last_confidence" not in st.session_state:
    st.session_state.last_confidence = 0.0

# =====================================================
# 🔍 PHRASES (MERGED FOLLOW-UP + ELI5)
# =====================================================

EXPLAIN_MORE_PHRASES = [
    "explain more", "more detail", "more details", "elaborate",
    "expand", "tell me more", "go deeper", "give me an example",
    "in detail", "give me details", "explain in detail",
    "make it clearer", "how does that work",
    "eli5", "explain like i'm 5", "explain like i am 5",
    "simple explanation", "explain simply"
]

EQUATION_PHRASES = [
    "equation", "formula", "mathematical form",
    "write equation", "write formula",
    "give equation", "show equation"
]

# =====================================================
# 🧾 HELPERS
# =====================================================

def format_equation(eq_obj):
    if isinstance(eq_obj, dict):
        output = f"\n**Equation:**\n{eq_obj['equation']}\n\n**Explanation:**\n"
        for term, meaning in eq_obj["terms"].items():
            output += f"- {term}: {meaning}\n"
        return output
    return str(eq_obj)


def ask_ollama(prompt: str, mode: str = "normal"):
    """Call local llama3 via Ollama"""

    if mode == "eli5":
        system_prompt = "Explain in very simple terms like teaching a 5 year old."
    elif mode == "detail":
        system_prompt = "Explain in detailed technical depth."
    else:
        system_prompt = "Answer clearly and correctly."

    payload = {
        "model": MODEL_NAME,
        "prompt": system_prompt + "\n\nUser: " + prompt,
        "stream": False,
    }

    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
        if r.status_code == 200:
            return r.json()["response"]
        return "⚠️ Ollama error"
    except Exception as e:
        return f"⚠️ Ollama connection failed: {e}"


# =====================================================
# 🤖 MAIN ROUTER (HYBRID BRAIN)
# =====================================================

def chatbot_response(user_input: str):
    lower_input = user_input.lower().strip()

    is_explain_more = any(p in lower_input for p in EXPLAIN_MORE_PHRASES)
    is_equation_request = any(p in lower_input for p in EQUATION_PHRASES)

    # -------------------------------------------------
    # 1️⃣ Equation follow-up
    # -------------------------------------------------
    if st.session_state.last_intent and is_equation_request:
        intent = st.session_state.last_intent

        if "equation_explaination" in intent:
            eq = random.choice(intent["equation_explaination"])
            return format_equation(eq)

    # -------------------------------------------------
    # 2️⃣ Follow-up / ELI5 → use Ollama on SAME topic
    # -------------------------------------------------
    if is_explain_more:
        if st.session_state.last_intent:
            topic = st.session_state.last_intent.get("tag", "")
            if topic:
                return ask_ollama(f"Explain {topic}", mode="eli5")
        return ask_ollama(user_input, mode="eli5")

    # -------------------------------------------------
    # 3️⃣ DATASET SEMANTIC MATCH
    # -------------------------------------------------
    user_embedding = model.encode([user_input])
    similarities = cosine_similarity(user_embedding, dataset_embeddings)[0]

    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]

    # -------------------------------------------------
    # 4️⃣ If confident → USE DATASET
    # -------------------------------------------------
    if best_score >= THRESHOLD:
        matched_intent = intent_map[best_idx]

        st.session_state.last_intent = matched_intent
        st.session_state.last_confidence = float(best_score)

        if is_equation_request and "equation_explaination" in matched_intent:
            eq = random.choice(matched_intent["equation_explaination"])
            return format_equation(eq)

        return random.choice(matched_intent.get("responses", ["Okay."]))

    # -------------------------------------------------
    # 5️⃣ Otherwise → Ollama fallback
    # -------------------------------------------------
    return ask_ollama(user_input, mode="detail")


# =====================================================
# 🎨 STREAMLIT UI
# =====================================================

st.title("🧠 Educational Hybrid Chatbot")

# display chat history
for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.markdown(msg)

# user input
user_input = st.chat_input("Ask something...")

if user_input:
    st.session_state.history.append(("user", user_input))

    with st.chat_message("assistant"):
        reply = chatbot_response(user_input)
        st.markdown(reply)

    st.session_state.history.append(("assistant", reply))