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
import numpy as np
import random
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# 1️⃣ Configuration
# ----------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"
THRESHOLD = 0.40

# --- Streamlit Caching ---
# We cache the model and dataset so Streamlit doesn't reload them on every message!
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def load_dataset():
    try:
        with open("DataSet.json", "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except FileNotFoundError:
        st.error("⚠️ Error: DataSet.json not found. Please make sure the file is in the same folder.")
        st.stop()
        
    sentences, intent_map = [], []
    for intent in data.get("intents", []):
        for pattern in intent.get("patterns", []):
            sentences.append(pattern)
            intent_map.append(intent)
    return sentences, intent_map

model = load_model()
sentences, intent_map = load_dataset()

@st.cache_resource
def get_embeddings(_model, _sentences):
    return _model.encode(_sentences)

dataset_embeddings = get_embeddings(model, sentences)

# ----------------------------
# 2️⃣ Constants & Phrases
# ----------------------------
ELI5_PHRASES = ["eli5", "explain like i'm 5", "explain like i am 5", "like i'm five", "explain simply", "simple explanation", "very simple"]
DETAIL_PHRASES = ["in detail", "detailed", "deep explanation", "technical explanation", "explain in detail", "tell me exactly", "go deeper", "explain me in detail", "give me a detailed response", "more detail", "explain more", "elaborate"]
equation_phrases = ["equation", "formula", "mathematical form", "write equation", "write formula", "give equation", "show equation", "equation of", "formula of"]

# ----------------------------
# 3️⃣ Helper Functions
# ----------------------------
def call_ollama(prompt: str) -> str:
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        return f"⚠️ Local AI model not responding. Ensure Ollama is running! Error: {e}"

def format_equation(eq_obj):
    if isinstance(eq_obj, dict):
        output = f"\n**Equation:**\n{eq_obj['equation']}\n\n**Explanation:**\n"
        for term, meaning in eq_obj["terms"].items():
            output += f"- **{term}**: {meaning}\n"
        return output
    return str(eq_obj)

def build_system_prompt(text: str):
    base = (
        "You are an expert educational tutor specializing STRICTLY in Artificial Intelligence, "
        "Machine Learning, and Data Science. You MUST assume every single question or acronym is about Machine Learning. "
        "For example, 'ML' ALWAYS means Machine Learning. Always answer the USER'S QUESTION directly. "
    )
    if any(p in text for p in ELI5_PHRASES):
        return base + "Explain in very simple terms like the user is 5 years old. Use easy, relatable, non-technical examples (like toys, pets, or food)."
    if any(p in text for p in DETAIL_PHRASES):
        return base + "Provide a thorough, step-by-step, and structured explanation of the ML concept. Go into deep detail, but KEEP IT SIMPLE. Use plain everyday English and always explain complex concepts using real-life analogies."
    return base + "Give a clear, easy-to-understand explanation suitable for a beginner student. Avoid overly complex jargon."

# ----------------------------
# 4️⃣ Core Chatbot Logic
# ----------------------------
def get_bot_response(user_input: str) -> str:
    user_input_lower = user_input.lower().strip()
    
    is_equation_request = any(p in user_input_lower for p in equation_phrases)
    is_detail_follow_up = any(p in user_input_lower for p in DETAIL_PHRASES) and len(user_input_lower.split()) <= 15
    is_equation_follow_up = is_equation_request and len(user_input_lower.split()) <= 12

    # A. Handle Pure Follow-ups using Streamlit Memory
    if is_equation_follow_up and st.session_state.last_intent:
        eq_key = "equation_explaination" if "equation_explaination" in st.session_state.last_intent else "equation_explanation"
        if eq_key in st.session_state.last_intent:
            return format_equation(random.choice(st.session_state.last_intent[eq_key]))
        else:
            prompt = f"Provide the mathematical equation and explain its terms clearly.\n\nContext: We were discussing '{st.session_state.last_meaningful_query}'.\nUser request: {user_input}\nAnswer:"
            return call_ollama(prompt)

    if is_detail_follow_up and st.session_state.last_meaningful_query:
        sys_prompt = build_system_prompt(user_input_lower)
        prompt = f"{sys_prompt}\n\nContext: We were discussing '{st.session_state.last_meaningful_query}'.\nUser request: {user_input}\nAnswer:"
        return call_ollama(prompt)

    # Update memory for the next round
    st.session_state.last_meaningful_query = user_input

    # B. Standard Intent Matching (Dataset)
    user_embedding = model.encode([user_input])
    similarities = cosine_similarity(user_embedding, dataset_embeddings)[0]
    best_match_idx = np.argmax(similarities)
    best_score = similarities[best_match_idx]

    if best_score >= THRESHOLD:
        matched_intent = intent_map[best_match_idx]
        st.session_state.last_intent = matched_intent

        if is_equation_request:
            eq_key = "equation_explaination" if "equation_explaination" in matched_intent else "equation_explanation"
            if eq_key in matched_intent:
                return format_equation(random.choice(matched_intent[eq_key]))

        return random.choice(matched_intent.get("responses", ["Okay."]))

    # C. Smart Fallback (Ollama)
    sys_prompt = build_system_prompt(user_input_lower)
    prompt = f"{sys_prompt}\n\nUser question: {user_input}\nAnswer:"
    return call_ollama(prompt)

# ----------------------------
# 5️⃣ Streamlit UI Setup
# ----------------------------
st.set_page_config(page_title="ELI5 ML Tutor", page_icon="🤖")
st.title("🤖 ELI5 Machine Learning Bot")
st.markdown("Ask me anything about how computers learn!")

# Initialize Memory
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your Machine Learning tutor. What would you like to learn today?"}]
if "last_intent" not in st.session_state:
    st.session_state.last_intent = None
if "last_meaningful_query" not in st.session_state:
    st.session_state.last_meaningful_query = ""

# --- The Dynamic Roadmap Sidebar ---
with st.sidebar:
    st.header("Few quick topics to ask in Machine Learning:")
    st.write("Click a topic below to ask the bot about it:")
    
    # These buttons will instantly trigger a message!
    topics = [
        "What is Machine Learning?", 
        "Explain Linear Regression", 
        "What is a Neural Network?", 
        "Explain Overfitting", 
        "What is Supervised Learning?",
        "What is SVM"
    ]
    for topic in topics:
        if st.button(topic):
            st.session_state.sidebar_prompt = topic

# Draw all past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Check if input came from the sidebar buttons OR the text box
prompt = st.chat_input("Ask a machine learning question...")
if "sidebar_prompt" in st.session_state:
    prompt = st.session_state.sidebar_prompt
    del st.session_state.sidebar_prompt

# Process the input and fix the UI lag
if prompt:
    # Immediately draw user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Show loading spinner while model runs
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_bot_response(prompt)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})