import streamlit as st
import json
import numpy as np
import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# 1. Load and Cache the Model
# =====================================================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# =====================================================
# 2. Load and Cache the Dataset (UTF-8 SAFE)
# =====================================================
@st.cache_data
def load_data():
    with open("DataSet.json", "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)

    sentences = []
    intent_map = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            sentences.append(pattern)
            intent_map.append(intent)

    embeddings = model.encode(sentences)
    return intent_map, embeddings

intent_map, dataset_embeddings = load_data()

# =====================================================
# 3. Session Memory
# =====================================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello there! I'm here to talk about how computers learn. What do you want to know?",
        }
    ]

if "last_intent" not in st.session_state:
    st.session_state.last_intent = None

if "last_confidence" not in st.session_state:
    st.session_state.last_confidence = 0.0

# =====================================================
# 4. Phrase Lists
# =====================================================
follow_up_phrases = [
    "explain more", "more detail", "more details", "elaborate",
    "expand", "tell me more", "detailed explanation", "go deeper",
    "give me an example", "what does that mean",
    "in detail", "detailed response", "give me in detail",
    "give me details", "explain in detail", "tell me exactly",
    "give me more", "explain it better", "make it clearer",
    "can you explain", "how does that work",
    "explain a bit more", "can you elaborate",
    "can you expand on that", "can you give me an example",
    "can you explain that in more detail",
    "can you explain that better", "can you explain that clearer",
]

equation_phrases = [
    "equation", "formula", "mathematical form",
    "write equation", "write formula",
    "give equation", "show equation",
    "equation of", "formula of", "mathematical representation of",
    "give me the equation of ", "what's the equation of", "what is the equation of ",
    "give me mathematical equation of ", "can you show the equation of ", "can you give me the formula of "
]

# =====================================================
# 5. Equation Formatter
# =====================================================
def format_equation(eq_obj):
    if isinstance(eq_obj, dict):
        output = f"**Equation:**\n\n{eq_obj['equation']}\n\n**Explaination:**\n"
        for term, meaning in eq_obj["terms"].items():
            output += f"- **{term}**: {meaning}\n"
        return output
    return str(eq_obj)

# =====================================================
# 6. Chatbot Brain
# =====================================================
def get_bot_response(user_input, threshold=0.55):
    lower_input = user_input.lower().strip()
    last_intent = st.session_state.last_intent

    # Detect equation intent
    is_equation_request = any(p in lower_input for p in equation_phrases)

    # -------------------------------------------------
    # 1️⃣ Very short follow-up (e.g., "give equation")
    # -------------------------------------------------
    if last_intent and is_equation_request and len(lower_input.split()) <= 3:
        if "equation_explaination" in last_intent:
            eq = random.choice(last_intent["equation_explaination"])
            return format_equation(eq)

    # -------------------------------------------------
    # 2️⃣ Standard Intent Matching
    # -------------------------------------------------
    user_embedding = model.encode([user_input])
    similarities = cosine_similarity(user_embedding, dataset_embeddings)[0]

    best_match_idx = np.argmax(similarities)
    best_score = similarities[best_match_idx]

    # -------------------------------------------------
    # 3️⃣ Confident match
    # -------------------------------------------------
    if best_score >= threshold:
        matched_intent = intent_map[best_match_idx]

        # Update memory FIRST
        st.session_state.last_intent = matched_intent
        st.session_state.last_confidence = best_score

        # 🔥 Equation override
        if is_equation_request and "equation_explaination" in matched_intent:
            eq = random.choice(matched_intent["equation_explaination"])
            return format_equation(eq)

        # Same-topic follow-up → detailed
        if (
            last_intent
            and matched_intent["tag"] == last_intent["tag"]
            and best_score < 0.85
        ):
            if "detailed_responses" in matched_intent:
                return random.choice(matched_intent["detailed_responses"])

        return random.choice(matched_intent.get("responses", ["Okay."]))

    # -------------------------------------------------
    # 4️⃣ Relaxed follow-up handling
    # -------------------------------------------------
    if last_intent:
        # equation follow-up even if similarity low
        if is_equation_request and "equation_explaination" in last_intent:
            eq = random.choice(last_intent["equation_explaination"])
            return format_equation(eq)

        # normal follow-up
        if any(p in lower_input for p in follow_up_phrases):
            if "detailed_responses" in last_intent:
                return random.choice(last_intent["detailed_responses"])

    # -------------------------------------------------
    # 5️⃣ Fallback
    # -------------------------------------------------
    return "Hmm, I'm not quite sure I understand. Are you asking about how computers learn, or something else?"

# =====================================================
# 7. Streamlit UI
# =====================================================
st.title("🤖 ELI5 Machine Learning Bot")
st.markdown("Ask me anything about how computers learn!")

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box
if prompt := st.chat_input("Type your question here..."):

    # user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # bot response
    response = get_bot_response(prompt)

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)