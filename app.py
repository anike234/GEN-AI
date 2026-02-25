import streamlit as st
import json
import numpy as np
import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. Load and Cache the Model ---
# This prevents the transformer from reloading on every single chat message
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# --- 2. Load and Cache the Dataset ---
@st.cache_data
def load_data():
    with open("DataSet.json", "r") as f:
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

# --- 3. Initialize Session State (Memory) ---
# This replaces the global variables so the bot remembers the context
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello there! I'm here to talk about how computers learn. What do you want to know?"}
    ]
if "last_intent" not in st.session_state:
    st.session_state.last_intent = None
if "last_confidence" not in st.session_state:
    st.session_state.last_confidence = 0

# Expanded follow-up list to catch variations
follow_up_phrases = [
    "explain more", "more detail", "more details", "elaborate", 
    "expand", "tell me more", "detailed explanation", "go deeper", 
    "give me an example", "what does that mean",
    "in detail", "detailed response", "give me in detail", 
    "give me details", "explain in detail", "tell me exactly", 
    "give me more", "explain it better", "make it clearer",
    "can you explain", "how does that work","explain a bit more", "can you elaborate", "can you expand on that",
    "can you give me an example", "can you explain that in more detail", "can you explain that better", "can you explain that clearer"

]

# --- 4. Chatbot Response Logic ---
def get_bot_response(user_input, threshold=0.55):
    lower_input = user_input.lower().strip()
    last_intent = st.session_state.last_intent
    
    # Check for explicit follow-up phrases
    if last_intent and any(phrase in lower_input for phrase in follow_up_phrases):
        if "detailed_responses" in last_intent:
            return random.choice(last_intent["detailed_responses"])
        else:
            return random.choice(last_intent["responses"])

    # Standard Intent Matching
    user_embedding = model.encode([user_input])
    similarities = cosine_similarity(user_embedding, dataset_embeddings)[0]
    
    best_match_idx = np.argmax(similarities)
    best_score = similarities[best_match_idx]

    if best_score >= threshold:
        matched_intent = intent_map[best_match_idx]
        
        # If asking about the same topic again, provide the detailed response
        if last_intent and matched_intent["tag"] == last_intent["tag"] and best_score < 0.85:
            if "detailed_responses" in matched_intent:
                return random.choice(matched_intent["detailed_responses"])

        # Update session memory
        st.session_state.last_intent = matched_intent
        st.session_state.last_confidence = best_score
        return random.choice(matched_intent["responses"])

    return "Hmm, I'm not quite sure I understand. Are you asking about how computers learn, or something else?"

# --- 5. Streamlit User Interface ---
st.title("🤖 ELI5 Machine Learning Bot")
st.markdown("Ask me anything about how computers learn!")

# Display previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input box at the bottom
if prompt := st.chat_input("Type your question here..."):
    
    # 1. Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # 2. Get bot response
    response = get_bot_response(prompt)
    
    # 3. Add bot response to chat history and display it
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)