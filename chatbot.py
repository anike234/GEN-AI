import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Load pretrained transformer
# ----------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# ----------------------------
# Load custom dataset
# ----------------------------
with open("intents.json", "r") as f:
    data = json.load(f)

# ----------------------------
# Prepare dataset texts
# ----------------------------
sentences = []
intent_map = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        sentences.append(pattern)
        intent_map.append(intent)

# ----------------------------
# Convert dataset to vectors
# ----------------------------
dataset_embeddings = model.encode(sentences)

# ----------------------------
# Chatbot response function
# ----------------------------
def chatbot_response(user_input, threshold=0.6, top_k=2):
    user_embedding = model.encode([user_input])

    similarities = cosine_similarity(user_embedding, dataset_embeddings)[0]

    # Get top matches
    top_indices = similarities.argsort()[::-1][:top_k]

    responses = []
    used_tags = set()

    for idx in top_indices:
        score = similarities[idx]
        if score >= threshold:
            intent = intent_map[idx]
            tag = intent["tag"]

            if tag not in used_tags:
                responses.append(intent["responses"][0])
                used_tags.add(tag)

    if not responses:
        return "Sorry, I couldn't understand your question."

    return " ".join(responses)

# ----------------------------
# Run chatbot
# ----------------------------
print("🤖 Chatbot is running (type 'quit' to stop)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    reply = chatbot_response(user_input)
    print("Bot:", reply)