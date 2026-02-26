# # import json
# # import numpy as np
# # import random
# # from sentence_transformers import SentenceTransformer
# # from sklearn.metrics.pairwise import cosine_similarity

# # # ----------------------------
# # # Load Transformer Model
# # # ----------------------------
# # print("Loading transformer model...")
# # model = SentenceTransformer('all-MiniLM-L6-v2')

# # # ----------------------------
# # # Load Dataset
# # # ----------------------------
# # with open("DataSet.json", "r") as f:
# #     data = json.load(f)

# # sentences = []
# # intent_map = []

# # for intent in data["intents"]:
# #     for pattern in intent["patterns"]:
# #         sentences.append(pattern)
# #         intent_map.append(intent)

# # # ----------------------------
# # # Convert Dataset to Embeddings
# # # ----------------------------
# # print("Encoding dataset...")
# # dataset_embeddings = model.encode(sentences)

# # # ----------------------------
# # # Chat Memory
# # # ----------------------------
# # chat_history = []

# # # ----------------------------
# # # Chatbot Response Function
# # # ----------------------------
# # def chatbot_response(user_input, threshold=0.5, top_k=3):

# #     # Add to memory
# #     chat_history.append(user_input)

# #     # Combine last 3 messages for context awareness
# #     combined_input = " ".join(chat_history[-3:])

# #     # Encode user input
# #     user_embedding = model.encode([combined_input])

# #     # Compute similarity
# #     similarities = cosine_similarity(user_embedding, dataset_embeddings)[0]

# #     # Sort highest similarity
# #     top_indices = similarities.argsort()[::-1][:top_k]

# #     responses = []
# #     used_tags = set()

# #     for idx in top_indices:
# #         score = similarities[idx]

# #         if score >= threshold:
# #             intent = intent_map[idx]
# #             tag = intent["tag"]

# #             if tag not in used_tags:
# #                 reply = random.choice(intent["responses"])
# #                 responses.append(f"{reply} (Confidence: {score:.2f})")
# #                 used_tags.add(tag)

# #     # If nothing matched
# #     if not responses:
# #         log_unknown_query(user_input)
# #         return "I'm not confident about that. Could you please rephrase?"

# #     return " ".join(responses)


# # # ----------------------------
# # # Log Unknown Queries
# # # ----------------------------
# # def log_unknown_query(query):
# #     with open("unknown_queries.txt", "a") as f:
# #         f.write(query + "\n")


# # # ----------------------------
# # # Run Chatbot
# # # ----------------------------
# # print("\n🤖 Advanced Chatbot is running (type 'quit' to stop)\n")

# # while True:
# #     user_input = input("You: ")

# #     if user_input.lower() == "quit":
# #         print("Bot: Goodbye!")
# #         break

# #     reply = chatbot_response(user_input)
# #     print("Bot:", reply)


# import json
# import numpy as np
# import random
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity

# print("Loading transformer model...")
# model = SentenceTransformer('all-MiniLM-L6-v2')

# with open("DataSet.json", "r") as f:
#     data = json.load(f)

# sentences = []
# intent_map = []

# for intent in data["intents"]:
#     for pattern in intent["patterns"]:
#         sentences.append(pattern)
#         intent_map.append(intent)

# print("Encoding dataset...")
# dataset_embeddings = model.encode(sentences)

# # Memory variables
# last_intent = None
# last_confidence = 0

# follow_up_phrases = [
#     # Original
#     "explain more", "more detail", "more details", "elaborate", 
#     "expand", "tell me more", "detailed explanation", "go deeper", 
#     "give me an example", "what does that mean",
    
#     # Newly added to catch your test cases
#     "in detail", "detailed response", "give me in detail", 
#     "give me details", "explain in detail", "tell me exactly", 
#     "give me more", "explain it better", "make it clearer",
#     "can you explain", "how does that work"
# ]

# def chatbot_response(user_input, threshold=0.55):
#     global last_intent, last_confidence
#     lower_input = user_input.lower().strip()

#     # 1. Explicit Follow-Up Check
#     # If the user explicitly asks for more details, expand on the LAST topic.
#     if last_intent and any(phrase in lower_input for phrase in follow_up_phrases):
#         if "detailed_responses" in last_intent:
#             return random.choice(last_intent["detailed_responses"])
#         elif "equation_explanations" in last_intent:
#             return random.choice(last_intent["equation_explanation"])
        
#         else:
#             return random.choice(last_intent["responses"])

#     # 2. Standard Intent Matching
#     user_embedding = model.encode([user_input])
#     similarities = cosine_similarity(user_embedding, dataset_embeddings)[0]
    
#     # Find the best match
#     best_match_idx = np.argmax(similarities)
#     best_score = similarities[best_match_idx]

#     # 3. Decision Logic
#     if best_score >= threshold:
#         matched_intent = intent_map[best_match_idx]
        
#         # If the user is asking about the EXACT same topic again, give them the detailed response
#         if last_intent and matched_intent["tag"] == last_intent["tag"] and best_score < 0.85:
#             if "detailed_responses" in matched_intent:
#                 return random.choice(matched_intent["detailed_responses"])

#         # Otherwise, update the context to the new topic and give the standard response
#         last_intent = matched_intent
#         last_confidence = best_score
#         return random.choice(matched_intent["responses"])

#     # 4. Fallback / Smart Recovery
#     # If the confidence is too low, we ask them to clarify instead of guessing blindly.
#     return "Hmm, I'm not quite sure I understand. Are you asking about how computers learn, or something else?"

# # --- Run Chatbot ---
# print("\nYour educational Explainable chatbot is running.... (type 'quit' to stop)\n")

# while True:
#     user_input = input("You: ")
#     if user_input.lower() == "quit":
#         print("Bot: Bye! Keep learning!")
#         break

#     reply = chatbot_response(user_input)
#     print("Bot:", reply)
import json
import numpy as np
import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Load Model
# ----------------------------
print("Loading transformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------
# Load Dataset (UTF-8 FIX)
# ----------------------------
def load_json_utf8(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return json.load(f)

data = load_json_utf8("DataSet.json")

sentences = []
intent_map = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        sentences.append(pattern)
        intent_map.append(intent)

print("Encoding dataset...")
dataset_embeddings = model.encode(sentences)

# ----------------------------
# Memory Variables
# ----------------------------
last_intent = None
last_confidence = 0.0

# ----------------------------
# Follow-up phrases
# ----------------------------
follow_up_phrases = [
    "explain more", "more detail", "more details", "elaborate",
    "expand", "tell me more", "detailed explaination", "go deeper",
    "give me an example", "what does that mean",
    "in detail", "detailed response", "give me in detail",
    "give me details", "explain in detail", "tell me exactly",
    "give me more", "explain it better", "make it clearer",
    "can you explain", "how does that work"
]

# ----------------------------
# Equation trigger phrases
# ----------------------------
equation_phrases = [
    "equation", "formula", "mathematical form",
    "write equation", "write formula",
    "give equation", "show equation",
    "equation of", "formula of", "mathematical representation of",
    "give me the equation of ", "what's the equation of", "what is the equation of ",
    "give me mathematical equation of ", "can you show the equation of ", "can you give me the formula of "
]

# ----------------------------
# Pretty print equation
# ----------------------------
def format_equation(eq_obj):
    """Format equation dictionary nicely"""
    if isinstance(eq_obj, dict):
        output = f"\nEquation:\n{eq_obj['equation']}\n\nExplaination:\n"
        for term, meaning in eq_obj["terms"].items():
            output += f"- {term}: {meaning}\n"
        return output
    return str(eq_obj)

# ----------------------------
# Chatbot Response Function
# ----------------------------
def chatbot_response(user_input, threshold=0.55):
    global last_intent, last_confidence

    lower_input = user_input.lower().strip()

    # Detect equation intent
    is_equation_request = any(p in lower_input for p in equation_phrases)

    # -------------------------------------------------
    # 1️⃣ Equation follow-up (highest priority)
    # -------------------------------------------------
    if last_intent and is_equation_request:
        if "equation_explaination" in last_intent:
            eq = random.choice(last_intent["equation_explaination"])
            return format_equation(eq)

    # -------------------------------------------------
    # 2️⃣ Explicit follow-up detection
    # -------------------------------------------------
    if (
        last_intent
        and last_confidence >= 0.6
        and any(p in lower_input for p in follow_up_phrases)
    ):
        if "detailed_responses" in last_intent:
            return random.choice(last_intent["detailed_responses"])
        elif "equation_explaination" in last_intent:
            eq = random.choice(last_intent["equation_explaination"])
            return format_equation(eq)
        else:
            return random.choice(last_intent.get("responses", ["Can you clarify?"]))

    # -------------------------------------------------
    # 3️⃣ Standard Intent Matching
    # -------------------------------------------------
    user_embedding = model.encode([user_input])
    similarities = cosine_similarity(user_embedding, dataset_embeddings)[0]

    best_match_idx = np.argmax(similarities)
    best_score = similarities[best_match_idx]

    # -------------------------------------------------
    # 4️⃣ Decision Logic
    # -------------------------------------------------
    if best_score >= threshold:
        matched_intent = intent_map[best_match_idx]

        # Update memory FIRST
        last_intent = matched_intent
        last_confidence = best_score

        # 🔷 If user asked equation directly
        if is_equation_request and "equation_explaination" in matched_intent:
            eq = random.choice(matched_intent["equation_explaination"])
            return format_equation(eq)

        return random.choice(matched_intent.get("responses", ["Okay."]))

    # -------------------------------------------------
    # 5️⃣ Fallback
    # -------------------------------------------------
    return "Hmm, I'm not quite sure I understand. Could you rephrase your question?"

# ----------------------------
# Run Chatbot
# ----------------------------
print("\nYour educational Explainable chatbot is running.... (type 'quit' to stop)\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "quit":
        print("Bot: Bye! Keep learning!")
        break

    reply = chatbot_response(user_input)
    print("Bot:", reply)