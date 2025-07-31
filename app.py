from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os
from dotenv import load_dotenv
import pandas as pd
import faiss
import pickle
import numpy as np


load_dotenv()

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Load FAISS Index and Tool Metadata ===
faiss_index = faiss.read_index("tool_index.faiss")

with open("tool_metadata.pkl", "rb") as f:
    tool_metadata = pickle.load(f)

# === Embed function ===
def embed_text(text):
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

# === Search FAISS and get top-k tool descriptions ===
def search_similar_tools(query, top_k=3):
    query_vector = embed_text(query)
    query_vector = np.array(query_vector, dtype="float32").reshape(1, -1)
    distances, indices = faiss_index.search(query_vector, top_k)    
    top_tools = tool_metadata.iloc[indices[0]].to_dict(orient="records")
    return top_tools

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    prompt = data.get("prompt")

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    try:
        relevant_tools = search_similar_tools(prompt)

        tool_context = "\n\n".join([
                f"Name: {tool['Name']}\n"
                f"Description: {tool['Short Description']}\n"
                f"Categories: {tool.get('Categories', 'N/A')}\n"
                f"Pricing: {tool.get('Pricing', 'N/A')}\n"
                f"Link: {tool.get('Homepage', 'N/A')}"
                for tool in relevant_tools
        ])

        full_prompt = (
            "You are an expert AI assistant that helps users discover the best AI tools for their specific needs.\n\n"
            f"User prompt: {prompt}\n\n"
            "Below are some tools that may be relevant:\n"
            f"{tool_context}\n\n"
            "Your job is to:\n"
            "- Recommend the best tool(s) based on the user's need.\n"
            "- Use the context above when possible, highlighting each tool’s category and pricing.\n"
            "- Provide a direct, clickable link (e.g., https://...) to the tool’s homepage using Markdown.\n"
            "- If no strong match is found, feel free to suggest a general-purpose AI tool like ChatGPT, Claude, or Gemini.\n"
            "- Be clear, concise, and helpful — like you're giving advice to a friend who wants to save time.\n\n"
            "Make sure your recommendation feels thoughtful and trustworthy. End with an encouraging tone so the user feels confident trying the tool."
        )

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.7
        )

        return jsonify({"response": response.choices[0].message.content})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
