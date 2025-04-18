#Sample Front end code using Flask for UI
from flask import Flask, render_template, request, jsonify
from qdrant_client import QdrantClient
from transformers import AutoTokenizer, AutoModel
import torch

app = Flask(__name__)

client = QdrantClient(
    url="https://cecd63ca-9699-4bc0-a60f-3c7e8c768fd3.europe-west3-0.gcp.cloud.qdrant.io:6333",
    api_key="jwv0ETSb8751Q1yrQ3RVSsChFudsv1M63mEvxXXRnLf_ROoeAfe_Wg"
)
collection_name = "LawEmbedding"
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().astype('float32').flatten().tolist()

@app.route('/')
def index():
    return render_template('index.html')  

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Get embedding
    embedding = get_embedding(text)

    # Search Qdrant collections
    search_result_cosine = client.search(
        collection_name=collection_name + 'COSINE',
        query_vector=embedding,
        limit=5
    )

    # Format results
    results = [{
        "id": result.id,
        "url": result.payload['url'],
        "section": result.payload['section'],
        "score": result.score
    } for result in search_result_cosine]

    return jsonify(results), 200

if __name__ == '__main__':
    app.run(debug=True)
