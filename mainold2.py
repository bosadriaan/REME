from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Initialize the FastAPI app
app = FastAPI()

# Initialize the SentenceTransformer model
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')
# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Initialize Faiss index with IDMap wrapper
d = 512  # sentence-transformers/distiluse-base-multilingual-cased-v1
# d = 384  # MiniLM-L6-v2 embedding dimension
base_index = faiss.IndexFlatIP(d)
index = faiss.IndexIDMap2(base_index)

# Initialize ID-to-Vector and ID-to-Sentence mappings
id_to_vector = {}
id_to_sentence = {}

# Pydantic model for the request body
class SentenceRequest(BaseModel):
    unique_id: str
    sentence: str

# Function to generate embeddings
def generate_embedding(sentence):
    embedding = model.encode(sentence)
    return embedding

# Utility function to add a vector to FAISS index and update mappings
def add_to_faiss(unique_id, sentence, id_to_vector, id_to_sentence, index):
    embedding = generate_embedding(sentence)
    id_to_vector[unique_id] = embedding
    id_to_sentence[unique_id] = sentence
    index.add_with_ids(np.array([embedding], dtype=np.float32).reshape(-1, d), np.array([int(unique_id)]))

# Utility function to remove a vector from FAISS index and update mappings
def remove_from_faiss(unique_id):
    id_array = faiss.vector_to_array(index.id_map)
    if unique_id in id_to_vector and int(unique_id) in id_array:
        del id_to_vector[unique_id]
        del id_to_sentence[unique_id]
        index.remove_ids(np.array([int(unique_id)]))


# FastAPI routes
@app.post("/add_sentence/")
async def add_sentence(sentence_request: SentenceRequest):
    unique_id = sentence_request.unique_id
    sentence = sentence_request.sentence
    add_to_faiss(unique_id, sentence, id_to_vector, id_to_sentence, index)
    return {"message": "Sentence added to Faiss", "unique_id": unique_id}

@app.post("/remove_sentence/")
async def remove_sentence(remove_request: SentenceRequest):
    unique_id = remove_request.unique_id
    remove_from_faiss(unique_id)
    return {"message": "Sentence removed from Faiss", "unique_id": unique_id}

@app.post("/get_similar_sentences/")
async def get_similar_sentences(request_body: SentenceRequest):
    unique_id = request_body.unique_id
    if unique_id not in id_to_vector:
        raise HTTPException(status_code=404, detail="Sentence not found")
    query_vector = id_to_vector[unique_id]
    D, I = index.search(np.array([query_vector], dtype=np.float32).reshape(1, -1), k=5)
    similar_sentences = [{"id": str(I[0][i]), "score": float(D[0][i])} for i in range(len(I[0]))]
    return {"similar_sentences": similar_sentences}

@app.get("/get_all_sentences/")
async def get_all_sentences():
    return {"sentences": id_to_sentence}

# Main entry point
if __name__ == "__main__":
    from populate_data import populate_initial_data
    populate_initial_data(id_to_vector, id_to_sentence, index) 
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
