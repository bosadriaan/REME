from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json


# Initialize the FastAPI app
app = FastAPI()

# Initialize the SentenceTransformer model
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')

# Initialize Faiss index with IDMap wrapper
d = 512
base_index = faiss.IndexFlatIP(d)
index = faiss.IndexIDMap2(base_index)

# Initialize ID-to-Vector and ID-to-Sentence mappings
id_to_vector = {}
id_to_sentence = {}

# Function to generate embeddings
def generate_embedding(sentence):
    embedding = model.encode(sentence)
    return embedding

def add_to_faiss(unique_id, sentence):
    embedding = generate_embedding(sentence)
    id_to_vector[unique_id] = embedding
    id_to_sentence[unique_id] = sentence
    index.add_with_ids(np.array([embedding], dtype=np.float32).reshape(-1, d), np.array([int(unique_id)]))

def remove_from_faiss(unique_id):
    id_array = faiss.vector_to_array(index.id_map)
    if unique_id in id_to_vector and int(unique_id) in id_array:
        del id_to_vector[unique_id]
        del id_to_sentence[unique_id]
        index.remove_ids(np.array([int(unique_id)]))



@app.post("/add_sentence/")
async def add_sentence(unique_id: str, sentence: str):
    add_to_faiss(unique_id, sentence)
    save_faiss_to_disk()
    return {"message": "Sentence added to Faiss", "unique_id": unique_id}

@app.post("/remove_sentence/")
async def remove_sentence(unique_id: str):
    remove_from_faiss(unique_id)
    save_faiss_to_disk()
    return {"message": "Sentence removed from Faiss", "unique_id": unique_id}


@app.get("/get_similar_sentences/")
async def get_similar_sentences(unique_id: str):
    if unique_id not in id_to_vector:
        raise HTTPException(status_code=404, detail="Sentence not found")
    query_vector = id_to_vector[unique_id]
    D, I = index.search(np.array([query_vector], dtype=np.float32).reshape(1, -1), k=5)
    similar_sentences = [{"id": str(I[0][i]), "score": float(D[0][i])} for i in range(len(I[0]))]
    return {"similar_sentences": similar_sentences}

@app.get("/get_sentence/")
async def get_sentence(unique_id: str):
    return {"sentence": id_to_sentence.get(unique_id, "Sentence not found")}

@app.get("/get_all_sentences/")
async def get_all_sentences():
    return {"sentences": id_to_sentence}


def save_faiss_to_disk():
    faiss.write_index(index, "faiss_index.faiss")
    with open("id_to_vector.json", "w") as f:
        json.dump({k: v.tolist() for k, v in id_to_vector.items()}, f)
    with open("id_to_sentence.json", "w") as f:
        json.dump(id_to_sentence, f)

def load_faiss_from_disk():
    global index, id_to_vector, id_to_sentence
    index = faiss.read_index("faiss_index.faiss")
    with open("id_to_vector.json", "r") as f:
        id_to_vector = {k: np.array(v) for k, v in json.load(f).items()}
    with open("id_to_sentence.json", "r") as f:
        id_to_sentence = json.load(f)

def populate_initial_data():
    sentences = [
        "A man at the grocery store wearing black pants and a red shirt.",
        "A girl on her bike in the park with a white dog and a child.",
        "An elderly woman with a cane standing near the bus stop.",
        "A teenager skateboarding down the street with headphones on.",
        "A child flying a kite in the open field, wearing a yellow cap.",
        "A couple jogging together along the river in matching outfits.",
        "A businessman in a suit talking on his phone, looking hurried.",
        "A musician playing the guitar on the sidewalk for tips.",
        "A group of kids playing soccer in the park.",
        "A woman pushing a stroller while talking on her phone.",
        "A man reading a newspaper on a bench with a coffee by his side.",
        "A woman selling flowers at a small roadside stall.",
        "A police officer directing traffic at a busy intersection.",
        "A food vendor making hot dogs at his cart.",
        "A cyclist riding against traffic, wearing a bright orange vest.",
        "A painter on a ladder, working on a mural on the side of a building.",
        "A mail carrier delivering packages, pushing a cart.",
        "A student with a heavy backpack waiting for the school bus.",
        "A dog walker managing several dogs of different sizes.",
        "A person handing out flyers for a local event."
    ]
    for i, sentence in enumerate(sentences):
        unique_id = str(i + 1)
        add_to_faiss(unique_id, sentence)    

# Main entry point
if __name__ == "__main__":
    try:
        load_faiss_from_disk()
    except RuntimeError:
        populate_initial_data()
        save_faiss_to_disk()
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)

