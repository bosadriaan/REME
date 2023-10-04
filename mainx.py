import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BertTokenizer, BertModel
import faiss
import numpy as np

# Initialize the FastAPI app
app = FastAPI()

# Initialize the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Initialize Faiss index
d = 768  # BERT embedding dimension
index = faiss.IndexFlatIP(d)

# Initialize ID-to-Vector and ID-to-Sentence mappings
id_to_vector = {}
id_to_sentence = {}
id_list = []


# Pydantic model for the request body
class SentenceRequest(BaseModel):
    unique_id: str
    sentence: str

# Function to generate BERT embeddings
def generate_bert_embedding(sentence):
    tokens = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model(**tokens)
    embedding = output.last_hidden_state.mean(dim=1).numpy()
    return embedding

# Populate initial data
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
        unique_id = str(i + 100)
        embedding = generate_bert_embedding(sentence)
        id_to_vector[unique_id] = embedding
        id_to_sentence[unique_id] = sentence
        id_list.append(unique_id)
        index.add(np.array([embedding], dtype=np.float32).reshape(-1, d))

        # print(i, unique_id, id_list[i], id_to_sentence[unique_id])

# Endpoint to add a sentence and its embedding
@app.post("/add_sentence/")
async def add_sentence(sentence_request: SentenceRequest):
    unique_id = sentence_request.unique_id
    sentence = sentence_request.sentence
    embedding = generate_bert_embedding(sentence)
    id_to_vector[unique_id] = embedding
    id_to_sentence[unique_id] = sentence
    id_list.append(unique_id)
    index.add(np.array([embedding], dtype=np.float32).reshape(-1, d))


    return {"message": "Sentence added to Faiss", "unique_id": unique_id}


# Endpoint to retrieve the most similar sentences
# Endpoint to retrieve the most similar sentences
@app.post("/get_similar_sentences/")
async def get_similar_sentences(request_body: SentenceRequest):
    unique_id = request_body.unique_id
    if unique_id not in id_to_vector:
        raise HTTPException(status_code=404, detail="Sentence not found")
    
    # Get the index of the unique ID in id_list
    try:
        faiss_index = id_list.index(unique_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Unique ID not found in id_list")
    
    # Extract the embedding directly from Faiss index
    embedding = index.reconstruct(faiss_index)
    
    # Perform the search
    query_vector = np.array([embedding], dtype=np.float32).reshape(1, -1)
    D, I = index.search(query_vector, k=5)
    
    # Build the response
    similar_sentences = [{"id": id_list[i], "score": float(D[0][i])} for i in I[0]]

    return {"similar_sentences": similar_sentences}




# Endpoint to return all stored sentences and their IDs
@app.get("/get_all_sentences/")
async def get_all_sentences():
    return {"sentences": id_to_sentence}

if __name__ == "__main__":
    populate_initial_data()
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
