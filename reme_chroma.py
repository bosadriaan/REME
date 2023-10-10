import hashlib
from fastapi import FastAPI, HTTPException, Query, Body
from pydantic import BaseModel, Field
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Optional
import json

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


client = chromadb.PersistentClient(path="/content/chroma_db")

# Embedding function setup
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="distiluse-base-multilingual-cased-v1"
)

# Delete the existing collection if it exists

collection_name = "sentences"
try:
    client.delete_collection(name=collection_name)
except ValueError as e:
    print(f"Could not delete collection: {e}")



# Create a collection or get it if it already exists
collection = client.get_or_create_collection(
    name="sentences", 
    embedding_function=sentence_transformer_ef,
    metadata={"hnsw:space": "cosine"}
)

#=============================== 
# Add a document to the collection

class Metadata(BaseModel):
    language: str
    country: str
    user_id: str
    time: str

class DocumentIn(BaseModel):
    sentence: str
    metadata: Metadata

class DocumentOut(DocumentIn):
    document_id: str

 
@app.post("/documents", response_model=DocumentOut)
async def add_document(document: DocumentIn):
    document_id = hashlib.sha256(
        (document.sentence + document.metadata.user_id + document.metadata.time).encode('utf-8')
    ).hexdigest()
    metadata = {
        "language": document.metadata.language,
        "country": document.metadata.country,
        "user_id": document.metadata.user_id,
        "time": document.metadata.time
    }
    collection.add(
        documents=[document.sentence],
        ids=[document_id],
        metadatas=[metadata]
    )
    return {
        "document_id": document_id,
        "sentence": document.sentence,
        "metadata": document.metadata

    }
#================================
#get document from collection with filter

@app.get("/documents")
async def get_documents(
        ids: str = None,
        where: str = None,
        where_document: str = None):
    ids_list = json.loads(ids) if ids else None
    where_dict = json.loads(where) if where else None
    where_document_dict = json.loads(where_document) if where_document else None

    result = collection.get(
        ids=ids_list,
        where=where_dict,
        where_document=where_document_dict
    )
    return result

#================================
# delete a document from collection

@app.delete("/documents")
async def delete_document(document_id: str):
    try:
        collection.delete(ids=[document_id])
        return {"message": "Document deleted successfully"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


#================================
#Query one or more IDs against filtered collection
 
class QueryByIDData(BaseModel):
    document_id: str = Field(..., description="The document ID to use as the basis for the query.")
    n_results: int = Field(4, description="The number of results to return.")
    where: Optional[dict] = Field(None, description="Metadata filter dictionary.")
    where_document: Optional[dict] = Field(None, description="Document content filter dictionary.")
    include: Optional[List[str]] = Field(None, description="Fields to include in the response.")

@app.post("/query_by_id")
async def query_by_id(query_data: QueryByIDData):
    # Get the embedding of the document with the supplied id
    doc_data = collection.get(ids=[query_data.document_id], include=["embeddings"])
    if not doc_data['embeddings']:
        raise HTTPException(status_code=404, detail="Document not found")
    query_embedding = doc_data['embeddings'][0]

    # Perform the query using the retrieved embedding
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=query_data.n_results,
        where=query_data.where,
        where_document=query_data.where_document,
        include=query_data.include
    )
    return result

#================================
# Query collection by sentence

class QueryBySentenceData(BaseModel):
    sentence: str = Field(..., description="The sentence to use as the basis for the query.")
    n_results: int = Field(4, description="The number of results to return.")
    where: Optional[dict] = Field(None, description="Metadata filter dictionary.")
    where_document: Optional[dict] = Field(None, description="Document content filter dictionary.")
    include: Optional[List[str]] = Field(None, description="Fields to include in the response.")

@app.post("/query_by_sentence")
async def query_by_sentence(query_data: QueryBySentenceData):

    # Perform the query using the computed embedding
    print(query_data)
    result = collection.query(
        query_texts= query_data.sentence,
        n_results=query_data.n_results,
        where=query_data.where,
        where_document=query_data.where_document,
        include=query_data.include
    )
    print(result)
    return result


#================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
