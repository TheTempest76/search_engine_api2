from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

index_path = "index/index.bin"
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def embed_chunks(chunks): # create embeddings
    vectors = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    return vectors

def build_faiss_index(vectors):
    # so basically the entire text will be converted to n dimensional tuple containing m dimensional vectors

    #this second parameter is the dimension of each vector, each embedding is same size
    index = faiss.IndexFlatL2(vectors.shape[1])  # L2 
    index.add(vectors)
    return index

def save_faiss_index(index: faiss.IndexFlatL2, path=index_path):
    faiss.write_index(index, path)

def load_faiss_index(path=index_path) -> faiss.IndexFlatL2:
    return faiss.read_index(path)


def search_faiss(query: str, all_chunks: list[str], index: faiss.IndexFlatL2, top_k: int = 5) -> list[str]:
    query_vector = model.encode([query])
    distances, indices = index.search(query_vector, top_k)
    return [all_chunks[i] for i in indices[0]]

