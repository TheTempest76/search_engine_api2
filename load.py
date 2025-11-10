# load.py
import json
import os
from pathlib import Path
from typing import List, Dict
from faiss_embedding import embed_chunks, build_faiss_index, save_faiss_index

# ---------- CONFIG ----------
DATA_PATH = "wiki_dataset.json"      # path to your JSON file
INDEX_DIR = "index"
INDEX_PATH = os.path.join(INDEX_DIR, "index.bin")
CHUNK_SIZE = 800                     # number of words per chunk
OVERLAP = 160                        # overlap between chunks
MIN_CONTENT_LENGTH = 200             # minimum chars to consider valid text


# ---------- HELPERS ----------
def load_json_records(path: str) -> List[Dict]:
    """Load JSON or JSONL file with id, content, metadata."""
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            data = json.load(f)
        else:
            data = [json.loads(line) for line in f if line.strip()]
    valid = [r for r in data if "content" in r and len(r["content"].strip()) > MIN_CONTENT_LENGTH]
    print(f"Loaded {len(valid)} valid records.")
    return valid


def simple_chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Simple whitespace-based chunking with overlap."""
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_size - overlap)
    chunks = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def extract_chunks(records: List[Dict]) -> List[str]:
    """Extract content chunks from all records."""
    all_chunks = []
    for r in records:
        text = r.get("content", "")
        chunks = simple_chunk_text(text, CHUNK_SIZE, OVERLAP)
        all_chunks.extend(chunks)
    print(f"Total chunks created: {len(all_chunks)}")
    return all_chunks


# ---------- MAIN ----------
def main():
    Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)

    records = load_json_records(DATA_PATH)
    all_chunks = extract_chunks(records)

    print("Embedding chunks...")
    vectors = embed_chunks(all_chunks)

    print("Building FAISS index...")
    index = build_faiss_index(vectors)

    print("Saving index...")
    save_faiss_index(index, INDEX_PATH)

    print(f"✅ Index built and saved at {INDEX_PATH}")
    print(f"Total chunks indexed: {len(all_chunks)}")

    # persist the chunk list so the server can serve results without recomputing
    with open(os.path.join(INDEX_DIR, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False)

if __name__ == "__main__":
    main()
