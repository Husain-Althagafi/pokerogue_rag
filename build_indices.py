import os
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
import json

def chunker(text: str, chunk_size: int = 350, overlap: int = 50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]
        chunks.append(' '.join(chunk))
        if i + chunk_size >= len(words):
            break
    return chunks

def load_docs(root_path: str = 'data/docs'):
    docs = []
    for path in Path(root_path).rglob('*.txt'):
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read()
            docs.append(content)
    return docs

def main():
    os.makedirs('artifacts', exist_ok=True)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    docs = load_docs()

    all_chunks = []
    metadata = []

    for i, doc in enumerate(docs):
        chunks = chunker(doc)
        all_chunks.extend(chunks)
        metadata.extend([{'source': doc, 'index': i}] * len(chunks))
    
    print(f'Total chunks created: {len(all_chunks)}')

    emb = model.encode(all_chunks, batch_size=32, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True)

    dim = emb.shape[1]
    print(f'Embedding shape: {emb.shape}')

    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    faiss.write_index(index, 'artifacts/faiss_index')
    with open('artifacts/chunks.json', 'w') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    print(f'Saved FAISS index and metadata to artifacts/')

if __name__ == '__main__':
    main()
