import os
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
import json
import argparse

def chunker(text: str, chunk_size: int = 350, overlap: int = 50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]
        chunks.append(' '.join(chunk))
        if i + chunk_size >= len(words):
            break
    return chunks

def load_docs(root_path: str):
    docs = []
    for path in Path(root_path).rglob('*.txt'):
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read()
            docs.append(content)
    return docs

def main():
    parser = argparse.ArgumentParser(
        description='Script for building indices from data for rag',
    )

    parser.add_argument(
        '--data_path',
        default='data/pokerogue_pages'
    )

    parser.add_argument(
        '--artifacts_path',
        default='artifacts'
    )

    parser.add_argument(
        '--embedding_model',
        default='all-MiniLM-L6-v2'
    )

    args = parser.parse_args()

    os.makedirs(args.artifacts_path, exist_ok=True)
    model = SentenceTransformer(args.embedding_model)
    docs = load_docs(args.data_path)

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

    faiss.write_index(index, os.path.join(args.artifacts_path, 'faiss_index'))
    with open(os.path.join(args.artifacts_path, 'chunks.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    print(f'Saved FAISS index and metadata to {args.artifacts_path}')

if __name__ == '__main__':
    # docs = load_docs('data/pokerogue_pages')
    # print(len(docs))
    main()