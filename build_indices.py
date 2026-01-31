import os
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
import json
import argparse
from bs4 import BeautifulSoup
import trafilatura

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
    for path in Path(root_path).rglob('*'):
        if path.suffix in ['.txt', '.html']:
            with open(path, 'r', encoding='utf-8') as file:
                content = file.read()
                if path.suffix.lower() == '.html':
                    content = trafilatura.extract(content, output_format='markdown')

                if content:
                    docs.append((content, str(path)))
    return docs

def parse_args():
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

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.artifacts_path, exist_ok=True)
    model = SentenceTransformer(args.embedding_model)
    docs = load_docs(args.data_path)

    all_chunks = []
    metadata = []

    for i, doc in enumerate(docs):
        chunk_idx = 0
        chunks = chunker(doc[0])
        all_chunks.extend(chunks)

        for chunk in chunks:
            metadata.append({
                'source': doc[1],
                'doc_idx': i,
                'chunk_idx': chunk_idx
            })
            chunk_idx +=1
    
    print(f'Total chunks created: {len(all_chunks)}')

    emb = model.encode(all_chunks, batch_size=32, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True)
    print(emb.shape)
    dim = emb.shape[1]
    print(f'Embedding shape: {emb.shape}')

    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    faiss.write_index(index, os.path.join(args.artifacts_path, 'faiss_index'))

    outputs = {
        'chunks': all_chunks,
        'metadata': metadata,
        'embedding_model': args.embedding_model
    }

    with open(os.path.join(args.artifacts_path, 'chunks.json'), 'w', encoding='utf-8') as f:
        json.dump(outputs, f, ensure_ascii=False, indent=4)

    print(f'Saved FAISS index and metadata to {args.artifacts_path}')

if __name__ == '__main__':
    main()