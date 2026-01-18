import faiss
from sentence_transformers import SentenceTransformer 
import json
import os

def load_artifacts(artifact_path):
    """Load artifacts from the specified path."""
    index = faiss.read_index(f"{artifact_path}/faiss_index")
    with open(f"{artifact_path}/chunks.json", "r") as f:
        chunks = json.load(f)


    return index, chunks


def create_query_embedding(model, query):
    return model.encode([query])


def retrieve_relevant_chunks(index, query_embedding, chunks, top_k=5, threshold=.25):
    scores, ids = index.search(query_embedding, top_k)
    if scores[0][0] < threshold:
        return []
    return [chunks[j] for j in ids[0]]


def construct_prompt(system_prompt, context_chunks, user_query):
    context_text = "\n".join([chunk['source'] for chunk in context_chunks])
    prompt = (
        f"{system_prompt}\n\n"
        f"Context:\n{context_text}\n\n"
        f"User Query: {user_query}\n"
        "Answer:"
    )
    return prompt


if __name__ == "__main__":

    SYSTEM_PROMPT = (
        'You are a helpful assistant. \n'
        'Answer only using the provided context, if insufficient, say "I don\'t know".\n'
    )
    
    artifact_path = "artifacts"
    user_query = "What is the companies refund policy?"

    # Load artifacts
    index, chunks = load_artifacts(os.path.join(artifact_path))

    # Load embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Create query embedding
    query_embedding = create_query_embedding(model, user_query)

    # Retrieve relevant chunks
    relevant_chunks = retrieve_relevant_chunks(index, query_embedding, chunks, top_k=1)

    # Construct prompt
    prompt = construct_prompt(SYSTEM_PROMPT, relevant_chunks, user_query)

    print(prompt)