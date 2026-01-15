from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from rag_query import construct_prompt, load_artifacts, create_query_embedding, retrieve_relevant_chunks
import faiss
from sentence_transformers import SentenceTransformer 

def load_model(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype="auto",
    )
    return model, tokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chat with a rag model yay')

    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-1.5B-Instruct', help='Name of the model to load')

    args = parser.parse_args()

    SYSTEM_PROMPT = (
        'You are a helpful assistant. \n'
        'Answer only using the provided context, if insufficient, say "I don\'t know".\n'
    )

    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    index, chunks = load_artifacts('artifacts')

    print(f'Hi there, this is a chat model for some random stuff rn based on some rag thingy, if you have any questions that the database can answer, feel free to ask!')

    prompt = input('Enter your prompt: ')
    while prompt.lower() != 'exit':
        prompt = input('Enter your prompt: ')
        query_embedding = create_query_embedding(embedding_model, prompt)
        relevant_chunks = retrieve_relevant_chunks(index, query_embedding, chunks, top_k=1)

        FULL_PROMPT = construct_prompt(SYSTEM_PROMPT, relevant_chunks, prompt)

        print('LOADING PLEASE BE PATIENT PLEASEEEEEEE...')
        model, tokenizer = load_model(model_name=args.model_name)

        inputs = tokenizer([FULL_PROMPT], return_tensors='pt')
        gen_ids = model.generate(**inputs, max_new_tokens=512)

        output_ids = gen_ids[:, inputs.input_ids.shape[-1]:]

        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f'Bot: {response}')


        print(f'Context used: {relevant_chunks[0]["source"]}')

