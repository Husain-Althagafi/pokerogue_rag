from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from rag_query import construct_prompt, load_artifacts, create_query_embedding, retrieve_relevant_chunks
import faiss
from sentence_transformers import SentenceTransformer 
import os
import torch

def load_model(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype=torch.float16,
    )
    return model, tokenizer
    
if __name__ == '__main__':

    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(description='Chat with a rag model yay')

    parser.add_argument(
        '--model_name',
        type=str, default='Qwen/Qwen2.5-1.5B-Instruct',
        help='Name of the model to load'
        )

    parser.add_argument(
        '--encoding_model',
        default='all-MiniLM-L6-v2'
    )

    parser.add_argument(
        '--artifacts_path',
        default='artifacts'
    )

    parser.add_argument(
        '--testing',
        action='store_true'
    )

    parser.add_argument(
          '--device',
          default='cuda'
    )

    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
                raise Exception('Cuda is not available but is selected as the device, either select a different device or fix cuda availability issue.')

    SYSTEM_PROMPT = (
        'You are a helpful assistant. \n'
        'Answer only using the provided context, if insufficient, say "I don\'t know".\n'
    )

    embedding_model = SentenceTransformer(args.encoding_model, device='cpu')
    index, chunks = load_artifacts(os.path.join(args.artifacts_path))

    model, tokenizer = load_model(model_name=args.model_name)
    model.eval()
    print(f'Loaded model')

    print(f'Hi there, this is a chat model for some random stuff rn based on some rag thingy, if you have any questions that the database can answer, feel free to ask!')

    prompt = None
    while not prompt or prompt.lower() != 'exit':
        if not prompt:
            prompt = input('Enter your prompt: ')

        query_embedding = create_query_embedding(embedding_model, prompt)
        relevant_chunks = retrieve_relevant_chunks(index, query_embedding, chunks, top_k=1)

        FULL_PROMPT = construct_prompt(SYSTEM_PROMPT, relevant_chunks, prompt)

        print('LOADING...')

        # print(f'len full prompt: {len(FULL_PROMPT)}')
        # print(f'num of relevant chunks: {len(relevant_chunks)}')
        # print(f'length of first chuck: {len(relevant_chunks[0]['source'])}')

        inputs = tokenizer([FULL_PROMPT], return_tensors='pt', truncation=True, max_length=1024).to(args.device)
        print(f'decoded inputs: {tokenizer.decode(inputs['input_ids'][0])}')
        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_new_tokens=256)
            print(f'generated')

        output_ids = gen_ids[:, inputs.input_ids.shape[-1]:]

        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f'Bot: {response}')

        if args.testing:
            print(f'Context used: {relevant_chunks[0]["source"][:20]}')

        prompt = input('Enter your prompt: ')
        

