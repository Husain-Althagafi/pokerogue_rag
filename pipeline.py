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


def build_maxlen_prompt(system_prompt, context, query, tokenizer, max_length=1024):
    prefix = f'SYSTEM: {system_prompt}\n\n\nCONTEXT:\n'
    suffix = f'\n\n\nUSER QUERY: {query}\n\n\nASSISTANT ANSWER:\n'

    prefix_ids = tokenizer(prefix)['input_ids']
    suffix_ids = tokenizer(suffix)['input_ids']

    fixed_len = len(prefix_ids) + len(suffix_ids)

    if fixed_len >= max_length:
        raise Exception('system prompt + user query too long, max tokens has been reached before a query could be added.')

    chunks_to_keep = []
    used = 0
    for chunk in context:
        chunk_ids = tokenizer(chunk['source'])['input_ids']
        combined_len = len(chunk_ids) + fixed_len + used

        if combined_len <= max_length:
            chunks_to_keep.append(chunk['source'])
            used += len(chunk_ids)

        else:
            remainder = max_length - (fixed_len + used)
            if remainder > 0:
                 chunk_ids = chunk_ids[:remainder]
                 chunk_remainder = tokenizer.decode(chunk_ids, skip_special_tokens=True)
                 chunks_to_keep.append(chunk_remainder)
            break
    context = f'\n\n'.join(chunks_to_keep)
    full_prompt = f'{prefix}{context}{suffix}'
    return full_prompt

def main():
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(description='Chat with a rag model yay')

    parser.add_argument(
        '--model_name',
        type=str, 
        default='Qwen/Qwen2.5-1.5B-Instruct',
        help='Name of the model to load'
    )

    parser.add_argument(
        '--encoding_model',
        type=str,
        default='all-MiniLM-L6-v2'
    )

    parser.add_argument(
        '--artifacts_path',
        type=str,
        default='artifacts'
    )

    parser.add_argument(
        '--testing',
        action='store_true'
    )

    parser.add_argument(
          '--device',
          type=str,
          default='cuda'
    )

    parser.add_argument(
         '--max_len',
         type=int,
         default=1024
    )

    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
                raise Exception('Cuda is not available but is selected as the device, either select a different device or fix cuda availability issue.')

    SYSTEM_PROMPT = (
        'You are a helpful assistant. Answer only using the provided context, if insufficient, say "I don\'t know". Make sure to respond to the users query, only using the context as needed based on the query. The context may be empty.'
    )

    embedding_model = SentenceTransformer(args.encoding_model, device='cpu')
    index, chunks = load_artifacts(os.path.join(args.artifacts_path))

    model, tokenizer = load_model(model_name=args.model_name)
    model.eval()

    print(f'Hi there, this is a chat model for some random stuff rn based on some rag thingy, if you have any questions that the database can answer, feel free to ask!')

    prompt = None
    while not prompt or prompt.lower() != 'exit':
        if not prompt:
            prompt = input('Enter your prompt: ')

        query_embedding = create_query_embedding(embedding_model, prompt)
        relevant_chunks = retrieve_relevant_chunks(index, query_embedding, chunks, top_k=3)

        FULL_PROMPT = build_maxlen_prompt(SYSTEM_PROMPT, relevant_chunks, prompt, tokenizer=tokenizer, max_length=args.max_len)

        print('\n\n\nLOADING...\n\n\n')

        inputs = tokenizer([FULL_PROMPT], return_tensors='pt', truncation=True, max_length=args.max_len).to(args.device)

        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_new_tokens=512, temperature)

        output_ids = gen_ids[:, inputs.input_ids.shape[-1]:]

        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f'Response: {response}')

        if args.testing:
            print(f'\n\n\nFull prompt used:\n\n\n{FULL_PROMPT}\n\n\nResponse: {response}')

        prompt = input('\n\n\nEnter your prompt: ')
     

if __name__ == '__main__':
    main()
    