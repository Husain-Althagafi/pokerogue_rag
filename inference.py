from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import argparse
from rag_query import construct_prompt, load_artifacts, create_query_embedding, retrieve_relevant_chunks
import faiss
from sentence_transformers import SentenceTransformer 
import os
import torch
from model import load_model, load_reranking_model
from utils import build_maxlen_prompt

def parse_args():
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

    parser.add_arguement(
         '--reranker',
         type=str,
         default='BAAI/bge-reranker-v2-m3'
    )

    parser.add_arguement(
         '--do_reranking',
         action='store_true'
    )

    return parser.parse_args()


def reranking(query, chunks, model_name, top_ranks=7, batch_size=4):
    reranker, tokenizer = load_reranking_model(model_name=model_name)
    reranker.eval()

    pairs = [[query, chunk] for chunk in chunks]

    reranking_scores = []

    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i:batch_size + i]

        input_tokens = tokenizer(batch_pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        reranking_scores.append(reranker(**input_tokens, return_dict=True).logits.view(-1).float())

    reranking_scores = torch.cat(reranking_scores, dim=0).tolist()
    reranked = sorted(
         reranking_scores,
         key=lambda x: x[1],
         reverse=True
    )
    return reranked[:top_ranks]
     

def main():
    torch.cuda.empty_cache()

    args = parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
                raise Exception('Cuda is not available but is selected as the device, either select a different device or fix cuda availability issue.')

    SYSTEM_PROMPT = (
        'You are a helpful assistant. Answer only using the provided context, if insufficient, say "I don\'t know". If the query is something that can be responded to without requiring information from the context, then go ahead and answer it as you see fit.'
    )

    embedding_model = SentenceTransformer(args.encoding_model, device='cpu')
    index, chunks, metadata = load_artifacts(os.path.join(args.artifacts_path))

    model, tokenizer = load_model(model_name=args.model_name)
    model.eval()

    with torch.inference_mode():
        prompt = None
        while not prompt or prompt.lower() != 'exit':
            if not prompt:
                prompt = input('Enter your prompt: ')

            # embed query and dense retreival
            query_embedding = create_query_embedding(embedding_model, prompt)
            relevant_chunks, metadata_indices = retrieve_relevant_chunks(index, query_embedding, chunks, top_k=20)

            # reranking
            if args.do_reranking:
                if args.testing:
                     print(f'reranking')

                relevant_chunks = reranking(query, relevant_chunks, top_ranks=7, model_name=args.reranker)

            system_prompt, context, query = build_maxlen_prompt(SYSTEM_PROMPT, relevant_chunks, prompt, tokenizer=tokenizer, max_length=args.max_len)
            messages = [      
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f'CONTEXT:\n{context}\n\nQUESTION:\n{query}\n\nANSWER:\n'}
            ]

            print('\n\n\nLOADING...\n\n\n')

            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = tokenizer(
                inputs,
                return_tensors='pt'
            ).to(args.device)

            with torch.no_grad():
                gen_ids = model.generate(**inputs, max_new_tokens=512, temperature=0.1)

            output_ids = gen_ids[:, inputs.input_ids.shape[-1]:]

            response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print(f'Response: {response}')

            if args.testing:
                print(f'\n\n\nFull prompt used:\n\n\n{messages}\n\n\nResponse: {response}')
                selected_metadata = []
                for m in metadata_indices:
                    selected_metadata.append(metadata[m])
                print(f'\n\nChunk Metadata: {selected_metadata}')

            prompt = input('\n\n\nEnter your prompt: ')
     

if __name__ == '__main__':
    main()