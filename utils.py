
import faiss
import json
from sentence_transformers import SentenceTransformer 


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
        chunk_ids = tokenizer(chunk)['input_ids']
        combined_len = len(chunk_ids) + fixed_len + used

        if combined_len <= max_length:
            chunks_to_keep.append(chunk)
            used += len(chunk_ids)

        else:
            remainder = max_length - (fixed_len + used)
            if remainder > 0:
                 chunk_ids = chunk_ids[:remainder]
                 chunk_remainder = tokenizer.decode(chunk_ids, skip_special_tokens=True)
                 chunks_to_keep.append(chunk_remainder)
            break
    context = f'\n\n'.join(chunks_to_keep)
    return system_prompt, context, query

