import torch
from transformers import T5Tokenizer, T5EncoderModel
import re

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #always going to be GPU
#code to initialize protT5
def ProtT5_initialize():
    '''Initializes the ProtT5 model and tokenizer, moves the model to the appropriate device, and sets it to evaluation mode.
    Returns:
    ProtT5: The initialized ProtT5 model ready for generating embeddings
    ProtT5_tokenizer: The corresponding tokenizer for the ProtT5 model
    '''
    ProtT5 = T5EncoderModel.from_pretrained("/home/jmalec/projects/def-ilie/jmalec/protT5", local_files_only=True)
    ProtT5.half()
    ProtT5 = ProtT5.to(device)
    ProtT5 = ProtT5.eval()
    ProtT5_tokenizer = T5Tokenizer.from_pretrained("/home/jmalec/projects/def-ilie/jmalec/protT5", do_lower_case=False)

    return ProtT5, ProtT5_tokenizer


def get_embs_T5(T5, tokenizer, sequences, n):
    '''
    Generates embeddings for a list of protein sequences using the ProtT5 model and tokenizer.
    Parameters:
    T5: The initialized ProtT5 model for generating embeddings
    tokenizer: The corresponding tokenizer for the ProtT5 model
    sequences: A list of protein sequences for which to generate embeddings
    n: The number of sequences to process (if n is less than the length of sequences, only the first n sequences will be processed)
    Returns:
    final_embs: A list of embeddings corresponding to the input sequences, where each embedding is a
    '''
    sequence_examples = sequences[:n]

    # this will replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
    sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

    # tokenize sequences and pad up to the longest sequence in the batch
    ids = tokenizer.batch_encode_plus(sequence_examples, add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    # generate embeddings
    with torch.no_grad():
        embedding_repr = T5(input_ids=input_ids, attention_mask=attention_mask)

    last_layer_repr = embedding_repr.last_hidden_state
    final_embs = []
    for i in range(len(last_layer_repr)):
        final_embs.append(last_layer_repr[i , :len(sequences[i])])

    return final_embs
