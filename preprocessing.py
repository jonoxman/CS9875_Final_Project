from protT5 import get_embs_T5
import numpy as np

####################################################################################################################################################################################
#for processing the input CSVs
def parse_dataframe(data, model, tokenizer):
    '''Generates embeddings for each sequence in the input DataFrame 
        and concatenates the embeddings into a single array. 
        It also concatenates all labels into a single string.
        
        Parameters:
        data (DataFrame): Input DataFrame containing 'seq' and 'sst3' columns
        model: Pre-trained model for generating embeddings
        tokenizer: Tokenizer corresponding to the pre-trained model

        Returns:
        concatenated_embeddings (numpy array): Concatenated embeddings for all sequences
        label_all (str): Concatenated labels for all sequences
        '''

    examples_seqs = []
    concatenated_embeddings = None
    label_all = ""

    # Step 1: Extract sequences and labels from DataFrame
    for index, row in data.iterrows():
        seq = row["seq"]
        labels8 = row["sst3"]
        examples_seqs.append((seq, labels8))

    # Step 2: Generate embeddings and collect labels
    for exampleSeq in examples_seqs:
        seq = exampleSeq[0]
        label = exampleSeq[1]

        # Generate embeddings (assuming get_embs_T5 outputs a single embedding array)
        emb = get_embs_T5(model, tokenizer, [seq], 1)[0].cpu().numpy()

        if concatenated_embeddings is None:  # Handle the first case
            concatenated_embeddings = emb
        else:
            concatenated_embeddings = np.concatenate((concatenated_embeddings, emb), axis=0)

        label_all = label_all + label

    return concatenated_embeddings, label_all
####################################################################################################################################################################################

####################################################################################################################################################################################
def shuffle_data(X, y):
    '''Shuffles the features and labels in unison to maintain the correct mapping between them.
    
        Parameters:
        X (numpy array): Array of features (embeddings)
        y (list): List of labels corresponding to the features
        Returns:
        shuffled_X (numpy array): Shuffled array of features
        shuffled_y (list): Shuffled list of labels
    '''
    #ensure consistent shuffling for features and labels
    assert len(X) == len(y), "Features and labels must have the same length"
    indices = np.arange(len(X))  #create an array of indices
    np.random.shuffle(indices)  #shuffle the indices
    return X[indices], [y[i] for i in indices]
####################################################################################################################################################################################
