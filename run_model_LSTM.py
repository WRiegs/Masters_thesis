# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

'''
I actually heavily modified the original Facebook code from the extract file.

Don't know how this works with their copyright statement seen above.

Anyways... This script is used to actually run and get results from squidly.

It's pretty fast but can be optimised more...

usually 2 seconds ish a sequence.

'''


import argparse
import pathlib
import torch
import torch.nn as nn
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained 
import xgboost as xgb
import json
import pandas as pd
import time
import pickle
import numpy as np

def create_parser():
    parser = argparse.ArgumentParser(
        description="Extract per-token representations and model outputs for sequences in a FASTA file"  # noqa
    )
    parser.add_argument(
        "file",
        type=pathlib.Path,
        help="FASTA or CSV file on which to extract representations. CSV must have column Sequence, with sequences, and Entry with id. Such as is used by Uniprot",
    )
    parser.add_argument(
        "AS_contrastive_model",
        type=pathlib.Path,
        help="contrastive model to use for generating contrastive representations"
    )
    parser.add_argument(
        "BS_contrastive_model",
        type=pathlib.Path,
        help="contrastive model to use for generating contrastive representations"
    )
    parser.add_argument(
        "AS_model",
        type=pathlib.Path,
        help="Model to use for predicting active sites"
    )
    parser.add_argument(
        "BS_model",
        type=pathlib.Path,
        help="Model to use for predicting binding sites"
    )
    parser.add_argument(
        "output_dir",
        type=pathlib.Path,
        help="output directory for extracted representations",
    )
    parser.add_argument(
        "--toks_per_batch", 
        type=int, 
        default=10, 
        help="maximum batch size"
    )
    parser.add_argument(
        "--AS_threshold",
        type = float,
        default=0.9,
        help="Threshold for active site prediction"
    )
    parser.add_argument(
        "--BS_threshold",
        type = float,
        default=0.9,
        help="Threshold for binding site prediction"
    )
    parser.add_argument(
        "--logits",
        action="store_true",
        default=False,
        help="Whether to output the logits for the active and binding site predictions"
    )

    return parser

class ContrastiveModel(nn.Module):
    def __init__(self, input_dim=5120, output_dim=128, dropout_prob=0.1):
        super(ContrastiveModel, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(input_dim, int(input_dim/2))
        self.fc2 = nn.Linear(int(input_dim/2), int(input_dim/4))
        self.fc3 = nn.Linear(int(input_dim/4),output_dim)
        
    def forward(self, x):
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    

class ProteinLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, num_layers, dropout_rate):
        super(ProteinLSTM, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes protein embeddings as inputs and outputs hidden states with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout_rate, bidirectional=True)

        # The linear layer that maps from hidden state space to the output space
        self.hidden2out = nn.Linear(hidden_dim*2, output_dim)
        
        self.best_model_path = ""
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.hidden2out(lstm_out)
        return output
        


def manual_pad_sequence_tensors(tensors, target_length, padding_value=0):
    """
    Manually pads a list of 2-dimensional tensors along the first dimension to the specified target length.

    Args:
    - tensors (list of Tensors): List of input tensors to pad.
    - target_length (int): Target length to pad/trim the tensors along the first dimension.
    - padding_value (scalar, optional): Value for padding, default is 0.

    Returns:
    - padded_tensors (list of Tensors): List of padded tensors.
    """
    padded_tensors = []
    for tensor in tensors:
        # Check if padding is needed along the first dimension
        if tensor.size(0) < target_length:
            pad_size = target_length - tensor.size(0)
            # Create a padding tensor with the specified value
            padding_tensor = torch.full((pad_size, tensor.size(1)), padding_value, dtype=tensor.dtype, device=tensor.device)
            # Concatenate the padding tensor to the original tensor along the first dimension
            padded_tensor = torch.cat([tensor, padding_tensor])
        # If the tensor is longer than the target length, trim it along the first dimension
        else:
            padded_tensor = tensor[:target_length, :]
        padded_tensors.append(padded_tensor)
    return padded_tensors



def get_fasta_from_df(df, output):
    # remove duplicates from the dataframe at the entry level
    # print how many are above 1024
    print(f"Number of sequences above 1024 length: {len(df[df['seq_len'] > 1024])}")
    df = df[df['seq_len'] <= 1024]
    duplicate_free = df.drop_duplicates(subset='Entry')
    # reset the index
    duplicate_free.reset_index(drop=True, inplace=True)
    seqs = duplicate_free['Sequence']
    entry = duplicate_free['Entry']
    # write the sequences to a fasta file
    with open(output+'.fasta', 'w') as f:
        for i, seq in enumerate(seqs):
            f.write(f'>{entry[i]}\n')
            f.write(f'{seq}\n')
    return pathlib.Path(output+'.fasta')

model_location = "/scratch/project/squid/models/ESM2/esm2_t48_15B_UR50D.pt"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

include = "per_tok"

use_binding = True

def main(args):
    if args.file.suffix == '.csv':
        csv = args.file
        # filter out any sequences that are longer than 1024 from the csv
        df = pd.read_csv(args.file)
        df['seq_len'] = df['Sequence'].apply(lambda x: len(x))
        args.file = get_fasta_from_df(df, str(args.output_dir / args.file.stem))
    elif args.file.suffix == '.tsv':
        csv = args.file
        # filter out any sequences that are longer than 1024 from the csv
        df = pd.read_csv(args.file, sep='\t')
        df['seq_len'] = df['Sequence'].apply(lambda x: len(x))
        args.file = get_fasta_from_df(df, str(args.output_dir / args.file.stem))
    else: 
        csv = None
    
    start_time = time.time()
    model, alphabet = pretrained.load_model_and_alphabet(model_location)
    model.eval()
    
    # load the contrastive model
    AS_contrastive_model = ContrastiveModel(output_dim=128)
    AS_contrastive_model.load_state_dict(torch.load(args.AS_contrastive_model))
    AS_contrastive_model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        AS_contrastive_model = AS_contrastive_model.cuda()
        print("Transferred representation models to GPU")
        
    # Load AS LSTM models
    AS_model = ProteinLSTM(embedding_dim=128, hidden_dim=128, output_dim=1, num_layers=2, dropout_rate=0.1)
    AS_model.load_state_dict(torch.load(args.AS_model))
    AS_model.eval()
    if torch.cuda.is_available():
        AS_model = AS_model.cuda()
        print("Transferred AS models to GPU")
    
    if use_binding:
        BS_contrastive_model = ContrastiveModel(output_dim=64)
        BS_contrastive_model.load_state_dict(torch.load(args.BS_contrastive_model))
        BS_contrastive_model.eval()
        if torch.cuda.is_available():
            BS_contrastive_model = BS_contrastive_model.cuda()
            print("Transferred representation models to GPU")
            
        # Load the BS LSTM model
        BS_model = ProteinLSTM(embedding_dim=64, hidden_dim=64, output_dim=1, num_layers=2, dropout_rate=0.1)
        BS_model.load_state_dict(torch.load(args.BS_model))
        BS_model.eval()
        if torch.cuda.is_available():
            BS_model = BS_model.cuda()
            print("Transferred BS models to GPU")

    dataset = FastaBatchedDataset.from_file(args.file)
    batches = dataset.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches
    )
    print(f"Read {args.file} with {len(dataset)} sequences")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        embeddings_labels = []
        predicted_AS_residues = []
        predicted_BS_residues = []
        AS_probabilities_list = []
        BS_probabilities_list = []
        highest_prob_ASs = []
        highest_prob_AS_possies = []
        highest_prob_BSs = []
        highest_prob_BS_possies = []
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(
                f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
            )
            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=[48], return_contacts=False)

            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }
            
            for i, label in enumerate(labels):
                args.output_file = args.output_dir / f"{label}.pt"
                args.output_file.parent.mkdir(parents=True, exist_ok=True)
                result = {"label": label}

                if "per_tok" in include:
                    result["representations"] = {
                        layer: t[i, 1 : len(strs[i]) + 1].clone()
                        for layer, t in representations.items()
                    }
                active_sites = []
                pos = 0
                
                seq = result["representations"][48]
                
                dataloader = torch.utils.data.DataLoader(seq, batch_size=len(seq), shuffle=False)
                AS_contrastive_rep = []
                # put the dataloader through the model and get the new sequence
                for batch in dataloader:
                    batch = batch.to(device)
                    AS_contrastive_rep.extend(AS_contrastive_model(batch))
                    del batch

                # make tensor of the representations
                AS_contrastive_rep = torch.stack(AS_contrastive_rep)
                AS_contrastive_rep = manual_pad_sequence_tensors([AS_contrastive_rep], 1024)[0]
                
                if use_binding:
                    BS_contrastive_rep = []
                    # put the dataloader through the model and get the new sequence
                    for batch in dataloader:
                        batch = batch.to(device)
                        BS_contrastive_rep.extend(BS_contrastive_model(batch))
                        del batch

                    # make tensor of the representations
                    BS_contrastive_rep = torch.stack(BS_contrastive_rep)
                    BS_contrastive_rep = manual_pad_sequence_tensors([BS_contrastive_rep], 1024)[0]
                
                # predict using the AS LSTM model
                active_sites = AS_model(AS_contrastive_rep.cuda())
                active_sites = torch.sigmoid(active_sites)
                AS_probabilities = list(np.array(active_sites.cpu()))
                if args.logits:
                    logits = list(active_sites.cpu())
                    logits = [float(logit[0]) for logit in logits]
                    AS_probabilities_list.append(logits)
                highest_prob_AS = max(AS_probabilities)
                highest_prob_AS_pos = AS_probabilities.index(highest_prob_AS)
                active_sites = [i for i, x in enumerate(active_sites) if x > args.AS_threshold] 
                highest_prob_ASs.append(highest_prob_AS)
                highest_prob_AS_possies.append(highest_prob_AS_pos)
                predicted_AS_residues.append(active_sites)
                
                
                # predict using the BS LSTM model
                if use_binding:
                    binding_sites = BS_model(BS_contrastive_rep.cuda())
                    binding_sites = torch.sigmoid(binding_sites)
                    BS_probabilities = list(np.array(binding_sites.cpu()))
                    if args.logits:
                        BS_probabilities_list.append(list(np.array(binding_sites.cpu())))
                    highest_prob_BS = max(BS_probabilities)
                    highest_prob_BS_pos = BS_probabilities.index(highest_prob_BS)
                    binding_sites = [i for i, x in enumerate(binding_sites) if x > args.BS_threshold]
                    highest_prob_BSs.append(highest_prob_BS)
                    highest_prob_BS_possies.append(highest_prob_BS_pos)
                    predicted_BS_residues.append(binding_sites)
                
                embeddings_labels.append(result["label"])

    results = []
    for label, AS_residues, AS_pos_highest, AS_highest_prob, BS_residues, BS_pos_highest, BS_highest_prob in zip(embeddings_labels, predicted_AS_residues, highest_prob_AS_possies, highest_prob_ASs, predicted_BS_residues, highest_prob_BS_possies, highest_prob_BSs):
        AS_residues = '|'.join([str(residue) for residue in AS_residues])
        BS_residues = '|'.join([str(residue) for residue in BS_residues])
        results.append([label, AS_residues, AS_pos_highest, AS_highest_prob, BS_residues, BS_pos_highest, BS_highest_prob])

    # save the results as a df, using the list of lists, with no index
    results_df = pd.DataFrame(results, columns=["label", "AS_residues", "AS_pos_highest", "AS_highest_prob", "BS_residues", "BS_pos_highest", "BS_highest_prob"], index=None)
        
    if csv is not None:
        results_df = pd.merge(df, results_df, how='left', left_on='Entry', right_on='label')
        results_df.to_csv(str(args.output_dir / args.file.stem) + 'filtered_LSTM.tsv', index=False, sep = '\t')
        args.file.unlink()
    else:
        results_df.to_csv(str(args.output_dir / args.file.stem) + 'filtered_LSTM.tsv', index=False, sep = '\t')
    
    elapsed_time = time.time() - start_time
    print (f"Elapsed time: {elapsed_time}")   
    
    # make a dictionary from the embedding labels and the probabilities
    
    # now make a dictionary where every sublist in the list is a dictionary with the label as key and the probabilities as a list of values
    dict_of_AS_probs = dict(zip(embeddings_labels, AS_probabilities_list))
    
    # save the dictionary as a pickle
    if args.logits:
        with open(args.output_dir / 'site_probs.pkl', 'wb') as f:
            # dump a dictionary with labels as key, and the probabilities of both AS and BS results as values
            pickle.dump(dict_of_AS_probs, f)
    


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)