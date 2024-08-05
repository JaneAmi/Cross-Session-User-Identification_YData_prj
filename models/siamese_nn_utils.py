"""
Description: A Python utility contains methods 
which are useful for Siamese NN
"""

import pandas as pd
import numpy as np
import itertools
import random
import sys

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pytorch_tabnet.tab_network import TabNetNoEmbeddings
import torch.nn.functional as F

import matplotlib.pyplot as plt

import evaluation.faiss_umap_utils as fuu


def get_uniq_user_session(df):
    '''This function return a DataFrame with two columns
    'user_id', 'session_id' which contain unique values from 
    the given df'''

    df_upd = df[['user_id', 'session_id']].copy()
    df_upd.drop_duplicates(inplace=True)
    df_upd = df_upd.reset_index(drop=True)

    return df_upd


def add_neg_column(df, df_pairs, neg_fsim = 'user_id_A', num_dev = 6, label = False, neg_ratio=1):

    '''This function creates triplets by adding a session that doesn't belong to the original user.
    Or, if label is True, it creates negative pairs with label
    It can generate a specified ratio of negative pairs based on neg_ratio.'''
    
    ldata, upd_df = int(len(df_pairs) / 5), None

    for i in range(1, num_dev+1):

        #get a part of the df_pairs
        df_pairs_tmp = df_pairs[None if i==1 else ldata * (i - 1) : None if i==num_dev else ldata * i].reset_index(drop=True)

        #get unique users_id
        u_user_id = list(df_pairs_tmp[neg_fsim].unique())

        neg_pairs_df = None

        for _ in range(neg_ratio):

            # randomly get rows with different user_id from df
            df_to_add_tmp = df[~df['user_id'].isin(u_user_id)].sample(n=len(df_pairs_tmp), replace=True).reset_index(drop=True)
            
            if label:
                neg_pairs_df_tmp = pd.concat([df_pairs_tmp.iloc[:, :2], df_to_add_tmp], axis=1)#, keys = ['', 'Negative'])
                neg_pairs_df_tmp = neg_pairs_df_tmp.rename(columns={'session_id': 'session_id_second',
                                                'user_id': 'user_id_second'})
                neg_pairs_df_tmp['label'] = 0 # 0 for dissimilar pairs in contrastive loss calc

                if neg_pairs_df is None:
                    neg_pairs_df = neg_pairs_df_tmp
                
                else:
                    neg_pairs_df = pd.concat([neg_pairs_df_tmp, neg_pairs_df], axis=0, ignore_index=True)
            else:
                # Create a new DataFrame by combining df1 and df2
                neg_pairs_df = pd.concat([df_pairs_tmp, df_to_add_tmp], axis=1)#, keys = ['', 'Negative'])
                neg_pairs_df = neg_pairs_df.rename(columns={'session_id': 'N_session_id',
                                                'user_id': 'user_id_N'})

        if upd_df is None:
            if label:
                upd_df = df_pairs
                upd_df = pd.concat([upd_df, neg_pairs_df], axis=0, ignore_index=True)
            else:
                upd_df = neg_pairs_df
            
        else:
            upd_df = pd.concat([upd_df, neg_pairs_df], axis=0)

    return upd_df.reset_index(drop=True)

    
def create_sim_user_pairs(df, label=False):

    '''This function creates  pairs of sessions,
    that belong to one user'''

    grouped = df.groupby('user_id')['session_id'].apply(list)
        
    # Step 2: Create positive pairs (two different sessions from the same user)
    positive_pairs = []
    for idx, sessions in zip(grouped.index, grouped):
        if len(sessions) > 1:
            for i in range(len(sessions)):
                for j in range(i + 1, len(sessions)):
                    positive_pairs.append((idx, sessions[i], sessions[j]))

    if label:
        # Convert to DataFrame
        positive_df = pd.DataFrame(positive_pairs, columns=['user_id_first', 'session_id_first', 'session_id_second'])
        positive_df['user_id_second'] = positive_df['user_id_first']
        positive_df = positive_df.loc[:,['user_id_first', 'session_id_first', 'user_id_second', 'session_id_second']]
        positive_df['label'] = 1 # 1 for similar pairs in contrastive loss calc
    else:
        # Convert to DataFrame
        positive_df = pd.DataFrame(positive_pairs, columns=['user_id_A', 'A_session_id', 'P_session_id'])

    return positive_df


def create_pairs_with_label(df, random_state=42, neg_ratio=1):

    '''This function create a DataFrame with pairs and labels:
    Labels:
        0 for dissimilar pairs in contrastive loss calc
        1 for similar pairs in contrastive loss calc'''

    tmp_df = df[['session_id', 'user_id']].copy()

    df_sess_pairs_sim = create_sim_user_pairs(tmp_df, label=True)
    df_sess_pairs = add_neg_column(tmp_df, df_sess_pairs_sim, neg_fsim = 'session_id_first', label=True, neg_ratio=neg_ratio)
    #shuffle
    df_sess_pairs = df_sess_pairs.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df_sess_pairs


def create_triplets(df):

    '''This function create a DataFrame with triplets:
    anchor: a session with the user, 
    positive: a session of the similar user,
    negative: a session of the different user'''

    tmp_df = df[['session_id', 'user_id']].copy()

    df_sess_pairs_sim = create_sim_user_pairs(tmp_df)
    df_sess_triplets = add_neg_column(tmp_df, df_sess_pairs_sim, neg_fsim = 'user_id_A')

    return df_sess_triplets
    


class APN_Dataset(Dataset):

    def __init__(self, df_triplets, df_data):
        self.df_triplets = df_triplets
        self.df_data = df_data

    def __len__(self):
        return len(self.df_triplets)

    def __getitem__(self, idx):
        row = self.df_triplets.iloc[idx]

        # Ensure only the necessary data is taken and immediately converted to tensor
        A_session = torch.tensor(self.df_data[self.df_data['session_id'] == row.A_session_id].iloc[0, 2:].values, dtype=torch.float32)
        P_session = torch.tensor(self.df_data[self.df_data['session_id'] == row.P_session_id].iloc[0, 2:].values, dtype=torch.float32)
        N_session = torch.tensor(self.df_data[self.df_data['session_id'] == row.N_session_id].iloc[0, 2:].values, dtype=torch.float32)

        return A_session, P_session, N_session
    

class Pairs_Dataset(Dataset):

    def __init__(self, df_pairs, df_data):
        self.df_pairs = df_pairs
        self.df_data = df_data

    def __len__(self):
        return len(self.df_pairs)

    def __getitem__(self, idx):
        row = self.df_pairs.iloc[idx]

        # Ensure only the necessary data is taken and immediately converted to tensor
        # iloc[0, 2:] - to remove user_id and session_id
        first_session = torch.tensor(self.df_data[self.df_data['session_id'] == row.session_id_first].iloc[0, 2:].values, dtype=torch.float32)
        second_session = torch.tensor(self.df_data[self.df_data['session_id'] == row.session_id_second].iloc[0, 2:].values, dtype=torch.float32)
        label = torch.tensor(row.label, dtype=torch.float32)

        return first_session, second_session, label
    

class TabularEmbeddingModel(nn.Module):
    def __init__(self, input_features, embedding_size):
        super(TabularEmbeddingModel, self).__init__()
        self.embedding_size = embedding_size
        self.fc1 = nn.Linear(input_features, 64)
        self.relu1 = nn.ReLU()  
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU() 
        self.fc3 = nn.Linear(32, embedding_size)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

class TabularEmbeddingModelSgmd(nn.Module):
    def __init__(self, input_features, embedding_size):
        super(TabularEmbeddingModelSgmd, self).__init__()
        self.embedding_size = embedding_size
        self.fc1 = nn.Linear(input_features, 64)
        self.relu1 = nn.Sigmoid()  
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.Sigmoid()  
        self.fc3 = nn.Linear(32, embedding_size)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


class TabularEmbeddingModel4Lrs(nn.Module):
    def __init__(self, input_features, embedding_size):
        super(TabularEmbeddingModel4Lrs, self).__init__()
        self.embedding_size = embedding_size
        self.fc1 = nn.Linear(input_features, 96)
        self.relu1 = nn.ReLU()   
        self.fc2 = nn.Linear(96, 64)
        self.relu2 = nn.ReLU()  
        self.fc3 = nn.Linear(64, 32)
        self.sigm = nn.Sigmoid()  
        self.fc4 = nn.Linear(32, embedding_size)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.sigm(self.fc3(x))
        x = self.fc4(x)
        return x



class TabularEmbeddingModel4Lrrelu(nn.Module):
    def __init__(self, input_features, embedding_size):
        super(TabularEmbeddingModel4Lrrelu, self).__init__()
        self.embedding_size = embedding_size
        self.fc1 = nn.Linear(input_features, 96)
        self.relu1 = nn.ReLU()   
        self.fc2 = nn.Linear(96, 64)
        self.relu2 = nn.ReLU()  
        self.fc3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(32, embedding_size)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        return x


class TabNetEmbeddingModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TabNetEmbeddingModel, self).__init__()
        self.embedding_size = output_dim
        self.tabnet = TabNetNoEmbeddings(input_dim=input_dim, output_dim=output_dim,
                                         n_d=8, n_a=8, n_steps=3, gamma=1.5,
                                         n_independent=2, n_shared=2, epsilon=1e-15,
                                         virtual_batch_size=150, momentum=0.02)

    def forward(self, x):
        x = self.tabnet(x)
        return x
    

    def print_device_allocation(self):
        for name, param in self.model.named_parameters():
            print(f"Layer: {name} | Size: {param.size()} | Device: {param.device}")
        print("Additional tensors:")
        print(f"Group attention matrix Device: {self.model.tabnet.encoder.group_attention_matrix.device}")
    

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
      # Calculate the euclidean distance and calculate the contrastive loss
      euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)

      loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                                    (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


      return loss_contrastive
    


class ContrastiveCosineLoss(torch.nn.Module):
    def __init__(self, margin=0.5):
        super(ContrastiveCosineLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        
        # Calculate the Cosine similarity
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        cosine_similarity = cos(output1, output2)
        
        # Cosine distance is defined as 1 - Cosine similarity
        cosine_distance = 1 - cosine_similarity

        # Calculate the contrastive loss
        loss_contrastive = torch.mean((1-label) * 0.5 * torch.pow(cosine_distance, 2) +
                                      label * 0.5 * torch.pow(torch.clamp(self.margin - cosine_distance, min=0.0), 2))

        return loss_contrastive


def train_fn_triplets(model, dataloader, optimizer, criterion, device, l2_strength=0, l1_strength=0, tabnet=False, pairs=False):
    model.train() 
    total_loss = 0.0 
    
    for A,P,N in tqdm(dataloader):
        A,P = A.to(device), P.to(device)

        if pairs:
            label = N.to(device)
        else: 
            N = N.to(device)

        if tabnet:
            A_embs, _ = model(A)
            P_embs, _ = model(P)

            if not pairs:
                N_embs, _ = model(N)
        
        else:
            A_embs = model(A)
            P_embs = model(P)

            if not pairs:
                N_embs = model(N)
        
        if pairs:

            loss = criterion(A_embs, P_embs, label)
            # Apply different weights based on the label
            loss_weights = torch.where(label == 1, 1.0, 0.5)
            loss = (loss * loss_weights).mean() 
            
        else:
            loss = criterion(A_embs, P_embs, N_embs)

        if l2_strength > 0:
            l2_reg_ = torch.tensor(0.).to(device)
            for param in model.parameters():
                l2_reg_ += torch.norm(param, 2)**2
            loss += l2_strength * l2_reg_
        
        if l1_strength > 0:
            l1_reg_ = torch.tensor(0.).to(device)
            for param in model.parameters():
                l1_reg_ += torch.norm(param, 1)
            loss += l1_strength * l1_reg_

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)


def eval_fn_triplets(model, dataloader, criterion, device, tabnet=False, pairs=False):
    model.eval()  
    total_loss = 0.0 
    
    with torch.no_grad():
        for A,P,N in tqdm(dataloader):
            A,P = A.to(device), P.to(device)

            if pairs:
                label = N.to(device)
            else: 
                N = N.to(device)

            if tabnet:
                A_embs, _ = model(A)
                P_embs, _ = model(P)

                if not pairs:
                    N_embs, _ = model(N)
            
            else:
                A_embs = model(A)
                P_embs = model(P)

                if not pairs:
                    N_embs = model(N)
            
            if pairs:
                loss = criterion(A_embs, P_embs, label)
            else:
                loss = criterion(A_embs, P_embs, N_embs)

            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    

def training_func(model, 
                  epochs, 
                  trainloader, 
                  valloader, 
                  lr, 
                  l1_strength, 
                  l2_strength, 
                  optimizer, 
                  criterion,
                  device, 
                  pairs=True, 
                  tabnet=False, 
                  best_valid_loss=np.inf,
                  train_loss_list=[], 
                  val_loss_list=[], 
                  save_res_to_file=False, 
                  plot_result = True):

    EPOCHS = epochs

    if save_res_to_file:
        # Open a file to append the output
        file_path = f"../models/logs/log_{model._get_name()}_{'pairs' if pairs else 'triplets'}.txt"
        f = open(file_path, "a")
        original_stdout = sys.stdout  # Save a reference to the original standard output
        sys.stdout = f  # Change the standard output to the file above
    else:
        f = None

    print(f"""embedding_size: {model.embedding_size}\nlr: {lr}\nl2_strength: {l2_strength}\nl1_strength: {l1_strength}\n""")

    for i in range(EPOCHS):
        train_loss = train_fn_triplets(model, trainloader, optimizer, criterion, device, \
                                            l2_strength=l2_strength, l1_strength=l1_strength, pairs=pairs, tabnet=tabnet)
        valid_loss = eval_fn_triplets(model, valloader, criterion, device, pairs=pairs, tabnet=tabnet)
        train_loss_list.append(train_loss)
        val_loss_list.append(valid_loss)

        
        if valid_loss < best_valid_loss:
            torch.save(model.state_dict(), f"../models/best_models/best_{model._get_name()}_{'pairs' if pairs else 'triplets'}.pt")
            best_valid_loss = valid_loss
            print("SAVED_WEIGHT_SUCCESS")
        
        print(f"EPOCHS: {i+1} train_loss: {train_loss} valid_loss: {valid_loss}")

    if save_res_to_file:
        sys.stdout = original_stdout  # Reset the standard output to its original value
        f.close()  # Close the file after writing

    if plot_result:
        print(f"""\nembedding_size: {model.embedding_size}\nlr: {lr}\nl2_strength: {l2_strength}\nl1_strength: {l1_strength}\n""")
        plot_training_results(train_loss_list, val_loss_list, model, pairs)

    return train_loss_list, val_loss_list




def training_func_wfaiss(model, 
                  epochs, 
                  trainloader, 
                  valloader, 
                  lr, 
                  l1_strength, 
                  l2_strength, 
                  optimizer, 
                  criterion,
                  device, 
                  pairs=True, 
                  tabnet=False, 
                  best_valid_loss=np.inf,
                  train_loss_list=[], 
                  val_loss_list=[], 
                  save_res_to_file=False, 
                  plot_result = True,
                  save_plt_to_file = True,
                  f_path = None,
                  data_train_wo_emb=None, 
                  test_set_wo_emb=None, 
                  data_train_wo_emb_10=None,
                  umap_hparams=None,
                  eval_faiss=False, 
                  save_model = True,
                  faiss_eval_dict = {'Epoch': [],
                                     'train_loss': [],
                                     'val_loss': [],
                                     'k1': [],
                                     'k3': [],
                                     'k5': []}):

    EPOCHS = epochs

    if save_res_to_file:
        # Open a file to append the output
        file_path = f"{f_path}/log_{model._get_name()}_{'pairs' if pairs else 'triplets'}.txt"
        f = open(file_path, "a")
        original_stdout = sys.stdout  # Save a reference to the original standard output
        sys.stdout = f  # Change the standard output to the file above
    else:
        f = None

    

    print(f"""embedding_size: {model.embedding_size}\nlr: {lr}\nl2_strength: {l2_strength}\nl1_strength: {l1_strength}\n""")

    for i in range(EPOCHS):
        train_loss = train_fn_triplets(model, trainloader, optimizer, criterion, device, \
                                            l2_strength=l2_strength, l1_strength=l1_strength, pairs=pairs, tabnet=tabnet)
        valid_loss = eval_fn_triplets(model, valloader, criterion, device, pairs=pairs, tabnet=tabnet)
        train_loss_list.append(train_loss)
        val_loss_list.append(valid_loss)

        faiss_eval_dict['train_loss'].append(train_loss)
        faiss_eval_dict['val_loss'].append(valid_loss)

        
        if valid_loss < best_valid_loss:
            torch.save(model.state_dict(), f"{f_path}/models/best_{model._get_name()}_{'pairs' if pairs else 'triplets'}.pt")
            best_valid_loss = valid_loss
            print("SAVED_WEIGHT_SUCCESS")
        
        print(f"EPOCHS: {i+1} train_loss: {train_loss} valid_loss: {valid_loss}")

        if save_model:
            torch.save(model.state_dict(), f"{f_path}/models/{model._get_name()}_{'pairs' if pairs else 'triplets'}_epoch_{i+1}.pt")

        if save_plt_to_file:
            file_path_plot = f"{f_path}/plot_loss_epoch_{i+1}.png"
            plot_training_results(train_loss_list, val_loss_list, model, pairs, save_to_file=True, f_path=file_path_plot)

        if eval_faiss:
            file_path_umap = f"{f_path}/umap_epoch_{i+1}.png"
            faiss_eval_dict['Epoch'].append(i+1)
            faiss_eval_dict = create_eval_faiss_umap(model, data_train_wo_emb, test_set_wo_emb, data_train_wo_emb_10, umap_hparams, faiss_eval_dict,
                                   save_to_file=True, f_path=file_path_umap, tabnet=tabnet)
            
            model.to(device)

            if tabnet:
                # Ensure group attention matrix is also on GPU
                if hasattr(model.tabnet.encoder, 'group_attention_matrix'):
                    model.tabnet.encoder.group_attention_matrix = model.tabnet.encoder.group_attention_matrix.to(device)

        if (i + 1) % 5 == 0:
            tmp_df = pd.DataFrame(faiss_eval_dict)
            tmp_df.to_csv(f"{f_path}/res_table_epoch{i+1}.csv")

    if save_res_to_file:
        sys.stdout = original_stdout  # Reset the standard output to its original value
        f.close()  # Close the file after writing

    if plot_result:
        print(f"""\nembedding_size: {model.embedding_size}\nlr: {lr}\nl2_strength: {l2_strength}\nl1_strength: {l1_strength}\n""")
        plot_training_results(train_loss_list, val_loss_list, model, pairs)


    return faiss_eval_dict
    



def plot_training_results(train_loss_list, val_loss_list, model, pairs=True, save_to_file=False, f_path=None, max_epochs=50):
    plt.figure(figsize=(10, 5))
    epochs = range(len(train_loss_list))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_list, 'r', label='Training Loss')
    plt.plot(epochs, val_loss_list, 'b', label='Validation Loss')
    plt.title(f"Training and Validation Loss. \n{model._get_name()} {'pairs' if pairs else 'triplets'}\n")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.xticks(ticks=range(0, max_epochs, 5))

    if save_to_file:
        plt.savefig(f_path)
        plt.close() 
    else:
        plt.show();


def create_eval_faiss_umap(model, data_train_wo_emb, test_set_wo_emb, data_train_wo_emb_10, umap_hparams, faiss_eval_dict,
                           save_to_file=False, f_path=None, tabnet=False):
    model.to('cpu')

    if tabnet:
        # Ensure group attention matrix is also on GPU
        if hasattr(model.tabnet.encoder, 'group_attention_matrix'):
            model.tabnet.encoder.group_attention_matrix = model.tabnet.encoder.group_attention_matrix.to('cpu')

    index_wo_emb, user_ids_wo_emb = fuu.create_faiss_db(data_train_wo_emb, model, tabnet=tabnet)
    faiss_eval_dict

    for k in [1, 3, 5]:
        # print("k =", k)
        faiss_search_res_cm_pairs = fuu.evaluate_faiss_db(k, 
                                                    test_set_wo_emb, 
                                                    index_wo_emb, 
                                                    user_ids_wo_emb, 
                                                    model=model,
                                                    tabnet=tabnet
                                                    )
        faiss_eval_dict[f'k{k}'].append(sum(faiss_search_res_cm_pairs['Result'])/len(faiss_search_res_cm_pairs['Result']))
        
        print()

    fuu.umap_plot(data_train_wo_emb_10, umap_hparams, model=model, save_to_file=save_to_file, f_path=f_path, tabnet=tabnet)
    return faiss_eval_dict