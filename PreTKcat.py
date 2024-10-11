from os.path import join

import torch
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
import xgboost as xgb
from build_vocab import WordVocab
from pretrain_trfm import TrfmSeq2seq
from utils import split
import json
from transformers import T5EncoderModel, T5Tokenizer
from transformers import AutoTokenizer, EsmModel, EsmTokenizer
from transformers import BertModel, BertTokenizer
import re
import gc
from sklearn import metrics
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
import random
import pickle
import math
from MPG_util.graph_bert import *
from MPG_util.mol2graph import *
from sklearn.model_selection import KFold
from hyperopt import fmin, tpe, rand, hp, Trials

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



def graph_to_vec(Smiles):
    MolEncoder = MolGT(num_layer=5, emb_dim=768, heads=12, num_message_passing=3, drop_ratio=0.5)
    MolEncoder.load_state_dict(torch.load('MolGNet.pt', map_location=torch.device('cpu')))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MolEncoder = MolEncoder.to(device)
    MolEncoder = MolEncoder.eval()
    Mol_vector_list = []
    i = 1
    with torch.no_grad():
        for smiles in Smiles:
            mol = Chem.MolFromSmiles(smiles)
            data = mol_to_graph_data_dic(mol).to(device)
            Mol_vector = MolEncoder(data)  # [19, 768]
            batch = torch.tensor(np.zeros(data.x.size(0), dtype=np.int64)).to(device)
            # Mol_vector = global_mean_pool(Mol_vector, batch)
            # Mol_vector = Mol_vector[data.dummy_node_indices]
            # Mol_vector = global_max_pool(Mol_vector, batch)
            Mol_vector = global_add_pool(Mol_vector, batch)

            Mol_vector = Mol_vector.detach().cpu().numpy()
            Mol_vector_list.append(Mol_vector)
            i = i + 1
    return Mol_vector_list

# def Seq_to_vec(Sequence):
#     tokenizer = BertTokenizer.from_pretrained("prot_bert", do_lower_case=False)
#     model = BertModel.from_pretrained("prot_bert")
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)
#     model = model.eval()
#     features = []
#     with torch.no_grad():
#         for i in range(len(Sequence)):
#             protein_sequence = Sequence[i].upper()
#             protein_sequence = " ".join(protein_sequence)
#             sequence_Example = re.sub(r"[UZOB]", "X", protein_sequence)
#             encoded_input = tokenizer(sequence_Example, return_tensors='pt', truncation=True, padding=True, max_length=1024).to(device)
#             output = model(**encoded_input).pooler_output.squeeze(0).cpu().detach().numpy()

#             features.append(output)
#     return np.vstack(features)

# def Seq_to_vec(Sequence):
#     tokenizer = EsmTokenizer.from_pretrained("esm2_t33_650M_UR50D")
#     model = EsmModel.from_pretrained("esm2_t33_650M_UR50D")  # esm1v_t33_650M_UR90S_1  esm2_t33_650M_UR50D
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)
#     model = model.eval()
#     features = []
#     with torch.no_grad():
#         for i in range(len(Sequence)):
#             inputs = tokenizer(Sequence[i], return_tensors="pt", truncation=True, padding=True, max_length=1024).to(device)
#             outputs = model(**inputs).pooler_output.squeeze(0).cpu().detach().numpy()
#             features.append(outputs)
#     return np.vstack(features)

def Seq_to_vec(Sequence):
    for i in range(len(Sequence)):
        if len(Sequence[i]) > 1000:
            Sequence[i] = Sequence[i][:500] + Sequence[i][-500:]
    sequences_Example = []
    for i in range(len(Sequence)):
        zj = ''
        for j in range(len(Sequence[i]) - 1):
            zj += Sequence[i][j] + ' '
        zj += Sequence[i][-1]
        sequences_Example.append(zj)
    tokenizer = T5Tokenizer.from_pretrained("prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("prot_t5_xl_uniref50")
    gc.collect()
    print(torch.cuda.is_available())
    # 'cuda:0' if torch.cuda.is_available() else
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()
    features = []
    for i in range(len(sequences_Example)):
        print('For sequence ', str(i+1))
        sequences_Example_i = sequences_Example[i]
        sequences_Example_i = [re.sub(r"[UZOB]", "X", sequences_Example_i)]
        ids = tokenizer.batch_encode_plus(sequences_Example_i, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
        embedding = embedding.last_hidden_state.cpu().numpy()
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][:seq_len - 1]
            features.append(seq_emd)
            # print("seq_emd shape:", seq_emd.shape)
    features_normalize = np.zeros([len(features), len(features[0][0])], dtype=float)
    for i in range(len(features)):
        for k in range(len(features[0][0])):
            for j in range(len(features[i])):
                features_normalize[i][k] += features[i][j][k]
            features_normalize[i][k] /= len(features[i])
    return features_normalize




def Kcat_predict(Ifeature, Label, sequence_new, Smiles_new, ECNumber_new, Substrate_new, Type_new, Temp_k_norm_new, Inv_Temp_norm_new, Temp_new, Temp_k_new, Inv_Temp_new):
    kf = KFold(n_splits=10, shuffle=True)
    fold = 1
    for train_index, test_index in kf.split(Ifeature):
        Train_data, Test_data = Ifeature[train_index], Ifeature[test_index]
        Train_label, Test_label = Label[train_index], Label[test_index]
        print("Train_data:", len(Train_data))
        print("Test_data:", len(Test_data))


        model = ExtraTreesRegressor()

        model.fit(Train_data, Train_label)
        Pre_all_label = model.predict(Ifeature)

        Training_or_test = np.zeros(len(Ifeature))
        Training_or_test[test_index] = 1

        res = pd.DataFrame({'sequence': sequence_new, 'smiles': Smiles_new, 'ECNumber': ECNumber_new,
                            'Substrate': Substrate_new, 'Type': Type_new,
                            'Label': Label, 'Predict_Label': Pre_all_label, 'Training or test': Training_or_test,
                            'Temp': Temp_new, 'Temp_k': Temp_k_new, 'Inv_Temp': Inv_Temp_new,
                            'Temp_K_norm': Temp_k_norm_new, 'Inv_Temp_norm': Inv_Temp_norm_new})


        res.to_csv(f'Results/K_Fold/PreTKcat/{fold}_metrics.csv', index=False)
        # Predict on the test set
        Pre_test_label = model.predict(Test_data)
        # Pre_test_label = model.predict(xgb.DMatrix(Test_data))

        # Calculate performance metrics
        r2 = r2_score(Test_label, Pre_test_label)
        rmse = np.sqrt(mean_squared_error(Test_label, Pre_test_label))
        mae = mean_absolute_error(Test_label, Pre_test_label)
        pcc = pearsonr(Test_label, Pre_test_label)[0]

        # Print performance metrics
        print(f'Fold {fold} - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, PCC: {pcc:.4f}')
        fold += 1


if __name__ == '__main__':
    # Dataset Load
    datasets = np.array(pd.read_csv('datasets/DLTKcat_data/kcat_merge_DLTKcat.csv')).T
    print(datasets.shape)
    sequence = [data for data in datasets[10]]
    Smiles = [data for data in datasets[9]]
    Label = [float(data) for data in datasets[5]]
    ECNumber = [data for data in datasets[0]]
    Substrate = [data for data in datasets[1]]
    Type = [data for data in datasets[3]]
    Temp = [data for data in datasets[6]]
    Temp_k = [data for data in datasets[7]]
    Inv_Temp = [data for data in datasets[8]]
    Temp_k_norm = [data for data in datasets[13]]
    Inv_Temp_norm = [data for data in datasets[14]]

    for i in range(len(Label)):
        if Label[i] == 0:
            Label[i] = -10000000000
        else:
            Label[i] = math.log(Label[i], 10)
    print(max(Label), min(Label))
    # Feature Extractor
    smiles_input = graph_to_vec(Smiles)
    smiles_input = np.vstack(smiles_input)
    print("MPG shape:", smiles_input.shape)

    sequence_input = Seq_to_vec(sequence)
    print("protein shape:", sequence_input.shape)

    feature = np.concatenate((smiles_input, sequence_input), axis=1)
    print("feature shape:", feature.shape)

    Label = np.array(Label)
    # Input dataset
    feature_new = []
    Label_new = []
    sequence_new = []
    Smiles_new = []
    ECNumber_new = []
    Substrate_new = []
    Type_new = []
    Temp_k_norm_new = []
    Inv_Temp_norm_new = []
    Temp_new = []
    Temp_k_new = []
    Inv_Temp_new = []
    for i in range(len(Label)):
        if -10000000000 < Label[i] and '.' not in Smiles[i]:
            feature_T = np.concatenate((feature[i], [Temp_k_norm[i]], [Inv_Temp_norm[i]]), axis=0)
            feature_new.append(feature_T)
            Label_new.append(Label[i])
            sequence_new.append(sequence[i])
            Smiles_new.append(Smiles[i])
            ECNumber_new.append(ECNumber[i])
            Substrate_new.append(Substrate[i])
            Type_new.append(Type[i])
            Temp_k_norm_new.append(Temp_k_norm[i])
            Inv_Temp_norm_new.append(Inv_Temp_norm[i])
            Temp_new.append(Temp[i])
            Temp_k_new.append(Temp_k[i])
            Inv_Temp_new.append(Inv_Temp[i])
    print(len(Label_new), min(Label_new), max(Label_new))
    Label_new = np.array(Label_new)
    feature_new = np.array(feature_new)
    print(feature_new.shape)

    # Modelling
    Kcat_predict(feature_new, Label_new, sequence_new, Smiles_new, ECNumber_new, Substrate_new, Type_new,
                 Temp_k_norm_new, Inv_Temp_norm_new, Temp_new, Temp_k_new, Inv_Temp_new)
