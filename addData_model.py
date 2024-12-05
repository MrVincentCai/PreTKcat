
import faulthandler
faulthandler.enable()

# import torch
from MPG_util.graph_bert import *
from MPG_util.mol2graph import *
from torch_geometric.nn import  global_add_pool
import pandas as pd
import numpy as np
import pickle
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from sklearn.ensemble import ExtraTreesRegressor
from build_vocab import WordVocab
from pretrain_trfm import TrfmSeq2seq
from utils import split
from transformers import T5EncoderModel, T5Tokenizer
import re
import gc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from os.path import join
import json
from transformers import AutoTokenizer, EsmModel, EsmTokenizer
from transformers import BertModel, BertTokenizer
import xgboost as xgb
from sklearn import metrics

import random




def smiles_to_vec(Smiles):
    pad_index = 0
    unk_index = 1
    eos_index = 2
    sos_index = 3
    mask_index = 4
    vocab = WordVocab.load_vocab('vocab.pkl')
    def get_inputs(sm):
        seq_len = 220
        sm = sm.split()
        if len(sm)>218:
            # print('SMILES is too long ({:d})'.format(len(sm)))
            sm = sm[:109]+sm[-109:]
        ids = [vocab.stoi.get(token, unk_index) for token in sm]
        ids = [sos_index] + ids + [eos_index]
        seg = [1]*len(ids)
        padding = [pad_index]*(seq_len - len(ids))
        ids.extend(padding), seg.extend(padding)
        return ids, seg
    def get_array(smiles):
        x_id, x_seg = [], []
        for sm in smiles:
            a,b = get_inputs(sm)
            x_id.append(a)
            x_seg.append(b)
        return torch.tensor(x_id), torch.tensor(x_seg)
    trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
    trfm.load_state_dict(torch.load('trfm_12_23000.pkl'))
    trfm.eval()
    x_split = [split(sm) for sm in Smiles]
    xid, xseg = get_array(x_split)
    X = trfm.encode(torch.t(xid))
    return X

def graph_to_vec(Smiles):
    MolEncoder = MolGT(num_layer=5, emb_dim=768, heads=12, num_message_passing=3, drop_ratio=0.5)
    MolEncoder.load_state_dict(torch.load('../DLKcat/DeeplearningApproach/Code/model/DLKcat_Multi/pretrained_model/MolGNet.pt', map_location=torch.device('cpu')))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MolEncoder = MolEncoder.to(device)
    MolEncoder = MolEncoder.eval()
    Mol_vector_list = []
    i = 1
    with torch.no_grad():
        for smiles in Smiles:
            mol = Chem.MolFromSmiles(smiles)
            data = mol_to_graph_data_dic(mol).to(device)
            # print("data: ", data)
            # print("data dummy_node_indices: ", data.dummy_node_indices)
            Mol_vector = MolEncoder(data)  # [19, 768]
            batch = torch.tensor(np.zeros(data.x.size(0), dtype=np.int64)).to(device)
            # Mol_vector = global_mean_pool(Mol_vector, batch)
            # Mol_vector = Mol_vector[data.dummy_node_indices]
            # Mol_vector = global_max_pool(Mol_vector, batch)
            Mol_vector = global_add_pool(Mol_vector, batch)

            # print("shape of Mol_vector:", Mol_vector.shape)  # [1, 768]
            # print("Mol_vector:", Mol_vector)

            Mol_vector = Mol_vector.detach().cpu().numpy()
            # print("type of Mol_vector:", type(Mol_vector))
            Mol_vector_list.append(Mol_vector)
            print(f"已处理第{i}个分子", i)
            i = i + 1
    return Mol_vector_list





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
    # print("features_normalize shape:", features_normalize.shape)
    return features_normalize





def Kcat_predict(feature, label):
    train_data, test_data, train_label, test_label = train_test_split(
        feature, label, test_size=0.1, random_state=42
    )
    print("train_data shape:", train_data.shape)
    print("test_data shape:", test_data.shape)

    sub_train_size = len(train_data) // 3
    subset1 = train_data[:sub_train_size]
    subset2 = train_data[sub_train_size:2 * sub_train_size]
    subset3 = train_data[2 * sub_train_size:]
    subset1_label = train_label[:sub_train_size]
    subset2_label = train_label[sub_train_size:2 * sub_train_size]
    subset3_label = train_label[2 * sub_train_size:]

    subsets = [
        (subset1, subset1_label, "Subset 1"),
        (np.concatenate((subset1, subset2), axis=0), np.concatenate((subset1_label, subset2_label)), "Subset 1+2"),
        (train_data, train_label, "All Subsets"),
    ]


    r2_scores, rmse_scores, mae_scores = [], [], []

    for train_set, train_set_label, subset_name in subsets:

        print("train_set shape:", train_set.shape)

        model = ExtraTreesRegressor(random_state=42)
        model.fit(train_set, train_set_label)

        test_pred = model.predict(test_data)

        r2 = r2_score(test_label, test_pred)
        rmse = np.sqrt(mean_squared_error(test_label, test_pred))
        mae = mean_absolute_error(test_label, test_pred)
        pcc = pearsonr(test_label, test_pred)[0]

        r2_scores.append(r2)
        rmse_scores.append(rmse)
        mae_scores.append(mae)

        print(f"{subset_name}: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, PCC={pcc:.4f}")





def scale_minmax(data):
    data_min = min(data)
    data_max = max(data)
    return [(x - data_min) / (data_max - data_min) for x in data]

if __name__ == '__main__':

    # Dataset Load
    datasets = np.array(pd.read_csv('./datasets/DLTKcat_data/kcat_merge_DLTKcat.csv')).T
    # datasets = datasets[:, :1000]
    print(datasets.shape)
    sequence = [data for data in datasets[10]]
    # print(sequence[0])
    Smiles = [data for data in datasets[9]]
    # print(Smiles[0])
    Label = [float(data) for data in datasets[5]]
    # print(Label[0])
    ECNumber = [data for data in datasets[0]]
    # print(ECNumber[0])
    Substrate = [data for data in datasets[1]]
    # print(Substrate[0])
    Type = [data for data in datasets[3]]
    # print(Type[0])
    Temp = [data for data in datasets[6]]
    Temp_k = [data for data in datasets[7]]
    Inv_Temp = [data for data in datasets[8]]
    Temp_k_norm = [data for data in datasets[13]]
    Inv_Temp_norm = [data for data in datasets[14]]

    # Dataset Load
    datasets = np.array(pd.read_csv('independent_test_unique.csv')).T
    sequence_add = [data for data in datasets[9]]
    Smiles_add = [data for data in datasets[8]]
    Label_add = [float(data) for data in datasets[4]]
    ECNumber_add = [data for data in datasets[0]]
    Substrate_add = [data for data in datasets[3]]
    Type_add = [data for data in datasets[1]]
    Temp_add = [data for data in datasets[7]]
    # 转换为开尔文单位
    Temp_k_add = [t + 273.15 for t in Temp_add]
    # 计算倒数
    Inv_Temp_add = [1 / t for t in Temp_k_add]
    # 归一化
    Temp_k_norm_add = scale_minmax(Temp_k_add)
    Inv_Temp_norm_add = scale_minmax(Inv_Temp_add)

    sequence.extend(sequence_add)
    Smiles.extend(Smiles_add)
    Label.extend(Label_add)
    ECNumber.extend(ECNumber_add)
    Substrate.extend(Substrate_add)
    Type.extend(Type_add)
    Temp.extend(Temp_add)
    Temp_k.extend(Temp_k_add)
    Inv_Temp.extend(Inv_Temp_add)
    Temp_k_norm.extend(Temp_k_norm_add)
    Inv_Temp_norm.extend(Inv_Temp_norm_add)

    print(len(Label))  # 25249



    for i in range(len(Label)):
        if Label[i] == 0:
            Label[i] = -10000000000
        else:
            Label[i] = math.log(Label[i], 10)
    print(max(Label), min(Label))
    # Feature Extractor
    smiles_input = pd.read_pickle('PreKcat_new/DLTKcat_addData_Compound_MPG.pkl')
    # smiles_input = graph_to_vec(Smiles)
    # smiles_input = np.vstack(smiles_input)
    # print(smiles_input.shape)
    # with open("PreKcat_new/DLTKcat_addData_Compound_MPG.pkl", "wb") as f:
    #     pickle.dump(smiles_input, f)

    # smiles_input = smiles_to_vec(Smiles)
    # with open("PreKcat_new/DLTKcat_addData_Compound_SMILES.pkl", "wb") as f:
    #     pickle.dump(smiles_input, f)

    sequence_input = pd.read_pickle('PreKcat_new/DLTKcat_addData_Protein_proT5.pkl')
    # sequence_input = Seq_to_vec(sequence)
    # print("protein shape:", sequence_input.shape)
    # with open("PreKcat_new/DLTKcat_addData_Protein_proT5.pkl", "wb") as f:
    #     pickle.dump(sequence_input, f)

    feature_add = np.concatenate((smiles_input, sequence_input), axis=1)
    feature = pd.read_pickle("PreKcat_new/DLTKcat_features_16249_MPG_Protein.pkl")
    feature = np.concatenate((feature, feature_add), axis=0)
    print("feature shape:", feature.shape)  # (25429, 1792)

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
            # print("i:", i)
            # print("Temp_k_norm:", [Temp_k_norm[i]])
            # print("Inv_Temp_norm:", [Inv_Temp_norm[i]])
            feature_T = np.concatenate((feature[i], [Temp_k_norm[i]], [Inv_Temp_norm[i]]), axis=0)
            # print("feature_T shape:", feature_T.shape)
            feature_new.append(feature_T)

            # feature_new.append(feature[i])
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
    print(feature_new.shape)  # (25429, 1794)

    # Modelling
    Kcat_predict(feature_new, Label_new)
