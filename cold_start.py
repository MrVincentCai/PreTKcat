# from os.path import join
# import torch
# from sklearn.linear_model import LinearRegression
# from sklearn.neural_network import MLPRegressor
# from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
# import xgboost as xgb
# from build_vocab import WordVocab
# from pretrain_trfm import TrfmSeq2seq
# from utils import split
# import json
# from transformers import T5EncoderModel, T5Tokenizer
# from transformers import AutoTokenizer, EsmModel, EsmTokenizer
# from transformers import BertModel, BertTokenizer
# import re
# import gc
# from sklearn import metrics
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
# from sklearn.model_selection import train_test_split
import random
import pickle
import math
import os
from sklearn.model_selection import KFold
# from MPG_util.graph_bert import *
# from MPG_util.mol2graph import *
# from sklearn.model_selection import KFold
# from hyperopt import fmin, tpe, rand, hp, Trials


# def smiles_to_vec(Smiles):
#     tokenizer = BertTokenizer.from_pretrained('SMILESBERT')
#     model = BertModel.from_pretrained('SMILESBERT')
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)
#     model = model.eval()
#     with torch.no_grad():
#         SMILES_vector_list = []
#         i = 1
#         for smiles in Smiles:
#             tokens = tokenizer(smiles, return_tensors='pt', truncation=True, padding='max_length', max_length=512).to(device)
#             predictions = model(**tokens).pooler_output.cpu().numpy()
#             # print("smiles len:", len(smiles))
#             # print("predictions shape:", predictions.shape) # [1, 768]
#             SMILES_vector_list.append(predictions)
#
#             print(f"已处理第{i}个分子", i)
#             i = i + 1
#     return SMILES_vector_list

# def smiles_to_vec(Smiles):
#     pad_index = 0
#     unk_index = 1
#     eos_index = 2
#     sos_index = 3
#     mask_index = 4
#     vocab = WordVocab.load_vocab('vocab.pkl')
#     def get_inputs(sm):
#         seq_len = 220
#         sm = sm.split()
#         if len(sm)>218:
#             # print('SMILES is too long ({:d})'.format(len(sm)))
#             sm = sm[:109]+sm[-109:]
#         ids = [vocab.stoi.get(token, unk_index) for token in sm]
#         ids = [sos_index] + ids + [eos_index]
#         seg = [1]*len(ids)
#         padding = [pad_index]*(seq_len - len(ids))
#         ids.extend(padding), seg.extend(padding)
#         return ids, seg
#     def get_array(smiles):
#         x_id, x_seg = [], []
#         for sm in smiles:
#             a,b = get_inputs(sm)
#             x_id.append(a)
#             x_seg.append(b)
#         return torch.tensor(x_id), torch.tensor(x_seg)
#     trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
#     trfm.load_state_dict(torch.load('trfm_12_23000.pkl'))
#     trfm.eval()
#     x_split = [split(sm) for sm in Smiles]
#     xid, xseg = get_array(x_split)
#     X = trfm.encode(torch.t(xid))
#     return X

# def graph_to_vec(Smiles):
#     MolEncoder = MolGT(num_layer=5, emb_dim=768, heads=12, num_message_passing=3, drop_ratio=0.5)
#     MolEncoder.load_state_dict(torch.load('../DLKcat/DeeplearningApproach/Code/model/DLKcat_Multi/pretrained_model/MolGNet.pt', map_location=torch.device('cpu')))
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     MolEncoder = MolEncoder.to(device)
#     MolEncoder = MolEncoder.eval()
#     Mol_vector_list = []
#     i = 1
#     with torch.no_grad():
#         for smiles in Smiles:
#             mol = Chem.MolFromSmiles(smiles)
#             data = mol_to_graph_data_dic(mol).to(device)
#             # print("data: ", data)
#             # print("data dummy_node_indices: ", data.dummy_node_indices)
#             Mol_vector = MolEncoder(data)  # [19, 768]
#             batch = torch.tensor(np.zeros(data.x.size(0), dtype=np.int64)).to(device)
#             # Mol_vector = global_mean_pool(Mol_vector, batch)
#             # Mol_vector = Mol_vector[data.dummy_node_indices]
#             # Mol_vector = global_max_pool(Mol_vector, batch)
#             Mol_vector = global_add_pool(Mol_vector, batch)
#
#             # print("shape of Mol_vector:", Mol_vector.shape)  # [1, 768]
#             # print("Mol_vector:", Mol_vector)
#
#             Mol_vector = Mol_vector.detach().cpu().numpy()
#             # print("type of Mol_vector:", type(Mol_vector))
#             Mol_vector_list.append(Mol_vector)
#             print(f"已处理第{i}个分子", i)
#             i = i + 1
#     return Mol_vector_list

# def Seq_to_vec(Sequence):
#     tokenizer = BertTokenizer.from_pretrained("prot_bert", do_lower_case=False)
#     model = BertModel.from_pretrained("prot_bert")
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)
#     model = model.eval()
#     features = []
#     with torch.no_grad():  # 加上这个可以有效减少所需内存
#         for i in range(len(Sequence)):
#             # 将蛋白质序列大写化
#             protein_sequence = Sequence[i].upper()
#             # 使用空格进行分词
#             protein_sequence = " ".join(protein_sequence)
#             sequence_Example = re.sub(r"[UZOB]", "X", protein_sequence)
#             encoded_input = tokenizer(sequence_Example, return_tensors='pt', truncation=True, padding=True, max_length=1024).to(device)
#             output = model(**encoded_input).pooler_output.squeeze(0).cpu().detach().numpy()
#             # output = torch.mean(last_hidden_states.squeeze(0), dim=0, keepdim=False).cpu().detach().numpy()
#             # print("last_hidden_states:", last_hidden_states)
#             # print("type(last_hidden_states):", type(last_hidden_states))
#             # print("last_hidden_states shape:", last_hidden_states.shape)
#             # print("output shape:", output.shape)  # 1024
#
#             features.append(output)
#
#             print(f'已生成第{i + 1}个ProtBert酶表示')
#     return np.vstack(features)

# def Seq_to_vec(Sequence):
#     tokenizer = EsmTokenizer.from_pretrained("esm2_t33_650M_UR50D")  #
#     # tokenizer = AutoTokenizer.from_pretrained("esm2_t6_8M_UR50D")
#     model = EsmModel.from_pretrained("esm2_t33_650M_UR50D")  # esm1v_t33_650M_UR90S_1  esm2_t33_650M_UR50D
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)
#     model = model.eval()
#     features = []
#     with torch.no_grad():  # 加上这个可以有效减少所需内存
#         for i in range(len(Sequence)):
#             # inputs = tokenizer(Sequence[i], return_tensors="pt")
#             inputs = tokenizer(Sequence[i], return_tensors="pt", truncation=True, padding=True, max_length=1024).to(device)
#             # print(inputs)
#             outputs = model(**inputs).pooler_output.squeeze(0).cpu().detach().numpy()
#             # last_hidden_states = outputs.last_hidden_state
#             # output = torch.mean(last_hidden_states.squeeze(0), dim=0, keepdim=False).cpu().detach().numpy()
#
#             # print("last_hidden_states:", last_hidden_states)
#             # print("type(last_hidden_states):", type(last_hidden_states))
#             # print("last_hidden_states shape:", last_hidden_states.shape)
#             # print("output shape:", outputs.shape)  # 1280
#
#             features.append(outputs)
#
#
#             print(f'已生成第{i + 1}个ESM2酶表示')
#     return np.vstack(features)

# def Seq_to_vec(Sequence):
#     for i in range(len(Sequence)):
#         if len(Sequence[i]) > 1000:
#             Sequence[i] = Sequence[i][:500] + Sequence[i][-500:]
#     sequences_Example = []
#     for i in range(len(Sequence)):
#         zj = ''
#         for j in range(len(Sequence[i]) - 1):
#             zj += Sequence[i][j] + ' '
#         zj += Sequence[i][-1]
#         sequences_Example.append(zj)
#     tokenizer = T5Tokenizer.from_pretrained("prot_t5_xl_uniref50", do_lower_case=False)
#     model = T5EncoderModel.from_pretrained("prot_t5_xl_uniref50")
#     gc.collect()
#     print(torch.cuda.is_available())
#     # 'cuda:0' if torch.cuda.is_available() else
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)
#     model = model.eval()
#     features = []
#     for i in range(len(sequences_Example)):
#         print('For sequence ', str(i+1))
#         sequences_Example_i = sequences_Example[i]
#         sequences_Example_i = [re.sub(r"[UZOB]", "X", sequences_Example_i)]
#         ids = tokenizer.batch_encode_plus(sequences_Example_i, add_special_tokens=True, padding=True)
#         input_ids = torch.tensor(ids['input_ids']).to(device)
#         attention_mask = torch.tensor(ids['attention_mask']).to(device)
#         with torch.no_grad():
#             embedding = model(input_ids=input_ids, attention_mask=attention_mask)
#         embedding = embedding.last_hidden_state.cpu().numpy()
#         for seq_num in range(len(embedding)):
#             seq_len = (attention_mask[seq_num] == 1).sum()
#             seq_emd = embedding[seq_num][:seq_len - 1]
#             features.append(seq_emd)
#             # print("seq_emd shape:", seq_emd.shape)
#     features_normalize = np.zeros([len(features), len(features[0][0])], dtype=float)
#     for i in range(len(features)):
#         for k in range(len(features[0][0])):
#             for j in range(len(features[i])):
#                 features_normalize[i][k] += features[i][j][k]
#             features_normalize[i][k] /= len(features[i])
#     # print("features_normalize shape:", features_normalize.shape)
#     return features_normalize




def split_cold_start(sequence_new, n_splits=10, random_state=42):

    unique_sequences = np.array(list(set(sequence_new)))
    np.random.seed(random_state)
    np.random.shuffle(unique_sequences)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = []

    for train_idx, test_idx in kf.split(unique_sequences):
        train_sequences = unique_sequences[train_idx]
        test_sequences = unique_sequences[test_idx]

        train_idx_all = [i for i, seq in enumerate(sequence_new) if seq in train_sequences]
        test_idx_all = [i for i, seq in enumerate(sequence_new) if seq in test_sequences]
        folds.append((train_idx_all, test_idx_all))

    return folds

def save_fold_indices(folds, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, (train_idx, test_idx) in enumerate(folds):
        with open(os.path.join(output_dir, f"train_idx_fold_{i+1}.pkl"), "wb") as f:
            pickle.dump(train_idx, f)
        with open(os.path.join(output_dir, f"test_idx_fold_{i+1}.pkl"), "wb") as f:
            pickle.dump(test_idx, f)

def load_fold_indices(input_dir, n_splits=10):

    folds = []
    for i in range(1, n_splits + 1):
        with open(os.path.join(input_dir, f"train_idx_fold_{i}.pkl"), "rb") as f:
            train_idx = pickle.load(f)
        with open(os.path.join(input_dir, f"test_idx_fold_{i}.pkl"), "rb") as f:
            test_idx = pickle.load(f)
        folds.append((train_idx, test_idx))
    return folds


def evaluate_and_save_results(model, Test_data, Test_label, output_file):

    Pre_test_label = model.predict(Test_data)

    # 计算性能指标
    r2 = r2_score(Test_label, Pre_test_label)
    rmse = np.sqrt(mean_squared_error(Test_label, Pre_test_label))
    mae = mean_absolute_error(Test_label, Pre_test_label)
    pcc = pearsonr(Test_label, Pre_test_label)[0]

    print(f"R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, PCC: {pcc:.4f}")

    with open(output_file, "a") as f:
        f.write(f"R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, PCC: {pcc:.4f}\n")

    return r2, rmse, mae, pcc

def Kcat_predict(Ifeature, Label, sequence_new, Smiles_new, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    metric_file = os.path.join(output_dir, "evaluation_metrics.txt")

    # folds = split_cold_start(sequence_new, n_splits=5, random_state=42)
    # save_fold_indices(folds, "./PreKcat_new/Cold_5/fold_index/sequence")
    folds = load_fold_indices("./PreKcat_new/Cold_5/fold_index/smiles", n_splits=5)

    for i, (train_idx, test_idx) in enumerate(folds):
        print(f"Processing Fold {i+1}...")
        print("train_idx len:", len(train_idx))
        print("test_idx len:", len(test_idx))

        Train_data = Ifeature[train_idx]
        Test_data = Ifeature[test_idx]
        Train_label = Label[train_idx]
        Test_label = Label[test_idx]


        model = ExtraTreesRegressor(random_state=42)
        model.fit(Train_data, Train_label)

        fold_metric_file = os.path.join(output_dir, f"fold_{i+1}_metrics.txt")
        evaluate_and_save_results(model, Test_data, Test_label, fold_metric_file)

    print(f"All folds processed. Metrics saved to {metric_file}")



if __name__ == '__main__':
    output_dir = "./PreKcat_new/Cold_5/PreTKcat/smiles"
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

    for i in range(len(Label)):
        if Label[i] == 0:
            Label[i] = -10000000000
        else:
            Label[i] = math.log(Label[i], 10)
    print(max(Label), min(Label))


    # feature = pd.read_pickle("PreKcat_new/DLTKcat_features_16249_SMILES_Protein.pkl")
    feature = pd.read_pickle("PreKcat_new/DLTKcat_features_16249_MPG_Protein.pkl")
    print("feature shape:", feature.shape)  # (16249, 1792)

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
            # print("feature_T shape:", feature_T.shape)  # (1794,)
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
    print(feature_new.shape)  # (16249, 1794)

    # Modelling
    Kcat_predict(feature_new, Label_new, sequence_new, Smiles_new, output_dir)



