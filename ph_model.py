
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





def Kcat_predict(Ifeature, Label, sequence_new, Smiles_new, ECNumber_new, Substrate_new, Type_new, Temp_k_norm_new, Inv_Temp_norm_new, Temp_new, Temp_k_new, Inv_Temp_new):
    kf = KFold(n_splits=10, shuffle=True, random_state=2024)

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


        res.to_csv(f'PreKcat_new/K_Fold/PreTKcat/ph/{fold}_PreTKcat_DLTKcat_temp_ph.csv', index=False)

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
    datasets = np.array(pd.read_csv('kcat_merge_DLTKcat_with_pH.csv')).T
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
    ph = [float(data) for data in datasets[20]]

    # Min-Max 归一化
    ph_min = min(ph)
    ph_max = max(ph)
    ph_norm = [(x - ph_min) / (ph_max - ph_min) for x in ph]
    # print(ph_norm[0])
    # sss
    for i in range(len(Label)):
        if Label[i] == 0:
            Label[i] = -10000000000
        else:
            Label[i] = math.log(Label[i], 10)
    print(max(Label), min(Label))
    # Feature Extractor
    smiles_input = graph_to_vec(Smiles)

    # smiles_input = smiles_to_vec(Smiles)

    sequence_input = Seq_to_vec(sequence)


    feature = np.concatenate((smiles_input, sequence_input), axis=1)
    print("feature shape:", feature.shape)  # (6305, 1792)

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
            # feature_T = np.concatenate((feature[i], [Temp_k_norm[i]], [Inv_Temp_norm[i]]), axis=0)
            # print("feature_T shape:", feature_T.shape)  # (1793,)
            # feature_new.append(feature_T)

            # feature_ph = np.concatenate((feature[i], [ph_norm[i]]), axis=0)
            # print("feature_ph shape:", feature_ph.shape)  # (1794,)
            # feature_new.append(feature_ph)

            feature_T_ph = np.concatenate((feature[i], [Temp_k_norm[i]], [Inv_Temp_norm[i]], [ph_norm[i]]), axis=0)
            feature_new.append(feature_T_ph)

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
    print(feature_new.shape)

    # Modelling
    Kcat_predict(feature_new, Label_new, sequence_new, Smiles_new, ECNumber_new, Substrate_new, Type_new,
                 Temp_k_norm_new, Inv_Temp_norm_new, Temp_new, Temp_k_new, Inv_Temp_new)



