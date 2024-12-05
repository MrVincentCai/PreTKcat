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
# from MPG_util.graph_bert import *
# from MPG_util.mol2graph import *
from sklearn.model_selection import KFold
# from hyperopt import fmin, tpe, rand, hp, Trials
from rdkit import Chem
from rdkit.Chem import AllChem
from Extract_feature import Get_features
from Pubchem_FIngerprint import GetPubChemFPs

def compute_compound_feature_(compound, fp_type):
    fp = []
    mol = Chem.MolFromSmiles(compound)
    if fp_type == 'mixed':
        fp_maccs = AllChem.GetMACCSKeysFingerprint(mol)
        fp_phaErGfp = AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3, maxPath=21, minPath=1)
        fp_pubcfp = GetPubChemFPs(mol)
        fp.extend(fp_maccs)
        fp.extend(fp_phaErGfp)
        fp.extend(fp_pubcfp)
    else:
        fp_morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        fp.extend(fp_morgan)

    fp = np.array(fp).reshape(1, -1)

    return fp

def smiles_to_vec(Smiles):
    SMILES_vector_list = []
    i = 1
    for smiles in Smiles:
        fp = compute_compound_feature_(smiles, 'mixed')
        SMILES_vector_list.append(fp)
        i = i + 1
    return np.vstack(SMILES_vector_list)



def Seq_to_vec(Sequence):
    shortest_seq = min(Sequence, key=len)
    longest_seq = max(Sequence, key=len)
    shortest_length = len(shortest_seq)  # 10
    longest_length = len(longest_seq)  # 3391
    print(shortest_length)
    print(longest_length)
    features_crafted = Get_features(Sequence, shortest_length)
    print("features_crafted shape:", features_crafted.shape)

    return features_crafted






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

        res.to_csv(f'PreKcat_new/K_Fold/PreTKcat/{fold}_PreTKcat_DLTKcat_fp_craft.csv', index=False)

        # Predict on the test set
        Pre_test_label = model.predict(Test_data)

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


    smiles_input = smiles_to_vec(Smiles)
    sequence_input = Seq_to_vec(sequence)

    feature = np.concatenate((smiles_input, sequence_input), axis=1)
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
    Kcat_predict(feature_new, Label_new, sequence_new, Smiles_new, ECNumber_new, Substrate_new, Type_new,
                 Temp_k_norm_new, Inv_Temp_norm_new, Temp_new, Temp_k_new, Inv_Temp_new)



