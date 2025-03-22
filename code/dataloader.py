import pandas as pd
import networkx as nx
import json, pickle
from collections import OrderedDict
from rdkit import Chem
from utils import *
from tqdm import tqdm
import numpy as np
import torch
def read_data(filename):
    df = pd.read_csv('data/' + filename)
    drugs, prots, Y = list(df['compound_iso_smiles']),list(df['target_sequence']),list(df['affinity'])
    return drugs, prots, Y

def dic_normalize(dic):
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    # print(max_value)
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic

VOCAB_LIGAND_ISO = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
				"1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
				"9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
				"D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
				"O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
				"V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
				"b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
				"l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                 'X']
seq_dict = {v:(i+1) for i,v in enumerate(pro_res_table)}
pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']

res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}
res_weight_table['X'] = np.average([res_weight_table[k] for k in res_weight_table.keys()])

res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}
res_pka_table['X'] = np.average([res_pka_table[k] for k in res_pka_table.keys()])

res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}
res_pkb_table['X'] = np.average([res_pkb_table[k] for k in res_pkb_table.keys()])

res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}
res_pkx_table['X'] = np.average([res_pkx_table[k] for k in res_pkx_table.keys()])

res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}
res_pl_table['X'] = np.average([res_pl_table[k] for k in res_pl_table.keys()])

res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}
res_hydrophobic_ph2_table['X'] = np.average([res_hydrophobic_ph2_table[k] for k in res_hydrophobic_ph2_table.keys()])

res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}
res_hydrophobic_ph7_table['X'] = np.average([res_hydrophobic_ph7_table[k] for k in res_hydrophobic_ph7_table.keys()])

# nomarlize the residue feature
res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)

def one_hot_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

# one ont encoding with unknown symbol
def one_hot_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def encoding_unk(x, allowable_set):
    list = [False for i in range(len(allowable_set))]
    i = 0
    for atom in x:
        if atom in allowable_set:
            list[allowable_set.index(atom)] = True
            i += 1
    if i != len(x):
        list[-1] = True
    return list

def seq_feature(seq):
    residue_feature = []
    for residue in seq:
        if residue not in pro_res_table:
            residue = 'X'
        res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                         1 if residue in pro_res_polar_neutral_table else 0,
                         1 if residue in pro_res_acidic_charged_table else 0,
                         1 if residue in pro_res_basic_charged_table else 0]
        res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue],
                         res_pkx_table[residue],
                         res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
        residue_feature.append(res_property1 + res_property2)

    pro_hot = np.zeros((len(seq), len(pro_res_table)))
    pro_property = np.zeros((len(seq), 12))
    for i in range(len(seq)):
        pro_hot[i,] = one_hot_encoding_unk(seq[i], pro_res_table)
        pro_property[i,] = residue_feature[i]
    seq_feature = np.concatenate((pro_hot, pro_property), axis=1)
    return seq_feature

def atom_features(atom):
    # 44 +11 +11 +11 +1
    return np.array(one_hot_encoding_unk(atom.GetSymbol(),
                                         ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                          'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                          'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                          'Pt', 'Hg', 'Pb', 'X']) +
                    one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_hot_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])

# mol smile to mol graph edge index
def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    mol_size = mol.GetNumAtoms()

    mol_features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        mol_features.append(feature / sum(feature))
    edges = []
    mol_features = torch.Tensor(np.array(mol_features))
    
    # 步骤1: 对原始矩阵A进行SVD分解
    U, S, V = torch.svd(mol_features)
    # 步骤2: 对U和V进行小的扰动
    epsilon = 1e-3  # 扰动的大小
    U_perturbed = U + torch.randn_like(U) * epsilon
    V_perturbed = V + torch.randn_like(V) * epsilon
    # 确保U'和V'仍然是正交的
    U_perturbed = torch.nn.functional.normalize(U_perturbed, p=2, dim=0)
    V_perturbed = torch.nn.functional.normalize(V_perturbed, p=2, dim=0)
    x_diff_mol1 = torch.mm(torch.mm(U_perturbed, torch.diag(S)), V_perturbed.t())
    U_perturbed = U + torch.randn_like(U) * epsilon
    V_perturbed = V + torch.randn_like(V) * epsilon
    U_perturbed = torch.nn.functional.normalize(U_perturbed, p=2, dim=0)
    V_perturbed = torch.nn.functional.normalize(V_perturbed, p=2, dim=0)
    x_diff_mol2 = torch.mm(torch.mm(U_perturbed, torch.diag(S)), V_perturbed.t())
    mol_features = mol_features.numpy()
    x_diff_mol1 = x_diff_mol1.numpy()
    x_diff_mol2 = x_diff_mol2.numpy()
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    return mol_size, mol_features, edge_index, x_diff_mol1, x_diff_mol2

# target sequence to target graph
def sequence_to_graph(target_key, target_sequence, distance_dir):
    target_edge_index = []
    target_edge_distance = []
    target_size = len(target_sequence)
    contact_map_file = os.path.join(distance_dir, target_key + '.npy')
    distance_map = np.load(contact_map_file)

    for i in range(target_size):
        distance_map[i, i] = 1
        if i + 1 < target_size:
            distance_map[i, i + 1] = 1
    
    index_row, index_col = np.where(distance_map >= 0.5)  # for threshold
    for i, j in zip(index_row, index_col):
        target_edge_index.append([i, j])  # dege
        target_edge_distance.append(distance_map[i, j])  # edge weight
    target_feature = seq_feature(target_sequence)
    
    
    # 奇异值分解
    target_edge_index_svd = []
    target_edge_distance_svd = []
    # 使用 torch.svd_lowrank 进行低秩 SVD 分解，这里我们选择保留 0.25
    adj_matrix = torch.tensor(distance_map, dtype=torch.float32)
    q = int(target_size * 0.3)
    U, S, Vh = torch.svd_lowrank(adj_matrix, q=q)
    # 重构邻接矩阵
    reconstructed_adj_matrix = torch.mm(torch.mm(U, torch.diag(S)), Vh.t())
    reconstructed_adj_matrix = np.array(reconstructed_adj_matrix)
    index_row, index_col = np.where(reconstructed_adj_matrix >= 0.5)  # for threshold
    for i, j in zip(index_row, index_col):
        target_edge_index_svd.append([i, j])  # dege
        if reconstructed_adj_matrix[i, j]<0:
            target_edge_distance_svd.append(0)  # edge weight
        else:
            target_edge_distance_svd.append(reconstructed_adj_matrix[i, j])
        
    return target_size, target_feature, target_edge_index, target_edge_distance, target_edge_index_svd, target_edge_distance_svd

# 构建邻接矩阵和特征矩阵
def cluster_graph(mol):
    n_atoms = mol.GetNumAtoms()  #获得分子图中原子数
    cliques = []  # 所有的的边（化学键和一对原子组成）和环
    for bond in mol.GetBonds():  #获得分子图中所有化学键
        a1 = bond.GetBeginAtom().GetIdx() #获得开始原子索引
        a2 = bond.GetEndAtom().GetIdx() #获得结束原子索引
        if not bond.IsInRing():   #判断化学键是否在环中
            cliques.append([a1, a2])  #将边加入到集合中

    ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]  # 获得分子图中的所有环
    cliques.extend(ssr)  #
    # cliques 所有的的边（化学键和一对原子组成）和环
    # nei_list为原子属于哪个基团\子结构
    nei_list = [[] for i in range(n_atoms)]
    for i in range(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)

    edges = []
    for i in range(len(cliques)-1):
        for j in range(i+1,len(cliques)):
            if len(set(cliques[i]) & set(cliques[j]))!= 0:
                edges.append([i,j])
                edges.append([j,i])
    return cliques, edges



def clique_features(clique, edges, clique_idx, smile):
    NumAtoms = len(clique)  # 子结构中原子数
    NumEdges = 0  # 与子结构所连的边数，子结构的度
    for edge in edges:
        if clique_idx == edge[0] or clique_idx == edge[1]:
            NumEdges += 1
    mol = Chem.MolFromSmiles(smile)
    atoms = []
    NumHs = 0  # 基团中氢原子的个数
    NumImplicitValence = 0
    for idx in clique:
        atom = mol.GetAtomWithIdx(idx)
        atoms.append(atom.GetSymbol())
        NumHs += atom.GetTotalNumHs()
        NumImplicitValence += atom.GetImplicitValence()
    # 基团中是否包含环
    IsRing = 0
    if len(clique) > 2:
        IsRing = 1
    # 基团中是否有键
    IsBond = 0
    if len(clique) == 2:
        IsBond = 1
    return np.array(encoding_unk(atoms,
                                         ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                          'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                          'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                          'Pt', 'Hg', 'Pb', 'X']) +
                    one_hot_encoding_unk(NumAtoms, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_hot_encoding_unk(NumEdges, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_hot_encoding_unk(NumHs, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]) +
                    one_hot_encoding_unk(NumImplicitValence, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]) +
                    [IsRing] +
                    [IsBond])

def clique_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    clique, edge = cluster_graph(mol)

    c_features = []  # 特征矩阵
    for idx in range(len(clique)):
        cq_features = clique_features(clique[idx], edge, idx, smile)
        c_features.append(cq_features / sum(cq_features))

    clique_size = len(clique)  # 子结构图节点数
    return clique_size, c_features, edge

# 加入蛋白质序列
seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000
def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]
    return x

# 加入药物序列
import numpy as np
CHAR_SMI_SET = {"(": 1, ".": 2, "0": 3, "2": 4, "4": 5, "6": 6, "8": 7, "@": 8,
                "B": 9, "D": 10, "F": 11, "H": 12, "L": 13, "N": 14, "P": 15, "R": 16,
                "T": 17, "V": 18, "Z": 19, "\\": 20, "b": 21, "d": 22, "f": 23, "h": 24,
                "l": 25, "n": 26, "r": 27, "t": 28, "#": 29, "%": 30, ")": 31, "+": 32,
                "-": 33, "/": 34, "1": 35, "3": 36, "5": 37, "7": 38, "9": 39, "=": 40,
                "A": 41, "C": 42, "E": 43, "G": 44, "I": 45, "K": 46, "M": 47, "O": 48,
                "S": 49, "U": 50, "W": 51, "Y": 52, "[": 53, "]": 54, "a": 55, "c": 56,
                "e": 57, "g": 58, "i": 59, "m": 60, "o": 61, "s": 62, "u": 63, "y": 64}
max_seq_len_drug = 46
def seq_drug(drug):
    x = np.zeros(max_seq_len_drug)
    for i, ch in enumerate(drug[:max_seq_len_drug]):
        x[i] = CHAR_SMI_SET[ch]
    return x

def create_DTA_dataset(dataset='davis'):
    
    # 加载超图
    # 从文件加载列表数据
    with open(dataset + '_hyper.json', 'r') as f:
        hyper_graph_values = json.load(f)
    # 从文件加载列表数据
    with open(dataset + '_smile.json', 'r') as f:
        smiles_list = json.load(f)
    hyper_graph_dict = {}
    for smiles, hyper_graph in zip(smiles_list, hyper_graph_values):
        hyper_graph_dict[smiles] = np.array(hyper_graph)
    
    
    dataset_dir = os.path.join('data', dataset)
    # drug smiles
    ligands = json.load(open(os.path.join(dataset_dir, 'ligands_can.txt')), object_pairs_hook=OrderedDict)
    # protein sequences
    proteins = json.load(open(os.path.join(dataset_dir, 'proteins.txt')), object_pairs_hook=OrderedDict)

    # load protein feature and predicted distance map
    process_dir = os.path.join('./', 'pre_process')
    pro_distance_dir = os.path.join(process_dir, dataset, 'sequence_representations')  # numpy .npy file

    # dataset process
    drugs = []  # rdkit entity
    prots = []  # sequences
    prot_keys = []  # protein id (or name)
    drug_smiles = []  # smiles
    # smiles
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        drugs.append(lg)
        drug_smiles.append(ligands[d])
    # seqs
    for t in proteins.keys():
        prots.append(proteins[t])
        prot_keys.append(t)

    smile_graph = {}
    for i in tqdm(range(len(drugs))):
        smile = drugs[i]
        g_d = smile_to_graph(smile)
        smile_graph[smile] = g_d

    target_graph = {}
    for i in tqdm(range(len(prot_keys))):
        key = prot_keys[i]
        protein = prots[i]
        g_t = sequence_to_graph(key, protein, pro_distance_dir)
        target_graph[protein] = g_t

    # read files(train and test)
    train_csv = dataset + '/' + dataset + '_train.csv'
    test_csv = dataset + '/' + dataset + '_test.csv'
    train_drugs, train_prots, train_Y = read_data(train_csv)
    test_drugs, test_prots, test_Y = read_data(test_csv)

    train_drugs, train_prots, train_Y = np.asarray(train_drugs), np.asarray(train_prots), np.asarray(train_Y)
    XT_train = [seq_cat(t) for t in train_prots] # 将蛋白质序列进行编码
    drug_seq_train = [seq_drug(t) for t in train_drugs] # 将药物序列进行编码
    
    test_drugs, test_prots, test_Y = np.asarray(test_drugs), np.asarray(test_prots), np.asarray(test_Y)
    XT_test = [seq_cat(t) for t in test_prots] # 将蛋白质序列进行编码
    drug_seq_test = [seq_drug(t) for t in test_drugs] # 将药物序列进行编码
    
    train_data = DTADataset(root='data', dataset=dataset + '_' + 'train', drug_smiles=train_drugs, target_sequence=train_prots, y=train_Y,
                            smile_graph=smile_graph,target_graph=target_graph, hyper_graph_dict=hyper_graph_dict, target_seq_cut=XT_train, mol_seq_cut=drug_seq_train)
    
    test_data = DTADataset(root='data', dataset=dataset + '_' + 'test', drug_smiles=test_drugs, target_sequence=test_prots, y=test_Y,
                           smile_graph=smile_graph, target_graph=target_graph, hyper_graph_dict=hyper_graph_dict, target_seq_cut=XT_test, mol_seq_cut=drug_seq_test)
    return train_data, test_data




