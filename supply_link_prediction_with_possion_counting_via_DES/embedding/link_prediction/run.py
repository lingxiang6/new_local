import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score
import argparse
import os
import torch
import random
import numpy as np
from data_load import GraphDataModule
from gcn import GCNModel

parser = argparse.ArgumentParser(description='link prediction')

parser.add_argument('--random_seed', type=int, default=2024, help='random seed')

parser.add_argument('--hidden_dim', type=int, default=1, help='')
parser.add_argument('--num_node_features', type=int, default=3068, help='num_node_features')
parser.add_argument('--batch_size', type=int, default=1, help='')
parser.add_argument('--LLM_batch_size', type=int, default=20, help='')
parser.add_argument('--epochs', type=int, default=1, help='')
parser.add_argument('--lr', type=int, default=1, help='')

parser.add_argument('--data_path', type=str, default='/your/root', help='')
parser.add_argument('--A_Matrix', type=str, default='/your/root', help='')
parser.add_argument('--model_path', type=str, default='/your/root', help='')

parser.add_argument('--location_prompt', type=str, default="根据输入的公司名称，输出对应的省市名称,注意：直辖市回复到市，英文公司用中文回答所在国家", help='')
parser.add_argument('--supplier_prompt', type=str, default="该公司的位置在哪里？", help='')
parser.add_argument('--customer_prompt', type=str, default="该公司是一个什么样的公司,他的主要需要哪些商品", help='')

# parser.add_argument('--after_prompt', type=str, default="After thinking step by step, summarize this sentence", help='')
args = parser.parse_args()
# random seed
fix_seed = args.random_seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

data=GraphDataModule(args)
model=GCNModel(args)
# 初始化和训练
def train_model(node_features_df, adj_matrix):

    num_node_features = node_features_df.shape[1]
    # 初始化模型和数据模块
    model = GCNModel(args, num_node_features)
    data_module = GraphDataModule(args, node_features_df, adj_matrix)
    # 使用PyTorch Lightning的Trainer进行训练
    trainer = pl.Trainer(max_epochs=args.epochs)
    trainer.fit(model, data_module)


