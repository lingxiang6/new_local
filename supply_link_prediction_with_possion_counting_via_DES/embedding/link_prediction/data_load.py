import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score
import argparse
import os
import random
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data
from embedding import EmbeddingGenerator
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import DataLoader
# 定义一个DataModule来管理数据
class GraphDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.loaction_numpy,self.supplier_numpy,self.customer_numpy = self.embedding(args)
        self.A = pd.read_csv(args.A_Matrix).values
        location_tensor = torch.tensor(self.loaction_numpy, dtype=torch.float).unsqueeze(-1)
        supplier_tensor = torch.tensor(self.supplier_numpy, dtype=torch.float).unsqueeze(-1)
        customer_tensor = torch.tensor(self.customer_numpy, dtype=torch.float).unsqueeze(-1)
        # 合并特征矩阵
        x = torch.cat([location_tensor, supplier_tensor, customer_tensor], dim=-1)
        # 将邻接矩阵转换为边索引格式 (edge_index)
        edge_index = torch.tensor(np.array(self.A).T, dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)
        transform = RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=False)
        # 划分数据集
        self.train_data, self.val_data, self.test_data = transform(data)

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.args.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.args.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.args.batch_size)

    def embedding(self,args):
        # embedding
        data = pd.read_csv(args.data_path)
        data=data.iloc[:,:]
        names = data['related_orig'].tolist()
        statement=data['statement'].tolist()
        location_embedding = EmbeddingGenerator(args.model_path,prompt=args.location_prompt, batch_size=args.LLM_batch_size)
        supplier_embedding = EmbeddingGenerator(args.model_path,prompt=args.supplier_prompt, batch_size=args.LLM_batch_size)
        customer_embedding = EmbeddingGenerator(args.model_path,prompt=args.customer_prompt, batch_size=args.LLM_batch_size)
        # 输入形式是列表，返回是一个超大的numpy数组
        def load_or_save_embedding(embedding, prompt, model_path, file_name, input_data):
            # 获取 model_path 的最后一个目录名
            file=prompt+'.npy'
            model_name = os.path.basename(model_path.rstrip('/'))  # 去除尾部斜杠
            checkpoint_path = os.path.join('link_prediction/checkpoint', model_name, file_name,file)
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            # checkpoint_path = os.path.join(checkpoint_path, prompt)
            if os.path.exists(checkpoint_path):
                return np.load(checkpoint_path)
            else:
                numpy_array = embedding.encode(input_data)
                np.save(checkpoint_path, numpy_array)
                return numpy_array
        location_numpy = load_or_save_embedding(location_embedding, 
                                                args.location_prompt,
                                                args.model_path, 
                                                'location_embedding',
                                                names)
        # 处理 supplier embedding
        supplier_numpy = load_or_save_embedding(supplier_embedding, 
                                                args.supplier_prompt, 
                                                args.model_path,
                                                 'supplier_embedding',
                                                  statement)
        # 处理 customer embedding
        customer_numpy = load_or_save_embedding(customer_embedding,
                                                 args.customer_prompt, 
                                                 args.model_path,
                                                  'customer_embedding', 
                                                  statement)
        return location_numpy, supplier_numpy, customer_numpy



