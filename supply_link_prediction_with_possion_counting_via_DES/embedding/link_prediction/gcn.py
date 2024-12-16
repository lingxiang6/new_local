import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score
# import argparse
import os
import torch
import random
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),  # 假设降维到128维
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)
    
class GCNModel(pl.LightningModule):
    def __init__(self, args):
        super(GCNModel, self).__init__()
        self.args = args
        # 定义GCN层
        self.conv1 = GCNConv(args.num_node_features, args.hidden_dim)
        self.conv2 = GCNConv(args.hidden_dim, args.hidden_dim)
        # 预测链接的全连接层
        self.fc = nn.Linear(args.hidden_dim * 2, 1)

    def forward(self, x, edge_index):
        # GCN的前向传播，传入节点特征和邻接矩阵
        x = self.conv1(x, edge_index)
        x = torch.relu(x)

        
        x = self.conv2(x, edge_index)
        return x
    
    def decode(self, z, pos_edge_index, neg_edge_index):
        # 链接预测，获取正负边的嵌入并计算得分
        pos_scores = self.score_edges(z, pos_edge_index)
        neg_scores = self.score_edges(z, neg_edge_index)
        return pos_scores, neg_scores

    def score_edges(self, z, edge_index):
        # 从节点嵌入中计算边的得分
        row, col = edge_index
        return (z[row] * z[col]).sum(dim=1)

    def training_step(self, batch, batch_idx):
        # 获取批次中的数据
        x, edge_index, pos_edge_index, neg_edge_index = batch
        z = self(x, edge_index)

        pos_scores, neg_scores = self.decode(z, pos_edge_index, neg_edge_index)

        # 链接预测的损失函数：基于正负样本的二元交叉熵
        loss = self.link_prediction_loss(pos_scores, neg_scores)
        self.log('train_loss', loss)
        return loss

    def link_prediction_loss(self, pos_scores, neg_scores):
        # 链接预测的二元交叉熵损失
        pos_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-15).mean()
        neg_loss = -torch.log(1 - torch.sigmoid(neg_scores) + 1e-15).mean()
        return pos_loss + neg_loss

    def configure_optimizers(self):
        # 定义优化器
        optimizer = optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        return optimizer

    def validation_step(self, batch, batch_idx):
        # 验证步骤：计算AUC
        x, edge_index, pos_edge_index, neg_edge_index = batch
        z = self(x, edge_index)

        pos_scores, neg_scores = self.decode(z, pos_edge_index, neg_edge_index)
        val_loss = self.link_prediction_loss(pos_scores, neg_scores)
        
        # 计算AUC
        y_true = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        y_scores = torch.cat([pos_scores, neg_scores])
        auc = roc_auc_score(y_true.cpu().numpy(), y_scores.cpu().detach().numpy())
        
        self.log('val_loss', val_loss)
        self.log('val_auc', auc)
        return val_loss
