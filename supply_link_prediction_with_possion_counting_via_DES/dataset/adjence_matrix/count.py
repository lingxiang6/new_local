import pandas as pd
import numpy as np
import json

# 读取数据
trade = pd.read_csv('supply_chain_trade.csv')
related_data = pd.read_csv('end_feature.csv')

# 获取相关ID
related_ids = related_data['related_id']

# 过滤供应链上的主要节点
trade = trade[
    (trade['supplier'].isin(related_ids)) & 
    (trade['customer'].isin(related_ids))
]

# 创建节点到索引的映射
node_index = {node: idx for idx, node in enumerate(related_ids)}
with open('adjence_matrix/node_index.json', 'w') as f:
    json.dump(node_index, f)

# 获取所有报告期
report_periods = trade['rpt'].unique()
num_periods = len(report_periods)

# 初始化边的状态字典
edge_status = {}

# 处理每个报告期的数据
for i, rpt in enumerate(report_periods):
    # 提取特定报告期的数据
    trade_rpt = trade[trade['rpt'] == rpt][['supplier', 'customer']]
    
    # 更新边的状态字典
    for _, row in trade_rpt.iterrows():
        supplier_idx = node_index[row['supplier']]
        customer_idx = node_index[row['customer']]
        edge_key = f"{supplier_idx}_{customer_idx}"
        # edge_status = {tuple(map(int, key.split('_'))): value for key, value in edge_status_str_keys.items()}
        # tuple(map(int, edge_key.split('_')))把单个字符转换回来
        if edge_key not in edge_status:
            edge_status[edge_key] = [0] * num_periods
        edge_status[edge_key][i] = 1

# 保存状态字典到文件
with open('adjence_matrix/edge_status.json', 'w') as f:
    json.dump(edge_status, f)

print("边的状态字典已保存到 adjence_matrix/edge_status.json")