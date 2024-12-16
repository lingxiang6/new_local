import pandas as pd
import numpy as np
import json
'''import json

# 读取状态字典
with open('data/adjence_matrix/edge_status.json', 'r') as f:
    edge_status_str_keys = json.load(f)

# 将字符串键还原为元组键
edge_status = {tuple(map(int, key.split('_'))): value for key, value in edge_status_str_keys.items()}'''


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

# 获取所有报告期
report_periods = trade['rpt'].unique()

# 定义映射函数
def maping_index(data, node_index):
    # 将节点名称替换为索引
    data['supplier'] = data['supplier'].map(node_index)
    data['customer'] = data['customer'].map(node_index)
    return data

# 创建节点到索引的映射
node_index = {node: idx for idx, node in enumerate(related_ids)}
with open('adjence_matrix/node_index.json', 'w') as f:
    json.dump(node_index, f)

# 处理每个报告期的数据
for rpt in report_periods:
    # 提取特定报告期的数据
    trade_rpt = trade[trade['rpt'] == rpt][['supplier', 'customer']]
    
    # 映射节点
    trade_rpt_mapped = maping_index(trade_rpt, node_index)
    
    # 保存到CSV文件
    filename = f'adjence_matrix/A_{rpt.replace("-", "_")}.csv'
    trade_rpt_mapped.to_csv(filename, index=False)
    print(f'Saved {filename}')

print("All files have been processed and saved.")