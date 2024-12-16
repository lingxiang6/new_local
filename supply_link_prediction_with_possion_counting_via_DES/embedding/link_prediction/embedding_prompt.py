from modelscope import AutoModelForCausalLM, AutoTokenizer
import time
import torch
import numpy as np
import copy
import os  
from modelscope import AutoTokenizer
import pandas as pd

class EmbeddingGenerator:
    def __init__(self, model_path, prompt,
                    batch_size=100):
        self.model_path = model_path  
        self.prompt = prompt 
        self.batch_size = batch_size
        self.max_length=500
        self.device='cuda:0'
    def cons_batch(self, sentences, tokenizer,):
        # 根据模板格式化每个句子，准备输入
        format_sentence = [f"{self.prompt} {sentence}" for sentence in sentences]
        batch = tokenizer.batch_encode_plus(
                            format_sentence,
                            return_tensors='pt',  # 返回PyTorch的张量
                            padding=True,  # 自动填充batch中较短的句子
                            max_length=self.max_length,  # 设定最大长度
                            truncation=self.max_length is not None)
        for k in batch:  
            batch[k] = batch[k].to(self.device) if batch[k] is not None else None  
        return batch
    def encode(self, sentences):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True)
        print("Start to load model")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",  
            torch_dtype='auto',
            return_dict_in_generate=True,  
            output_hidden_states=True, 
            trust_remote_code=True).eval()  
        # 对输入句子列表进行编码，生成embedding
        batch_size = self.batch_size
        result = []       
        # 按批次处理输入句子
        for i in range(0, len(sentences), batch_size):
            batch = self.cons_batch(sentences[i:i + batch_size],tokenizer=tokenizer)  # 构建批次            
            with torch.no_grad():  
                outputs = model(output_hidden_states=True, return_dict=True, **batch)  
                embedding = outputs.hidden_states  
                #这里是每个层的输出，每个层的输出都被存下来了
                last_hidden_states = embedding[-1].mean(dim=1)           
                last_hidden_states = last_hidden_states.float().cpu().numpy()
                result.append(last_hidden_states)

        torch.cuda.empty_cache()
        return np.concatenate(result, axis=0).astype('float')# 将结果拼接为一个矩阵并转换为float类型，尺寸：(sentence_length, 4096)