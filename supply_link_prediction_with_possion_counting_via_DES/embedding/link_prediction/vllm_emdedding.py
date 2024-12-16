import copy
import os  # 操作系统相关的模块，用于设置环境变量
from vllm import LLM  # vLLM库，用于加载和推理大模型
from vllm.sampling_params import SamplingParams  # 用于定义采样参数（如生成文本时的停止条件）
from modelscope import AutoTokenizer, GenerationConfig, snapshot_download 
import pandas as pd
# 可以批量生成token，速度很快，但llm好像并不能获取隐藏层输出

class EmbeddingGenerator:
    def __init__(self, model_path, 
                            prompt,
                            batch_size=1000,
                            quantization = None,
                            dtype="float16",
                            tensor_parallel_size=1,
                            gpu_memory_utilization=0.6):
        # 通义千问模型的特殊token定义
        IMSTART = '<|im_start|>'  # 用于标记对话开始
        IMEND = '<|im_end|>'  # 用于标记对话结束
        ENDOFTEXT = '<|endoftext|>'  # 用作EOS（句子结束标记）和PAD（填充标记）

        self.quantization = quantization
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.model_path = model_path

        self.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
        # 加载分词器（tokenizer），用于将文本转为token
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # 设置结束符号的token id，用于模型推理时识别结束点
        self.tokenizer.eos_token_id = self.generation_config.eos_token_id
        # 定义推理过程中遇到这些token id时停止继续生成
        self.stop_words_ids = [self.tokenizer.im_start_id, self.tokenizer.im_end_id, self.tokenizer.eos_token_id]
        os.environ['VLLM_USE_MODELSCOPE'] = 'True'
        # 暂停词
        extra_stop_words_ids=[]
        self.stop_words_ids = self.stop_words_ids + extra_stop_words_ids

        self.prompt = prompt
        self.batch_size=batch_size
    
    def run_LLM(self, data_list):
        self.sampling_params = SamplingParams(
                stop_token_ids=self.stop_words_ids,  # 停止生成的token id
                early_stopping=False,  # 是否提前停止
                top_p=self.generation_config.top_p,  # nucleus采样参数
                top_k=-1 if self.generation_config.top_k == 0 else self.generation_config.top_k,  # top_k采样参数
                temperature=self.generation_config.temperature,  # 生成的随机性控制
                repetition_penalty=self.generation_config.repetition_penalty,  # 惩罚重复生成的权重
                max_tokens=self.generation_config.max_new_tokens  # 最大生成token数
                )
        # 加载vLLM模型
        self.model = LLM(
            model=self.model_path,  # 模型目录
            tokenizer=self.model_path,  # 分词器目录
            tensor_parallel_size=self.tensor_parallel_size,  # 并行处理大小
            trust_remote_code=True,  # 是否信任远程代码
            quantization=self.quantization,  # 量化配置
            gpu_memory_utilization=self.gpu_memory_utilization,  # GPU内存利用率
            dtype=self.dtype  # 数据类型
        )
        all_embeddings = []
        # 分批处理数据
        for i in range(0, len(names), self.batch_size):
            # 获取当前批次的数据
            batch = names[i:i + self.batch_size]
            # 调用模型进行预测
            response_list = self.generate_embeddings(querys=batch)
            print(response_list)
            # 将结果追加到总的列表中
            all_embeddings.extend(response_list)
        print(all_embeddings)
        return all_embeddings

    def generate_embeddings(self, querys):
        prompt_list = []  # 用于存储每个 query 对应的 prompt 文本和 token
        for query in querys:  # 遍历 queries 列表中的每个 query
            prompt_text, prompt_tokens = self._build_prompt(
                self.generation_config,  # 生成配置
                self.tokenizer,  # 分词器
                query,  # 当前用户输入的查询
                prompt=self.prompt,  # 历史聊天记录
                system='You are a helpful assistant.'  # 系统消息，例如系统人格描述'  # 系统消息，例如系统人格描述
            )
            prompt_list.append(prompt_tokens)  # 文本不返回，只返回token
        # 批量生成嵌入
        req_outputs = self.model.generate(
            prompt_token_ids=prompt_list, 
            sampling_params=self.sampling_params, 
            use_tqdm=False  # 禁用进度条
        )
        response_list =[]
        for i in req_outputs:
        # 移除停用词，确保输出内容不包含这些词
            response_token_ids = self.remove_stop_words(i.outputs[0].token_ids,self.stop_words_ids)
            response_list.append(response_token_ids)
        return response_list 


    def _build_prompt(self,
                generation_config,  # 模型的生成配置，包含最大窗口大小、结束标记等
                tokenizer,  # 分词器，用于将文本编码为 token
                query,  # 用户当前的输入问题
                prompt,  # prompt
                system="You are a helpful assistant."):  # 系统提示信息（通常是对助手角色的描述）
            # 定义特殊token：<|im_start|> 表示发言开始，<|im_end|> 表示发言结束
        im_start, im_start_tokens = '<|im_start|>', [tokenizer.im_start_id]
        im_end, im_end_tokens = '<|im_end|>', [tokenizer.im_end_id]

        nl_tokens = tokenizer.encode("\n")
        # 定义一个函数，用于将角色（system/user/assistant）和对应内容编码为字符串和 token 列表
        def _tokenize_str(role, content):
        # 返回格式化后的文本和其对应的 token ids
            return f"{role}\n{content}", tokenizer.encode(role) + nl_tokens + tokenizer.encode(content)

        # 剩余的 token 数量，初始为模型最大窗口大小
        left_token_space = generation_config.max_window_size
        # 构建 prompt 头部：系统提示内容 (system message)
        system_text_part, system_tokens_part = _tokenize_str("system", system)  # 例如 "system\nYou are a helpful assistant."
        system_text = f'{im_start}{system_text_part}{im_end}'  # 加上 <|im_start|> 和 <|im_end|> 标记
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens  # 对应的 token 列表
        left_token_space -= len(system_tokens)  # 减去头部消耗的 token 数

        # 构建带有引导信息的完整查询
        query = prompt + '\n' + query
        # 构建 prompt 尾部：用户问题 (query) 和助手的引导信息
        query_text_part, query_tokens_part = _tokenize_str('user', query)  # 用户问题部分
        query_tokens_prefix = nl_tokens + im_start_tokens  # 开始部分的换行符和 <|im_start|>
        query_tokens_suffix = im_end_tokens + nl_tokens + im_start_tokens + tokenizer.encode('assistant') + nl_tokens  # 结束部分和助手标识
        # 如果 query 太长，超过剩余 token 空间，则对其进行截断
        if len(query_tokens_prefix) + len(query_tokens_part) + len(query_tokens_suffix) > left_token_space:
            query_token_len = left_token_space - len(query_tokens_prefix) - len(query_tokens_suffix)
            query_tokens_part = query_tokens_part[:query_token_len]  # 截断后的 token
            query_text_part = tokenizer.decode(query_tokens_part)  # 截断后的文本

        query_tokens = query_tokens_prefix + query_tokens_part + query_tokens_suffix  # 最终用户问题部分的 token 列表
        query_text = f"\n{im_start}{query_text_part}{im_end}\n{im_start}assistant\n"  # 最终构造的用户问题部分
        left_token_space -= len(query_tokens)  # 更新剩余的 token 数

        # 生成完整的 Prompt,不用历史信息
        # prompt_str = f'{system_text}{history_text}{query_text}'  # 包含系统提示、历史对话、用户问题的完整文本
        prompt_str = f'{system_text}{query_text}'
        prompt_tokens = system_tokens  + query_tokens  # 对应的完整 token 列表
        return prompt_str, prompt_tokens  # 返回完整的 Prompt 和其对应的 token

    def remove_stop_words(self,token_ids, stop_words_ids):
        token_ids = copy.deepcopy(token_ids) 
        token_ids_list = list(token_ids) # 深拷贝 token_ids，避免修改原始数据
        # 从后向前移除停用词，直到遇到非停用词为止
        while len(token_ids) > 0:
            if token_ids_list[-1] in stop_words_ids:  # 如果最后一个 token 是停用词
                token_ids_list.pop(-1) # 移除该 token
            else:
                break  # 如果遇到非停用词，停止移除
        return token_ids_list  # 返回移除停用词后的 token 列表

data = pd.read_csv('/your/root')
data['related_orig']
data=data.iloc[:,:100]
names = data['related_orig'].tolist()
model_path = "LLM_model/qwen/qwenlocation1000"
LLM_batch_size = 100
location_prompt = "根据输入的公司名称，输出对应的省市名称,注意：直辖市回复到市，英文公司用中文回答所在国家"
supplier_prompt = "该公司是一个什么样的公司,他的主营商品有哪些"
customer_prompt = "该公司是一个什么样的公司,他的主要需要哪些商品"
location_embedding = EmbeddingGenerator(model_path,prompt=location_prompt, batch_size=LLM_batch_size,
                            quantization = None,
                            dtype="float16",
                            tensor_parallel_size=1,
                            gpu_memory_utilization=0.6)

supplier_embedding = EmbeddingGenerator(model_path,prompt=supplier_prompt, batch_size=LLM_batch_size,
                            quantization = None,
                            dtype="float16",
                            tensor_parallel_size=1,
                            gpu_memory_utilization=0.6)

customer_embedding = EmbeddingGenerator(model_path,prompt=customer_prompt, batch_size=LLM_batch_size,
                            quantization = None,
                            dtype="float16",
                            tensor_parallel_size=1,
                            gpu_memory_utilization=0.6)
tensor=location_embedding.run_LLM(names)
tensor=supplier_embedding.run_LLM(names)
tensor=supplier_embedding.run_LLM(names)