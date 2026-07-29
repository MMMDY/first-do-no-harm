import json
import os
import time
import logging

import vllm
import fire
from llamafactory import model
from matplotlib.pyplot import step
from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

from websockets import serve


  
class Critic_Type_Classify:
    def __init__(self, model_name: str, api_base: str, api_key: str, template_path: str):
        """
        初始化 OpenAI 客户端。

        Args:
            model_name (str): 要使用的模型名称。
            api_base (str): 服务的 API 端点。
            api_key (str): API 密钥。
        """
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.prompt_template = None
        with open(template_path, 'r', encoding='utf-8') as f:
            self.prompt_template = f.read()
        
        print(f"客户端已初始化，目标 API: {api_base}, 模型: {model_name}")
        
        

    def get_prediction(self, system_prompt: str, history: str, enable_thinking: bool, temperature: float) -> str:
        """
        发送单个请求并返回模型的响应。
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
     
        history_formatted = ""
        for item in history:
            history_formatted += f"""{item.get("id")}. {item.get("role")}: {item.get("content")}\n"""
            
        #history_formatted += " /no_think"
        #print(f"History: {history}")
        #print(f"Formatted History: {history_formatted}")
        prompt = self.prompt_template.format(history=history_formatted)
        #print(f"Prompt: {prompt}")
        messages.append({"role": "user", "content": prompt})
        
        if not messages:
            return ""
        
        response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=False,
                temperature=temperature,
                max_tokens=5000,
                #top_p=0.95
                extra_body={"chat_template_kwargs": {
                "enable_thinking": enable_thinking  # 关闭思考模式。改为 True 则开启。
            }}
            )
        return response.choices[0].message.content.strip()

        # try:
        #     response = self.client.chat.completions.create(
        #         model=self.model_name,
        #         messages=messages,
        #         stream=False,
        #         temperature=0,
        #         max_tokens=1024,
        #         extra_body={"enable_thinking": False},
        #     )
        #     return response.choices[0].message.content.strip()
        # except Exception as e:
        #     print(f"调用 API 时发生错误: {e}")
        #     return ""
        
        
    def generate_predictions(self, input_file: str, output_file: str, max_workers: int, num_samples: int = None, enable_thinking: bool = False, temperature: float = 0.01):
        """
        读取输入文件，使用多线程并发生成预测，并将结果保存到输出文件。
        可以通过 num_samples 指定读取的数据个数。
        """
        print(f"\n--- 开始生成预测 (使用 {max_workers} 个并发线程) ---")
        print(f"正在从文件读取数据: {input_file}")

        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if num_samples is not None:
            data = data[:num_samples]

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {
                executor.submit(self.get_prediction, item.get("instruction", ""), item.get("history", ""), enable_thinking, temperature): item
                for item in data
            }

            for future in tqdm(as_completed(future_to_item), total=len(data), desc="生成预测中"):
                original_item = future_to_item[future]
                prediction = future.result()
                result_item = original_item.copy()
                result_item["predict"] = prediction
                if 'output' in result_item:
                    result_item["label"] = result_item.pop("output")
                results.append(result_item)
            
        # 如果没有文件夹，则创建文件夹
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
        print(f"预测结果已成功保存至: {output_file}")
        print("-" * 40)
        
    
def run(max_workers: int = 50, num_samples: int = None):
    """
    执行完整的预测和评估流程。
    """

    prompt_template_path = "/WX24061/lzy/test_model/mistake_classify_promt.txt" # /WX24061/models/qwen3-14b
    #input_file = "/WX24061/LLaMA-Factory/data/critic_test_set_aplace.json"
    input_file = "/WX24061/LLaMA-Factory/data/critic_test_set_aplace.json"
    #predictions_output_file = "/WX24061/lzy/test_model/mistake_classify/llama3.1-8b-nothink.json" 
    #predictions_output_dir = "/WX24061/lzy/test_model/mistake_classify_grpo/"
    #predictions_output_dir= "/WX24061/lzy/test_model/mistake_classify_mate5_v1/"
    predictions_output_dir="/WX24061/lzy/test_model/mistake_classify/"
    predictions_output_file = predictions_output_dir + "llama3.1-8b.json"
    # /WX24061/models/qwen3-1.7b-lora-90step

    #model_name =  "/models/qwen3-8b-lora-700step"
    # model_name =  "/models/qwen3-8b-instruct"
    # model_name =  "models/qwen3-8b-lora-700step"
    #"/models/qwen3-8b-instruct"

    
    # api_base = "https://api.deepseek.com/v1"
    # api_key = "sk-960253e600d141de9a0577ae5eb65ba0"
    # model_name = "deepseek-chat"
    
    api_base = "http://0.0.0.0:8000/v1"
    api_key = "not-needed"
    
    # model_name =  "/WX24061/models/qwen3-14b-60step"
    
    # api_base = "https://api.ai-gaochao.cn/v1"
    # api_key=
    # model_name="gemini-2.5-pro"
    
    #model_name = "/WX24061/lzy/Models/qwen3-8b-instruct"
    model_name = "/WX24061/lzy/Models/llama3.1-8b-instruct"
    #model_name = "/WX24061/lzy/models/qwen3-14b"
    #model_name = "/WX24061/lzy/models/qwen3-32b"
    # model_name = "/WX24061/lzy/models/qwen3-8b-mate1-700step"
    #model_name = "/WX24061/lzy/models/qwen3-8b-mate2-v1-200step"
    #model_name="/WX24061/lzy/models/qwen3-14b-mate2-v1-100step"
    #model_name='/WX24061/lzy/models/llama3.1-8b-mate2-v1-100step'
    #model_name="/WX24061/lzy/grpo/output/v14-20260203-040007/checkpoint-700-merged"
    #model_name="/WX24061/lzy/models/Qwen3-8B-Instruct-mate4-cot-v1-200step"
    #model_name="/WX24061/lzy/models/Qwen3-8B-Instruct-mate5-cot-v1-150step"
    #model_name="/WX24061/lzy/models/Qwen3-8B-Instruct-mate5-cot-v1-250-grpo-370"
    #model_name="/WX24061/lzy/models/Qwen3-14B-Instruct-mate5-cot-v1-600step"
    #model_name="/WX24061/lzy/models/Qwen3-8B-Instruct-mate5-250-grpo_plugin3-370-plugin4-270"
    #model_name="/WX24061/lzy/models/Qwen3-8B-Instruct-mate5-250-grpo_plugin3-370-plugin4-630"
    
    enable_thinking = False  # 是否启用思考模式
    temperature = 0.01

    # 步骤 1: 生成预测
    generator = Critic_Type_Classify(
        model_name=model_name, 
        api_base=api_base, 
        api_key=api_key,
        template_path=prompt_template_path
    )
    generator.generate_predictions(input_file, predictions_output_file, max_workers, num_samples,enable_thinking, temperature)
    
    
    # 步骤 2: 
    #evaluate_scores(predictions_output_file)


if __name__ == "__main__":
    # 使用 fire 库创建命令行接口
    # 现在可以直接运行: python your_script_name.py
    # 或者调整并发数: python your_script_name.py --max_workers=32
    fire.Fire(run)
    
    # vllm serve /models/qwen3-8b-lora-700step --max-model-len 4096
    
    # CUDA_VISIBLE_DEVICES=6 vllm serve /WX24061/lzy/Models/qwen3-8b-instruct --enable-lora --lora-modules lora1=/WX24061/LLaMA-Factory/saves/Qwen3-8B-Instruct/lora/train_2025-07-25-16-32-57/checkpoint-300 --max-model-len 4096
    
    # vllm serve /models/llama3.1-8b
    # vllm serve /models/llama3.1-8b-lora-4ep
    #  vllm serve /models/Mistral-7B-Instruct-v0.3
   # vllm serve /WX24061/models/qwen3-1.7b-lora-90step
   # CUDA_VISIBLE_DEVICES=6,7 vllm serve /WX24061/lzy/models/qwen3-1.7b
   
   # modelscope download --model Qwen/Qwen3-32B --local_dir /WX24061/models/qwen3-32b
   
   # modelscope download --model Qwen/Qwen3-14B --local_dir /WX24061/models/qwen3-14b
   # CUDA_VISIBLE_DEVICES=6,7 vllm serve /WX24061/lzy/Models/qwen3-8b-instruct
   # CUDA_VISIBLE_DEVICES=4 vllm serve /WX24061/lzy/Models/qwen3-8b-instruct --gpu-memory-utilization 0.8
   
#    CUDA_VISIBLE_DEVICES=6 vllm serve /WX24061/lzy/Models/qwen3-8b-instruct --enable-lora --lora-modules lora1=/WX24061/LLaMA-Factory/saves/Qwen3-8B-Instruct/lora/train_mate2-v1/checkpoint-200 --max-model-len 4096
#    CUDA_VISIBLE_DEVICES=6 vllm serve /WX24061/lzy/Models/llama3.1-8b-instruct --enable-lora --lora-modules lora1=/WX24061/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/train_llama3.1_mate2-v1/checkpoint-200 --max-model-len 4096
# CUDA_VISIBLE_DEVICES=6 vllm serve /WX24061/lzy/models/qwen3-14b --enable-lora --lora-modules lora1=/WX24061/LLaMA-Factory/saves/Qwen3-14B-Instruct/lora/train_mate2-v1/checkpoint-100 --max-model-len 4096

#    CUDA_VISIBLE_DEVICES=6 vllm serve /WX24061/lzy/Models/qwen3-8b-instruct --enable-lora --lora-modules lora1=/WX24061/LLaMA-Factory/saves/Qwen3-8B-Instruct/lora/train_mate2-v2/checkpoint-100 --max-model-len 4096
#    CUDA_VISIBLE_DEVICES=3 vllm serve /WX24061/lzy/models/qwen3-14b --enable-lora --lora-modules lora1=/WX24061/LLaMA-Factory/saves/Qwen3-14B-Instruct/lora/train_mate2-v2/checkpoint-50 --max-model-len 4096

# CUDA_VISIBLE_DEVICES=3 vllm serve /WX24061/lzy/models/qwen3-14b
# CUDA_VISIBLE_DEVICES=3 vllm serve /WX24061/lzy/Models/llama3.1-8b-instruct --enable-lora --lora-modules lora1=/WX24061/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/train_mate2_v2/checkpoint-150 --max-model-len 4096

# CUDA_VISIBLE_DEVICES=6 vllm serve /WX24061/lzy/models/qwen3-32b --max-model-len 4096

# CUDA_VISIBLE_DEVICES=7 vllm serve /WX24061/lzy/models/qwen3-32b --enable-lora --lora-modules lora1=/WX24061/LLaMA-Factory/saves/Qwen3-32B-Instruct/lora/train_mate2-v1/checkpoint-400 --max-model-len 4096

# CUDA_VISIBLE_DEVICES=3 vllm serve /WX24061/lzy/models/qwen3-32b --enable-lora --lora-modules lora1=/WX24061/LLaMA-Factory/saves/Qwen3-32B-Instruct/lora/train_mate3-v1/checkpoint-400 --max-model-len 4096

# CUDA_VISIBLE_DEVICES=3 vllm serve /WX24061/lzy/models/qwen3-14b --enable-lora --lora-modules lora1=/WX24061/LLaMA-Factory/saves/Qwen3-14B-Instruct/lora/train_mate3_v1/checkpoint-100 --max-model-len 4096

#    CUDA_VISIBLE_DEVICES=3 vllm serve /WX24061/lzy/models/qwen3-8b-mate1-700step --enable-lora --lora-modules lora1=/WX24061/lzy/grpo/output/v0-20260128-064654/checkpoint-1150 --max-model-len 4096 --max-lora-rank 64

#    CUDA_VISIBLE_DEVICES=3 vllm serve /WX24061/lzy/models/qwen3-8b-mate3-50step --max-model-len 4096

#    CUDA_VISIBLE_DEVICES=3 vllm serve /WX24061/lzy/models/qwen3-8b-mate3-150step --max-model-len 4096

#    CUDA_VISIBLE_DEVICES=3 vllm serve /WX24061/lzy/Models/qwen3-8b-instruct --enable-lora --lora-modules lora1=/WX24061/LLaMA-Factory/saves/Qwen3-8B-Instruct/lora/train_mate2-v1/checkpoint-200 --max-model-len 4096

#    CUDA_VISIBLE_DEVICES=4 vllm serve /WX24061/lzy/models/qwen3-8b-origin-grpo-1300 --max-model-len 4096 

# CUDA_VISIBLE_DEVICES=4 vllm serve /WX24061/lzy/models/qwen3-8b-mate2-v1-100step --max-model-len 4096 --gpu-memory-utilization 0.8

# CUDA_VISIBLE_DEVICES=4 vllm serve /WX24061/lzy/models/qwen3-8b-mate2-v1-200step --max-model-len 4096 --gpu-memory-utilization 0.8

# CUDA_VISIBLE_DEVICES=4 vllm serve /WX24061/lzy/models/qwen3-14b-mate2-v1-100step --max-model-len 4096 --gpu-memory-utilization 0.8

# CUDA_VISIBLE_DEVICES=5 vllm serve /WX24061/lzy/models/llama3.1-8b-mate2-v1-200step --max-model-len 4096 --gpu-memory-utilization 0.8

# CUDA_VISIBLE_DEVICES=5 vllm serve /WX24061/lzy/grpo/output/v14-20260203-040007/checkpoint-700-merged --max-model-len 4096 --gpu-memory-utilization 0.8 

# CUDA_VISIBLE_DEVICES=1 vllm serve /WX24061/lzy/models/Qwen3-8B-Instruct-mate4-cot-v1-50step --max-model-len 8192 --gpu-memory-utilization 0.8 
# CUDA_VISIBLE_DEVICES=1 vllm serve /WX24061/lzy/models/Qwen3-8B-Instruct-mate4-cot-v1-200step --max-model-len 8192 --gpu-memory-utilization 0.8 

# CUDA_VISIBLE_DEVICES=1 vllm serve /WX24061/lzy/models/Qwen3-8B-Instruct-mate5-cot-v1-250step --max-model-len 8192 --gpu-memory-utilization 0.8 
# CUDA_VISIBLE_DEVICES=1 vllm serve /WX24061/lzy/models/Qwen3-8B-Instruct-mate5-cot-v1-150step --max-model-len 8192 --gpu-memory-utilization 0.8 


# CUDA_VISIBLE_DEVICES=1 vllm serve /WX24061/lzy/models/Qwen3-8B-Instruct-mate5-cot-v1-250-grpo-370 --max-model-len 8192 --gpu-memory-utilization 0.8 

# CUDA_VISIBLE_DEVICES=1 vllm serve /WX24061/lzy/models/Qwen3-14B-Instruct-mate5-cot-v1-600step --max-model-len 8192 --gpu-memory-utilization 0.9

# CUDA_VISIBLE_DEVICES=1 vllm serve /WX24061/lzy/models/Qwen3-14B-Instruct-mate5-cot-v1-300step --max-model-len 8192 --gpu-memory-utilization 0.9

# CUDA_VISIBLE_DEVICES=1 vllm serve /WX24061/lzy/models/Qwen3-8B-Instruct-mate5-250-grpo_plugin3-370-plugin4-270 --max-model-len 8192 --gpu-memory-utilization 0.9

# CUDA_VISIBLE_DEVICES=1 vllm serve /WX24061/lzy/models/Qwen3-8B-Instruct-mate5-250-grpo_plugin3-370-plugin4-630 --max-model-len 8192 --gpu-memory-utilization 0.9

# CUDA_VISIBLE_DEVICES=5 vllm serve /WX24061/lzy/Models/qwen3-8b-instruct --max-model-len 4096 --gpu-memory-utilization 0.95 --port 8188