import json
import time
import logging
import fire
import requests
from tqdm import tqdm
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 依赖项检查与安装 ---
# 确保已安装必要的库:
# pip install requests "datasets>=2.0.0" rouge-chinese nltk jieba fire tqdm bert-score
try:
    import jieba
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from rouge_chinese import Rouge
    #from bert_score import score as bert_score_fn  # 新增

    # 初始化 jieba
    jieba.setLogLevel(logging.CRITICAL)
    jieba.initialize()
except ImportError:
    print("评估所需的依赖库未找到。")
    print("请运行: pip install requests 'datasets>=2.0.0' rouge-chinese nltk jieba fire tqdm bert-score")
    exit()


# ===================================================================
# Part 1: 使用 requests 库从 vLLM 获取预测（并发优化）
# ===================================================================

# --- 在此硬编码您的 API 和模型配置 ---
VLLM_API_URL = "http://localhost:8000/v1/chat/completions"
# MODEL_NAME = "/models/qwen3-8b-lora-700step"
#MODEL_NAME = "/models/llama3.1-8b-lora-4ep"
MODEL_NAME = "/WX24061/models/qwen3-14b"


def get_vllm_prediction(system_prompt: str, user_prompt: str) -> str:
    """
    使用 requests 发送单个请求到 vLLM API 并返回模型的预测。
    """
    headers = {"Content-Type": "application/json"}
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if user_prompt:
        #user_prompt += " /no_think"
        messages.append({"role": "user", "content": user_prompt})

    if not messages:
        return ""
        
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "top_p": 0.95,
        "max_tokens": 2048,
        "temperature": 0.05,
    }

    try:
        response = requests.post(VLLM_API_URL, headers=headers, json=payload)
        response.raise_for_status()  # 如果状态码不是 2xx，则引发异常
        return response.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        print(f"调用 API 时发生错误: {e}")
        return "" # 出错时返回空字符串

def generate_predictions(input_file: str, output_file: str, max_workers: int, num_samples: int = None):
    """
    读取输入文件，使用多线程并发生成预测，并将结果保存到输出文件。
    可指定读取的数据条数（num_samples），为 None 时读取全部。
    """
    print(f"\n--- 开始生成预测 (使用 {max_workers} 个并发线程) ---")
    print(f"正在从文件读取数据: {input_file}")
    
    # 如果output_file不存在，报错
    # if os.path

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if num_samples is not None:
            data = data[:num_samples]

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {
            executor.submit(get_vllm_prediction, item.get("instruction", ""), item.get("input", "")): item
            for item in data
        }

        for future in tqdm(as_completed(future_to_item), total=len(data), desc="生成预测中"):
            original_item = future_to_item[future]
            try:
                prediction = future.result()
            except Exception as e:
                print(f"一个任务在执行中产生异常: {e}")
                prediction = ""

            result_item = original_item.copy()
            result_item["predict"] = prediction
            if 'output' in result_item:
                result_item["label"] = result_item.pop("output")
            results.append(result_item)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"预测结果已成功保存至: {output_file}")
    print("-" * 40)


# ===================================================================
# Part 2: 计算 BLEU、ROUGE 和 BERTScore 分数
# ===================================================================

def compute_metrics(sample):
    """
    为单个样本计算 BLEU、ROUGE 和 BERTScore 分数。
    """
    hypothesis = list(jieba.cut(str(sample.get("predict", ""))))
    reference = list(jieba.cut(str(sample.get("label", ""))))

    if not hypothesis or not reference:
        return {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0, "bleu-4": 0.0}

    bleu_score = sentence_bleu(
        [reference],
        hypothesis,
        smoothing_function=SmoothingFunction().method3,
    )

    rouge = Rouge()
    hypothesis_str = " ".join(hypothesis)
    reference_str = " ".join(reference)
    
    try:
        scores = rouge.get_scores(hypothesis_str, reference_str)
        result = scores[0]
    except ValueError:
        result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}

    metric_result = {k: round(v["f"] * 100, 4) for k, v in result.items()}
    metric_result["bleu-4"] = round(bleu_score * 100, 4)

    # 新增 BERTScore 计算
    # try:
    #     P, R, F1 = bert_score_fn([sample.get("predict", "")], [sample.get("label", "")], lang="zh", rescale_with_baseline=True)
    #     metric_result["bertscore"] = round(float(F1[0]) * 100, 4)
    # except Exception as e:
    #     metric_result["bertscore"] = 0.0
        
    # P, R, F1 = bert_score_fn([sample.get("predict", "")], [sample.get("label", "")], lang="zh", rescale_with_baseline=True)
    # metric_result["bertscore"] = round(float(F1[0]) * 100, 4)

    return metric_result


def evaluate_scores(filename: str):
    """
    加载预测文件并计算平均分数。
    """
    print(f"\n--- 开始评估分数 ---")
    print(f"正在加载预测文件: {filename}")
    start_time = time.time()
    
    dataset = load_dataset("json", data_files=filename, split="train")
    
    columns_to_keep = {'predict', 'label'}
    columns_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]
    dataset = dataset.remove_columns(columns_to_remove)

    dataset = dataset.map(compute_metrics, num_proc=16)
    score_dict = dataset.to_dict()

    print("\n--- 平均分数 ---")
    average_score = {}
    for task, scores in sorted(score_dict.items(), key=lambda x: x[0]):
        # 只统计分数字段（字段名包含 'rouge'、'bleu' 或 'bertscore'）
        if not any(key in task for key in ['rouge', 'bleu']):
            continue
        valid_scores = []
        for s in scores:
            try:
                valid_scores.append(float(s))
            except (ValueError, TypeError):
                continue
        if valid_scores:
            avg = sum(valid_scores) / len(valid_scores)
            average_score[task] = avg
            print(f"{task}: {avg:.4f}")
    
    score_filename = "/WX24061/test_model/predictions_score.json"
    with open(score_filename, "w", encoding="utf-8") as f:
        json.dump(average_score, f, indent=4)

    print(f"\n评估完成，耗时 {time.time() - start_time:.3f} 秒。")
    print(f"分数文件已保存至: {score_filename}")
    print("-" * 40)


# ===================================================================
# Part 3: 主执行函数
# ===================================================================

def run(max_workers: int = 30, num_samples: int = None):
    """
    执行完整的预测和评估流程。
    所有配置均已硬编码。

    Args:
        max_workers (int): 生成预测时的最大并发线程数。
        num_samples (int): 读取的数据条数（None 表示全部）。
    """
    input_file = "/WX24061/LLaMA-Factory/data/critic_test_set_aplace_1.json"
    predictions_output_file = "/WX24061/test_model/critic_predict/qwen3-14b.json"
    
    generate_predictions(input_file, predictions_output_file, max_workers, num_samples)
    evaluate_scores(predictions_output_file)


if __name__ == "__main__":
    # 使用 fire 库创建命令行接口
    # 现在可以直接运行: python your_script_name.py
    # 或者调整并发数: python your_script_name.py --max_workers=32 --num_samples=100  "/WX24061/models/qwen3-1.7b-lora-90step
    fire.Fire(run)
    
    # vllm serve /models/qwen3-8b-instruct --enable-lora --lora-modules lora1=/WX24061/LLaMA-Factory/saves/Qwen3-8B-Instruct/lora/train_2025-07-25-16-32-57/checkpoint-300
    
    # CUDA_VISIBLE_DEVICES=5 vllm serve /models/llama3.1-8b-lora-4ep
    
    # CUDA_VISIBLE_DEVICES=3,4,5,6,7 vllm serve /models/llama3.1-8b --enable-lora --lora-modules lora1=/WX24061/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/train_2025-07-27-14-23-18/checkpoint-140
    
    # vllm serve /models/llama3.1-8b-60step
    # CUDA_VISIBLE_DEVICES=7 vllm serve /WX24061/models/qwen3-14b-90step
    
    # CUDA_VISIBLE_DEVICES=7 vllm serve /WX24061/models/qwen3-14b
    # CUDA_VISIBLE_DEVICES=7 vllm serve /WX24061/models/qwen3-14b-150step
    #CUDA_VISIBLE_DEVICES=7 vllm serve /WX24061/models/qwen3-14b-60step