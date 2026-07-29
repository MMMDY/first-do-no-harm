import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
import time
from bert_score import score as bert_score_fn
from datasets import load_dataset

def compute_bertscore(input_json, output_json, batch_size=256):
    print(f"加载数据: {input_json}")
    dataset = load_dataset("json", data_files=input_json, split="train")
    preds = dataset["predict"]
    refs = dataset["label"]

    print(f"共 {len(preds)} 条样本，开始批量计算 BERTScore ...")
    start = time.time()
    # 使用GPU，lang="zh"，rescale_with_baseline=True
    P, R, F1 = bert_score_fn(
        preds, refs, lang="en",  batch_size=batch_size, device="cuda"
    )
    avg_f1 = float(F1.mean()) * 100
    print(f"BERTScore F1 平均分: {avg_f1:.4f}")

    # 保存每条样本的分数
    results = []
    for pred, ref, f1 in zip(preds, refs, F1):
        results.append({
            "predict": pred,
            "label": ref,
            "bertscore": float(f1) * 100
        })
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({
            "average_bertscore": avg_f1,
            "details": results
        }, f, ensure_ascii=False, indent=2)
    print(f"已保存到: {output_json}")
    print(f"总耗时: {time.time() - start:.2f} 秒")

if __name__ == "__main__":
    # import fire
    # fire.Fire(compute_bertscore)

    input_file = "/WX24061/test_model/critic_predict/llama3.1-8b-60step.json"
    output_json = "./output.json"
    
    #generate_predictions(input_file, predictions_output_file, max_workers, num_samples)
    compute_bertscore(input_file,output_json)