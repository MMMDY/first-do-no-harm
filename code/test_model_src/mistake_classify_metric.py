import json
import re
import os
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

def parse_prediction_robust(predict_string: str) -> tuple[set[int] | None, int | None]:
    """
    更鲁棒地从模型预测文本中提取信息，兼容多种格式。
    支持如 "1. IDs: 1,2"、"2. Error Type ID: 3"、"## Problematic Utterance ID(s): 3, 5, 7"、"## Error Type ID: 5" 等。
    """
    predicted_lines = None
    predicted_id = None
    lines = predict_string.splitlines()

    for line in lines:
        stripped_line = line.strip()
        # 兼容 "1." 或 "## Problematic Utterance ID(s):"
        if stripped_line.startswith('1.') or 'Problematic Utterance ID' in stripped_line:
            # 使用正则表达式找到所有数字
            numbers_found = re.findall(r'\d+', stripped_line)
            if numbers_found:
                # 移除可能存在的序号 "1." 本身
                if stripped_line.startswith('1.') and numbers_found[0] == '1' and len(numbers_found) > 1:
                     predicted_lines = {int(num) for num in numbers_found[1:]}
                else:
                     predicted_lines = {int(num) for num in numbers_found}

        # 兼容 "2." 或 "## Error Type ID:"
        elif stripped_line.startswith('2.') or 'Error Type ID' in stripped_line:
            numbers_found = re.findall(r'\d+', stripped_line)
            if numbers_found:
                # 取最后一个数字作为ID，以避免将序号 "2." 误判为ID
                predicted_id = int(numbers_found[-1])

    if predicted_lines is not None and predicted_id is not None:
        return predicted_lines, predicted_id
    else:
        # 如果只解析出一部分，也返回None，确保数据对的完整性
        return None, None

def evaluate_predictions(file_path: str):
    """
    读取单个预测文件，解析预测结果，与真实标签比较，并计算打印各项指标。
    (已更新，增加了Task 2的Recall, F1-Score, 和 Exact Match Ratio 指标)
    返回计算出的指标字典。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: File not found -> {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"ERROR: File content is not a valid JSON -> {file_path}")
        return None

    # Task 1: Mistake ID 指标列表
    true_mistake_ids = []
    pred_mistake_ids = []
    
    # Task 2: Line Identification 指标列表
    line_precisions = []
    line_recalls = []
    line_f1_scores = []
    line_jaccard_accuracies = []
    exact_matches = []

    successful_parses = 0
    total_samples = len(data)
    skipped_samples = 0
    
    results = {}

    for item in data:
        predict_text = item.get("predict", "")
        pred_lines, pred_id = parse_prediction_robust(predict_text)
        
        # 如果模型输出无法被成功解析，则跳过该样本
        if pred_lines is None or pred_id is None:
            skipped_samples += 1
            continue
            
        true_id = item.get("mistake_id")
        
        # --- 跳过 id=16 (优良实践) 样本的逻辑 (可根据需要启用) ---
        if (pred_id == 16) or (true_id == 16):
            skipped_samples += 1
            continue

        successful_parses += 1
        
        # --- 收集 Task 1 数据 ---
        true_mistake_ids.append(true_id)
        pred_mistake_ids.append(pred_id)

        # --- 计算并收集 Task 2 指标 ---
        true_lines = set(item.get("false_label", []))
        intersection = true_lines.intersection(pred_lines)
        union = true_lines.union(pred_lines)
        
        # 计算 Precision
        precision = len(intersection) / len(pred_lines) if len(pred_lines) > 0 else 0
        line_precisions.append(precision)

        # 计算 Recall
        recall = len(intersection) / len(true_lines) if len(true_lines) > 0 else 1.0
        line_recalls.append(recall)

        # 计算 F1-Score
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        line_f1_scores.append(f1)

        # 计算 Jaccard Accuracy
        jaccard_accuracy = len(intersection) / len(union) if len(union) > 0 else 1.0
        line_jaccard_accuracies.append(jaccard_accuracy)
        
        # 计算 Exact Match
        exact_matches.append(1 if true_lines == pred_lines else 0)

    print("-" * 50)
    print(f"Evaluation complete. Total: {total_samples}, Skipped: {skipped_samples}, Evaluated: {successful_parses}.")
    print("-" * 50)

    if successful_parses > 0:
        # --- Task 1 结果 ---
        print("\n📊 Task 1: Mistake ID Classification")
        report_dict = classification_report(
            true_mistake_ids, 
            pred_mistake_ids, 
            output_dict=True,
            zero_division=0
        )
        
        print("\n   --- Per-Class Metrics ---")
        print(f"{'Class':>5s} | {'Precision':>10s} | {'Recall':>10s} | {'F1-Score':>10s} | {'Support':>8s}")
        print("-" * 62)
        class_labels = sorted([k for k in report_dict.keys() if k.isdigit()], key=int)
        for label in class_labels:
            metrics = report_dict[label]
            p, r, f1 = metrics.get('precision', 0), metrics.get('recall', 0), metrics.get('f1-score', 0)
            s = int(metrics.get('support', 0))
            print(f"{label:>5s} | {p*100:>9.2f}% | {r*100:>9.2f}% | {f1*100:>9.2f}% | {s:>8d}")
        print("-" * 62)
        
        print("\n   --- Overall Metrics ---")
        accuracy = accuracy_score(true_mistake_ids, pred_mistake_ids)
        weighted_avg = report_dict.get('weighted avg', {})
        w_precision = weighted_avg.get('precision', 0)
        w_recall = weighted_avg.get('recall', 0)
        w_f1_score = weighted_avg.get('f1-score', 0)

        print(f"   - Accuracy:  {accuracy * 100:.2f}%")
        print(f"   - Precision: {w_precision * 100:.2f}% (Weighted)")
        print(f"   - Recall:    {w_recall * 100:.2f}% (Weighted)")
        print(f"   - F1-Score:  {w_f1_score * 100:.2f}% (Weighted)")
        
        results['task1'] = {
            'accuracy': accuracy,
            'precision': w_precision,
            'recall': w_recall,
            'f1': w_f1_score
        }

        # --- Task 2 结果 ---
        print("\n🎯 Task 2: Problematic Line Identification")
        avg_line_precision = np.mean(line_precisions) if line_precisions else 0
        avg_line_recall = np.mean(line_recalls) if line_recalls else 0
        avg_line_f1 = np.mean(line_f1_scores) if line_f1_scores else 0
        avg_jaccard_accuracy = np.mean(line_jaccard_accuracies) if line_jaccard_accuracies else 0
        exact_match_ratio = np.mean(exact_matches) if exact_matches else 0

        print(f"   - Mean Precision:        {avg_line_precision * 100:.2f}%")
        print(f"   - Mean Recall:           {avg_line_recall * 100:.2f}%")
        print(f"   - Mean F1-Score:         {avg_line_f1 * 100:.2f}%")
        print(f"   - Mean Jaccard Accuracy: {avg_jaccard_accuracy * 100:.2f}%")
        print(f"   - Exact Match Ratio:     {exact_match_ratio * 100:.2f}%")
        
        results['task2'] = {
            'precision': avg_line_precision,
            'recall': avg_line_recall,
            'f1': avg_line_f1,
            'jaccard': avg_jaccard_accuracy,
            'exact_match': exact_match_ratio
        }
        
    else:
        print("No valid samples were available for evaluation.")
        
    return results

def main(directory_path: str):
    """
    遍历指定文件夹中的所有 .json 文件并进行评估，并将结果保存到 Markdown 文件。
    """
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at '{directory_path}'")
        return

    json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]

    if not json_files:
        print(f"No .json files found in '{directory_path}'")
        return

    print(f"Found {len(json_files)} JSON files to evaluate in '{directory_path}'.")
    print("="*70)

    result_file_path = os.path.join(directory_path, "evaluation_results.md")
    print(f"Results will be sorted and saved to: {result_file_path}")

    with open(result_file_path, "w", encoding="utf-8") as out_file:
        # 写入 Markdown 表头 (删除 T2_Acc)
        header = "| Filename | T1_Acc | T1_Prec | T1_Recall | T1_F1 | T2_Prec | T2_Recall | T2_F1 | T2_Jaccard | T2_EM |\n"
        separator = "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        out_file.write(header)
        out_file.write(separator)

        for filename in sorted(json_files):
            print(f"Processing file: {filename}")
            full_path = os.path.join(directory_path, filename)
            metrics = evaluate_predictions(full_path)
            
            if metrics and 'task1' in metrics and 'task2' in metrics:
                t1 = metrics['task1']
                t2 = metrics['task2']
                # 格式化输出行: Markdown 表格行 (删除 T2_Acc 数据列)
                line_str = (
                    f"| {filename} | "
                    f"{t1.get('accuracy', 0) * 100:.2f} | {t1.get('precision', 0) * 100:.2f} | {t1.get('recall', 0) * 100:.2f} | {t1.get('f1', 0) * 100:.2f} | "
                    f"{t2.get('precision', 0) * 100:.2f} | {t2.get('recall', 0) * 100:.2f} | {t2.get('f1', 0) * 100:.2f} | {t2.get('jaccard', 0) * 100:.2f} | {t2.get('exact_match', 0) * 100:.2f} |\n"
                )
                out_file.write(line_str)
            
            print("\n" + "="*70 + "\n")

    print(f"All evaluations finished. Results saved to {result_file_path}")

if __name__ == "__main__":
    # --- 用户需要修改的部分 ---
    # 请将这里的路径设置为包含所有预测结果 .json 文件的文件夹
    # 提示: 在Windows上，使用正斜杠 / 或者双反斜杠 \\ 来避免路径解析问题
    results_folder = "/WX24061/lzy/test_model/mistake_classify"
    #results_folder = "/WX24061/lzy/test_model/mistake_classify_train_set_rag"
    #results_folder = "/WX24061/lzy/test_model/mistake_classify_train_set_100_rag"
    #results_folder = "/WX24061/lzy/test_model/mistake_classify_mate2_v1"
    #results_folder = "/WX24061/lzy/test_model/mistake_classify_mistake_set"
    #results_folder = "/WX24061/lzy/test_model/mistake_classify_mate2_v2"
    #results_folder = "/WX24061/lzy/test_model/mistake_classify_mate3_v1"
    #results_folder = "/WX24061/lzy/test_model/mistake_classify_grpo"
    #results_folder = "/WX24061/lzy/test_model/mistake_classify_mate5_v1"
    #results_folder = "/WX24061/lzy/test_model/mistake_classify_mate4_v1"
    #results_folder = "/WX24061/lzy/test_model/mistake_classify_fewshot"
    
    # 执行主函数
    main(results_folder)
# /WX24061/LLaMA-Factory/saves/Qwen3-8B-Instruct/lora/train_mate2-v1/checkpoint-200

                                                                  
             