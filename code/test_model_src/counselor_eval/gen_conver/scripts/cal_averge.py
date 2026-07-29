from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_DIR = Path("/WX24061/lzy/test_model/counselor_eval/gen_conver/results/novice_1")
DEFAULT_DIR = Path("/WX24061/lzy/test_model/counselor_eval/gen_conver/results/normal_1")
DEFAULT_DIR = Path("/WX24061/lzy/test_model/counselor_eval/gen_conver/results/test_5")
METRICS = ["EFT_TFS", "HTAIS", "MITI", "PSC", "TES", "WAI"]


def collect_metric_values(file_path: Path) -> dict[str, float] | None:
	try:
		data = json.loads(file_path.read_text(encoding="utf-8"))
	except Exception as exc:
		print(f"[WARN] 读取失败，跳过: {file_path} ({exc})")
		return None

	counselor = data.get("evaluation_results", {}).get("counselor", {})
	if not counselor:
		print(f"[WARN] 缺少 evaluation_results.counselor，跳过: {file_path}")
		return None

	values: dict[str, float] = {}
	for metric in METRICS:
		value = counselor.get(metric)
		if value is None:
			print(f"[WARN] 缺少指标 {metric}，跳过文件: {file_path}")
			return None
		try:
			values[metric] = float(value)
		except (TypeError, ValueError):
			print(f"[WARN] 指标 {metric} 不是数值，跳过文件: {file_path}")
			return None

	return values


def calculate_average(result_dir: Path) -> dict[str, float]:
	pattern = "simpsydial_*_session1.json"
	files = sorted(result_dir.glob(pattern))

	if not files:
		raise FileNotFoundError(f"未找到匹配文件: {result_dir / pattern}")

	sums = {metric: 0.0 for metric in METRICS}
	valid_count = 0

	for file_path in files:
		values = collect_metric_values(file_path)
		if values is None:
			continue

		for metric in METRICS:
			sums[metric] += values[metric]
		valid_count += 1

	if valid_count == 0:
		raise ValueError("没有可用的有效文件可用于计算平均值")

	return {metric: sums[metric] / valid_count for metric in METRICS}


def main() -> None:
	parser = argparse.ArgumentParser(description="计算 counselor 指标平均值并输出 summary.json")
	parser.add_argument(
		"--result-dir",
		type=Path,
		default=DEFAULT_DIR,
		help="待遍历结果目录（默认: novice_1 目录）",
	)
	args = parser.parse_args()

	result_dir: Path = args.result_dir
	if not result_dir.exists() or not result_dir.is_dir():
		raise NotADirectoryError(f"目录不存在或不是文件夹: {result_dir}")

	averages = calculate_average(result_dir)
	output_path = result_dir / "summary.json"
	output_path.write_text(
		json.dumps(averages, ensure_ascii=False, indent=2),
		encoding="utf-8",
	)

	print(f"已写入: {output_path}")
	print(json.dumps(averages, ensure_ascii=False, indent=2))


if __name__ == "__main__":
	main()

'''
## Task
You are a senior psychological supervisor. Your task is to evaluate the performance of a counselor based on the dialogue between the counselor and the client. Analyze the counselor's performance in the dialogue and provide feedback on their strengths and areas for improvement. 

If the counselor's responses are appropriate and demonstrate good counseling skills, please provide positive feedback. 

If the counselor's responses are inappropriate, unprofessional, or indicate a lack of counseling skills, please provide constructive criticism and give correct responses that the counselor could have used instead.


## Therapist and Client Dialogue Transcript
```
{{history}}
```

## Format
1.  **Brief Analysis:**
2.  **Suggestions for Improvement:**



'''
