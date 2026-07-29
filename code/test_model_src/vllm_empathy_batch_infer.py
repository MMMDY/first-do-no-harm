import json
import os
import threading
import time
from tqdm import tqdm
from openai import OpenAI
from queue import Queue



class MakeDpoData:
    def __init__(self, start_line=0, end_line=None):

        
        # self.api_base ="https://api.deepseek.com/v1"
        # self.api_key="sk-960253e600d141de9a0577ae5eb65ba0"
        
        self.api_base ="http://0.0.0.0:8000/v1"
        self.api_key="none"
        
        self.model_name="/WX24061/lzy/models/Qwen3-8B-Instruct-mate5-cot-v1-250step"
        
        self.enable_thinking = True  # 是否启用思考模式，默认为 False
        
        
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)  
        self.data_queue = Queue()
        self.result_lock = threading.Lock()
        self.progress_lock = threading.Lock()
        self.processed_count = 0
        self.total_items = 0
        self.pbar = None
        self.start_line = start_line  # 添加起始行属性
        self.end_line = end_line  # 添加结束行属性
        


    def get_client(self, system_prompt, user_prompt):
        client = OpenAI(api_key=self.api_key, base_url=self.api_base)  
        max_retries = 10  # 最大重试次数
        retry_delay = 8  # 每次重试的间隔时间（秒）

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    #model="gpt-4o",
                    model=self.model_name,
                    messages=[{"role": "system", "content": system_prompt},
                              {"role": "user", "content": user_prompt}],
                    stream=False,
                    temperature=0.01,
                    max_tokens=2048,
                    #extra_headers={"lora_id": "0"},  
                    #stream_options={"include_usage": True},
                    extra_body={"chat_template_kwargs": {
                "enable_thinking": self.enable_thinking  # 关闭思考模式。改为 True 则开启。
            }}
                )
                
                # 提取 content 和 reasoning_content
                content = response.choices[0].message.content
                reasoning_content = response.choices[0].message.reasoning_content if hasattr(response.choices[0].message, 'reasoning_content') else None
                
                return content, reasoning_content  # 返回 content 和 reasoning_content

            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:  # 如果不是最后一次尝试
                    time.sleep(retry_delay)  # 等待指定的时间再重试
        return None, None  # 如果所有尝试都失败，返回 None
    

    
    def worker(self, thread_id, output_data_path):
        while True:
            try:
                item = self.data_queue.get_nowait()
            except:
                break

            try:
                
                # {"custom_id": "ER-4ibahh-d2wmrq8-fc361c", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "deepseek-r1", "messages": [{"role": "system", "content": "In the context of empathy, there are three key aspects to consider: (1) Emotional Reactions – expressing emotions like warmth, compassion, and concern that the peer supporter feels after reading the seeker's post; (2) Interpretations – conveying an understanding of the feelings and experiences inferred from the seeker's post; (3) Explorations – seeking a deeper understanding of the seeker by delving into feelings and experiences not explicitly stated in the post. Each aspect can exhibit varying degrees of communication—none, weak, or strong—based on the manner in which related content is expressed. The overall level of empathy is determined by the highest level achieved across these three aspects.\n\nYour task is to identify the level of empathy in the Supporter's response within the provided conversation."}, {"role": "user", "content": "Help-Seeker says: 'why do I feel this. Why do us as humans feel the need to love, and be loved in return. And why if we dont recieve this level of effection do we turn to depression. Is our depression a weird form of coping cause by a lack of affection?'\nSupporter says: 'Can't really explain why but it is just a basic need'\nIdentify the empathy level of the Supporter's response. Choose one of the following options: No Empathy, Weak Empathy, Strong Empathy. JUST output the option, do NOT output any other information."}], "max_completion_tokens": 64, "temperature": 0}}
                #print(item)
                custom_id = item["custom_id"]
                system_prompt = item["body"]["messages"][0]["content"]
                user_prompt = item["body"]["messages"][1]["content"]
                print("--------------------------------")
                print(custom_id)
                # print(system_prompt)
                # print(user_prompt)
                
                content, reasoning_content = self.get_client(system_prompt, user_prompt)
                print(content)
                
                cur_dpo_data = {"id":"vllm-9fe62a33b3514b569d3de1cc5db9e6d4","custom_id":"","response":{"status_code":200,"request_id":"vllm-batch-114b0007f2cf4043b0c61b93a2b090d4","body":{"id":"chatcmpl-5cd123719099488a935aa85f6a755d55","object":"chat.completion","created":1744043847,"model":"deepseek r1","choices":[{"index":0,"message":{"role":"assistant","reasoning_content":"null","content":"","tool_calls":[]},"logprobs":"null","finish_reason":"stop","stop_reason":"null"}],"usage":{"prompt_tokens":335,"total_tokens":339,"completion_tokens":4,"prompt_tokens_details":"null"},"prompt_logprobs":"null"}},"error":""}
                
                cur_dpo_data["custom_id"] = custom_id
                cur_dpo_data["response"]["body"]["choices"][0]["message"]["content"] = content


                with self.result_lock:
                    with open(output_data_path, "a", encoding="utf-8") as output_file:
                        json.dump(cur_dpo_data, output_file, ensure_ascii=False)
                        output_file.write("\n")
                
                with self.progress_lock:
                    self.processed_count += 1
                    self.pbar.update(1)
                    self.pbar.set_description(f"线程 {thread_id:02d} ") # 处理: {item['mistake_id']}

            except Exception as e:
                print(f"\n线程 {thread_id} 处理数据时出错: {str(e)}")
            finally:
                self.data_queue.task_done()

    def make_dpo_data(self, input_file, output_data_path):
        print(f"初始化处理...从第 {self.start_line} 行开始")
        if self.end_line:
            print(f"将处理到第 {self.end_line} 行")
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
        
        # 如果是从头开始，则创建新文件；否则追加模式
        if self.start_line == 0:
            open(output_data_path, "w", encoding="utf-8").close()
        
        input_data_path = input_file
        
        # 读取所有数据并放入队列
        all_items = []
        current_line = 0
        with open(input_data_path, "r", encoding="utf-8") as input_file:
            for line in input_file:
                current_line += 1
                if current_line < self.start_line:
                    continue
                if self.end_line and current_line > self.end_line:
                    break
                item = json.loads(line)
                all_items.append(item)
        
        self.total_items = len(all_items)
        print(f"从第 {self.start_line} 行开始，总共发现 {self.total_items} 条数据待处理")
        
        if self.total_items == 0:
            print("没有需要处理的数据！")
            return
        
        # 初始化进度条
        self.pbar = tqdm(total=self.total_items, desc="总体进度", unit="条")
        
        # 将数据放入队列
        for item in all_items:
            self.data_queue.put(item)

        # 创建并启动线程
        threads = []
        num_threads = min(50, self.total_items)  # 根据数据量动态调整线程数
        print(f"启动 {num_threads} 个工作线程...")
        for i in range(num_threads):
            t = threading.Thread(target=self.worker, args=(i, output_data_path))
            t.start()
            threads.append(t)

        # 等待所有线程完成
        for t in threads:
            t.join()
        
        self.pbar.close()
        print(f"\n处理完成！从第 {self.start_line} 行开始，共处理 {self.processed_count} 条数据")
        print(f"输出文件：{output_data_path}")


# 从第1行开始处理，到第100行结束
mkdata = MakeDpoData(start_line=1)

input_file = "/WX24061/lzy/test_model/empathy_infer/deepseek_r1.jsonl"

name = "Qwen3-8B-Instruct-mate5-cot-v1-250step"

out_data_path = f"/WX24061/lzy/test_model/empathy/{name}.jsonl"


mkdata.make_dpo_data(input_file, out_data_path)
