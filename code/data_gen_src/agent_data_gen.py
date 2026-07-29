import asyncio
import json
import os
import argparse
import random
random.seed(42) 
from uuid import uuid4
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
from openai import OpenAI
import time
from tqdm import tqdm
from datetime import datetime
try:
    from autogen_ext.models.ollama import OllamaChatCompletionClient
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False



class MakeData:
    def __init__(self, model_type="api", model_name=None, api_base=None, api_key=None, 
                 ollama_base_url=None, output_data_path=None, patient_info_path=None, mistake_path=None,
                 patient_start_num=0, patient_end_num=1, counselor_start_num=-1, counselor_end_num=-1, sample_num=1,
                 max_turns=5, counselor_temp=None, patient_temp=None, test_mode=False, max_concurrent=10):
        self.model_type = model_type
        self.model_name = model_name
        self.api_base = api_base
        self.api_key = api_key
        self.ollama_base_url = ollama_base_url
        self.max_turns = max_turns
        self.counselor_temp = counselor_temp
        self.patient_temp = patient_temp

        self.patient_start_num=patient_start_num
        self.counselor_start_num= counselor_start_num

        self.patient_end_num = patient_end_num
        self.counselor_end_num = counselor_end_num
        self.sample_num = sample_num

        self.test_mode = test_mode
        self.max_concurrent = max_concurrent

        self.init_message = "Hello. How are you feeling today?"

        self.paths = {
            'output_path': output_data_path or '',
            'patient_info': '',
            'mistake': '',
            "counselor_prompt_path": '',
            "patient_prompt_path": '',
            "supervisor_prompt_path": '',
        }

        with open(self.paths['counselor_prompt_path'], 'r', encoding='utf-8') as f:
            self.counselor_prompt_template = f.read()
        with open(self.paths['patient_prompt_path'], 'r', encoding='utf-8') as f:
            self.patient_prompt_template = f.read()
        with open(self.paths['supervisor_prompt_path'], 'r', encoding='utf-8') as f:
            self.supervisor_prompt_template = f.read()

        if not os.path.exists(os.path.dirname(self.paths['output_path'])):
            os.makedirs(os.path.dirname(self.paths['output_path']))

        self.config_list = self._get_config_list()
        self.patient = None
        self.counselor = None
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        self.prompts = self._generate_prompts()

    def _get_api_config(self):
        model_name = self.model_name or ""
        api_base = self.api_base or ""
        api_key = self.api_key or ""
        return {
            "model": model_name,
            "base_url": api_base,
            "api_key": api_key,
        }

    def _get_ollama_config(self):
        model_name = self.model_name or "llama3.1:latest"
        ollama_base_url = self.ollama_base_url or "http://localhost:11434"
        if not OLLAMA_AVAILABLE:
            raise ImportError("Error1")
        ollama_client = OllamaChatCompletionClient(
            model=model_name,
            base_url=ollama_base_url
        )
        return {
            "model": model_name,
            "client": ollama_client
        }

    def _get_config_list(self):
        if self.model_type == "api":
            return [self._get_api_config()]
        elif self.model_type == "ollama":
            return [self._get_ollama_config()]
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}。")

    def _load_patient_info(self):
        with open(self.paths['patient_info'], 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_mistakes(self):
        with open(self.paths['mistake'], 'r', encoding='utf-8') as f:
            return json.load(f)

    def _generate_prompts(self):
        prompts = []
        patient_info = self._load_patient_info()
        mistakes = self._load_mistakes()
        for i in range(self.patient_start_num, self.patient_end_num):
            for j in range(self.counselor_start_num, self.counselor_end_num):
                patient_item = patient_info[i % len(patient_info)]
                mistake_item = mistakes[j % len(mistakes)]
                patient_prompt = self.patient_prompt_template.format(name=patient_item['name'], cognitive_model=patient_item['cognitive_model']) # ,client_reaction=mistake_item['client_reaction']
                conselor_prompt = self.counselor_prompt_template.format(mistakeType=mistake_item['mistakeType'], mistakeContent=mistake_item['mistakeContent'])
                supervisor_prompt = self.supervisor_prompt_template.format(mistakeType=mistake_item['mistakeType'], mistakeContent=mistake_item['mistakeContent'],supervisor_criteria=mistake_item['supervisor_criteria'])
                prompts.append({
                    "key": f"{patient_item['id']}_{mistake_item['id']}",
                    "mistake_id": mistake_item['id'],
                    "patient_id": patient_item['id'],
                    "mistakeType": mistake_item['mistakeType'],
                    "patient_prompt": patient_prompt,
                    "counselor_prompt": conselor_prompt,
                    "supervisor_prompt": supervisor_prompt,
                    "patient": patient_item,
                    "mistake": mistake_item })
        return prompts

    def create_agents(self, patient_prompt, counselor_prompt):
        patient = self._create_patient_agent(patient_prompt)
        counselor = self._create_counselor_agent(counselor_prompt)
        client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        return counselor, patient, client

    def _create_patient_agent(self, prompt):
        temperature = self.patient_temp if self.patient_temp is not None else 0.15
        return AssistantAgent(
            name="patient", 
            code_execution_config={"work_dir": "coding", "use_docker": False},
            llm_config={
                "config_list": self.config_list,
                "temperature": temperature,
            },
            human_input_mode="NEVER",
            system_message=prompt,
        )

    def _create_counselor_agent(self, prompt):
        temperature = self.counselor_temp if self.counselor_temp is not None else 0.1
        return UserProxyAgent(
            name="counselor", 
            llm_config={
                "config_list": self.config_list,
                "temperature": temperature,
            },
            human_input_mode="NEVER", 
            system_message=prompt,
            code_execution_config={"use_docker": False},
        )

    def start_conversation(self, counselor, patient, test_mode=False):
        if test_mode:
            return ["测试模式"]   
        else:
            return counselor.initiate_chat(
                patient,
                message=self.init_message,
                max_turns=random.choice(self.max_turns)
            )

    def process_chat_history(self, chat_result,test_mode=False):
        chat_history = []
        history_string = ""
        if test_mode:
            return chat_history, history_string
        for i, message in enumerate(chat_result.chat_history):
            sender_name = message["name"]
            content = message["content"].strip()
            role = "Therapist" if sender_name == "counselor" else "Client"
            history_string += f"{role}: {content}\t"
            chat_history.append({"role": role, "content": content})
        return chat_history, history_string

    def save_chat_history(self, uuid, patient_id,mistake_id, chat_history, history_string, critic_content, critic_reasoning=None, classify_result=None, file_path=None):
        file_path = file_path or self.paths['output_path']
        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
        content = {"uuid": uuid, "patient_id": patient_id, "mistake_id": mistake_id, "history": chat_history, "history_string": history_string, "origin_critic": critic_content} # , "origin_critic_reasoning": critic_reasoning, "classify_result": classify_result
        with open(file_path, "a", encoding="utf-8") as f:
            json.dump(content, f, ensure_ascii=False) 
            f.write("\n")
        return file_path

    def get_critic_content(self, history, supervisor_user_prompt, test_mode=False):
        supervisor_user_prompt = supervisor_user_prompt.format(history=history)
        if test_mode:
            return "测试模式批评内容", "测试模式批评推理内容"
        try:
            response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": supervisor_user_prompt}],
                    stream=False,
                    temperature=0,
                    max_tokens=4000,
                )
            content = response.choices[0].message.content
            reasoning_content = response.choices[0].message.reasoning_content if hasattr(response.choices[0].message, 'reasoning_content') else None
        except Exception as e:
            raise ValueError(f"critic生成失败: {e}")
        return content, reasoning_content

    def mistake_classify(self, history, test_mode=False):
        if test_mode:
            return "测试模式错误分类"
        with open("F:\Code\src\makeData_2\prompt\mistake_classify.txt", "r", encoding="utf-8") as f:
            mistake_classify_prompt = f.read()
        mistake_classify_prompt = mistake_classify_prompt.format(history=history)
        #print(mistake_classify_prompt)
        try:
            response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": mistake_classify_prompt}],
                    stream=False,
                    temperature=0,
                    max_tokens=4000,
                )
            content = response.choices[0].message.content
            reasoning_content = response.choices[0].message.reasoning_content if hasattr(response.choices[0].message, 'reasoning_content') else None
        except Exception as e:
            raise ValueError(f"生成失败: {e}")
        
        return content,reasoning_content

    async def process_single_prompt(self, prompt, sample_idx, semaphore):
        key = prompt['key']
        patient_id = prompt['patient_id']
        mistake_id = prompt['mistake_id']
        patient_prompt = prompt['patient_prompt']
        counselor_prompt = prompt['counselor_prompt']
        supervisor_prompt = prompt['supervisor_prompt']
        uuid = str(uuid4())[:7]
        async with semaphore:
            try:
                loop = asyncio.get_running_loop()

                counselor, patient, client = await loop.run_in_executor(None, self.create_agents, patient_prompt, counselor_prompt)
        
                chat_result = await loop.run_in_executor(None, self.start_conversation, counselor, patient, self.test_mode)
                chat_history, history_string = await loop.run_in_executor(None, self.process_chat_history, chat_result, self.test_mode)
                
                # 使用任务特定的client实例进行API调用
                critic_content, critic_reasoning = None, None
                if not self.test_mode:
                    supervisor_user_prompt = supervisor_prompt.format(history=history_string)
                    response = await loop.run_in_executor(
                        None,
                        lambda: client.chat.completions.create(
                            model=self.model_name,
                            messages=[{"role": "user", "content": supervisor_user_prompt}],
                            stream=False,
                            temperature=0,
                            max_tokens=4000,
                        )
                    )
                    critic_content = response.choices[0].message.content
                    critic_reasoning = response.choices[0].message.reasoning_content if hasattr(response.choices[0].message, 'reasoning_content') else None
                else:
                    critic_content, critic_reasoning = "测试模式批评内容", "测试模式批评推理内容"
                
                mistake_classify_result = None
                
                # 返回结果而不是直接保存
                return {
                    "uuid": uuid,
                    "patient_id": patient_id,
                    "mistake_id": mistake_id,
                    "chat_history": chat_history,
                    "history_string": history_string,
                    "critic_content": critic_content,
                    "critic_reasoning": critic_reasoning,
                    "mistake_classify_result": mistake_classify_result,
                }
            except Exception as e:
                print(f"处理 {key} 第 {sample_idx+1} 次采样失败: {e}")
                return None


def parse_arguments():
    parser = argparse.ArgumentParser(description='生成医生和患者之间的对话')
    parser.add_argument('--model-type', type=str, default='api',
                        choices=['api', 'ollama'],
                        help='模型类型: api 或 ollama (默认: api)')
    parser.add_argument('--model-name', type=str, default="deepseek-chat",  # deepseek官网：deepseek-chat  goachao：deepseek-v3-0324
                        help='模型名称')
    parser.add_argument('--api-base', type=str, default="",
                        help='API基础URL (仅用于API模式)')
    parser.add_argument('--api-key', type=str,default="",
                        help='API密钥 (仅用于API模式)')
    parser.add_argument('--ollama-url', type=str, default=None,
                        help='Ollama服务器URL (仅用于Ollama模式)')
    parser.add_argument('--patient_start_num', type=int, default=80,
                        help='患者信息起始编号 (默认: 0)')
    parser.add_argument('--patient_end_num', type=int, default=106,
                        help='患者信息结束编号 (默认: 1)')
    
    parser.add_argument('--counselor_start_num', type=int, default=0,
                        help='错误模板起始编号 (默认: 0)')
    parser.add_argument('--counselor_end_num', type=int, default=16)
    parser.add_argument('--sample_num', type=int, default=1,
                        help='采样次数 (默认: 1)')
    parser.add_argument('--max_turns', type=int, default=[5,6,7],
                        help='最大对话轮次 (默认: 5)')
    parser.add_argument('--patient_info_path', type=str, default='',
                        help='患者信息文件路径')
    parser.add_argument('--mistake_path', type=str, default='',
                        help='错误模板文件路径')
    parser.add_argument('--output_path', type=str, default='')
    parser.add_argument('--counselor_temp', type=float, default=0.05)
    parser.add_argument('--patient_temp', type=float, default=0.05,
                        help='患者智能体的temperature参数')
    parser.add_argument('--test_mode', type=bool, default=False, help='启用测试模式')
    parser.add_argument('--max_concurrent', type=int, default=30, help='最大并发线程数')
    return parser.parse_args()

def main():
    args = parse_arguments()
    try:
        data_maker = MakeData(
            model_type=args.model_type,
            model_name=args.model_name,
            api_base=args.api_base,
            api_key=args.api_key,
            ollama_base_url=args.ollama_url,
            output_data_path=args.output_path,
            patient_info_path=args.patient_info_path,
            mistake_path=args.mistake_path,
            patient_start_num=args.patient_start_num,
            patient_end_num=args.patient_end_num, 
            
            counselor_start_num=args.counselor_start_num,
            counselor_end_num=args.counselor_end_num,
            
            sample_num=args.sample_num,
            max_turns=args.max_turns,
            
            counselor_temp=args.counselor_temp,
            patient_temp=args.patient_temp,
            
            test_mode=args.test_mode,
            max_concurrent=args.max_concurrent
        )
        asyncio.run(data_maker.run_async())
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()