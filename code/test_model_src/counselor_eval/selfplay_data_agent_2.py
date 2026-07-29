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
from typing import Optional, Tuple, Dict, Any, List

try:
    from autogen_ext.models.ollama import OllamaChatCompletionClient
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# CUDA_VISIBLE_DEVICES=2 vllm serve /WX24061/lzy/Models/qwen3-8b-instruct --max-model-len 4096 --gpu-memory-utilization 0.95 --port 8177


class MakeData:
    def __init__(self, model_type="api", model_name=None, api_base=None, api_key=None, 
                 patient_api_base=None, patient_api_key=None,
                 counselor_api_base=None, counselor_api_key=None,
                 supervisor_api_base=None, supervisor_api_key=None,
                 ollama_base_url=None, output_data_path=None, patient_info_path=None, mistake_path=None,
                 patient_start_num=0, patient_end_num=1, counselor_start_num=-1, counselor_end_num=-1, sample_num=1,
                 max_turns=5, counselor_temp=None, patient_temp=None, test_mode=False, max_concurrent=10,
                 group="B", keep_c_internal_history=False):
        self.model_type = model_type
        self.model_name = model_name
        self.api_base = api_base
        self.api_key = api_key
        self.patient_api_base = patient_api_base or api_base
        self.patient_api_key = patient_api_key or api_key
        self.counselor_api_base = counselor_api_base or api_base
        self.counselor_api_key = counselor_api_key or api_key
        self.supervisor_api_base = supervisor_api_base or api_base
        self.supervisor_api_key = supervisor_api_key or api_key
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
        self.group = str(group).upper()
        self.keep_c_internal_history = keep_c_internal_history
        if self.group not in ["B", "C"]:
            raise ValueError("group 仅支持 'B' 或 'C'")

        self.init_message = "Hello. How are you feeling today?"

        self.paths = {
            'output_path': output_data_path or '/WX24061/lzy/test_model/counselor_eval/gen_conver/output_data/test/test_2.jsonl',
            'patient_info': '/WX24061/lzy/test_model/counselor_eval/gen_conver/Patient-Psi-CM_dataset_2.json',
            'mistake': '/WX24061/lzy/test_model/counselor_eval/gen_conver/mistake_3.json',
            "normal_counselor_prompt_path": '/WX24061/lzy/test_model/counselor_eval/gen_conver/prompt/normal_counselor_prompt.txt',
            "mistake_counselor_prompt_path": '/WX24061/lzy/test_model/counselor_eval/gen_conver/prompt/counselor_prompt.txt',
            "patient_prompt_path": '/WX24061/lzy/test_model/counselor_eval/gen_conver/prompt/patient_prompt.txt',
            "supervisor_prompt_path": '/WX24061/lzy/test_model/counselor_eval/gen_conver/prompt/supervisor_prompt.txt',
            #'failed_log': 'src/makeData_2/output_data/failed_cases.json',
            #'raw_responses': 'src/makeData_2/output_data/raw_responses.json'
        }
        self.paths["counselor_prompt_path"] = (
            self.paths["mistake_counselor_prompt_path"] if self.group == "C" else self.paths["normal_counselor_prompt_path"]
        )

        with open(self.paths['counselor_prompt_path'], 'r', encoding='utf-8') as f:
            self.counselor_prompt_template = f.read()
        with open(self.paths['patient_prompt_path'], 'r', encoding='utf-8') as f:
            self.patient_prompt_template = f.read()
        with open(self.paths['supervisor_prompt_path'], 'r', encoding='utf-8') as f:
            self.supervisor_prompt_template = f.read()

        if not os.path.exists(os.path.dirname(self.paths['output_path'])):
            os.makedirs(os.path.dirname(self.paths['output_path']))

        self.patient_config_list = self._get_config_list(role="patient")
        self.counselor_config_list = self._get_config_list(role="counselor")
        self.patient = None
        self.counselor = None
        self.patient_client = OpenAI(api_key=self.patient_api_key, base_url=self.patient_api_base)
        self.counselor_client = OpenAI(api_key=self.counselor_api_key, base_url=self.counselor_api_base)
        self.supervisor_client = OpenAI(api_key=self.supervisor_api_key, base_url=self.supervisor_api_base)
        self.prompts = self._generate_prompts()

    def _get_api_config(self, role="default"):
        model_name = self.model_name or "x1"
        if role == "patient":
            api_base = self.patient_api_base
            api_key = self.patient_api_key
        elif role == "counselor":
            api_base = self.counselor_api_base
            api_key = self.counselor_api_key
        elif role == "supervisor":
            api_base = self.supervisor_api_base
            api_key = self.supervisor_api_key
        else:
            api_base = self.api_base
            api_key = self.api_key
        return {
            "model": model_name,
            "base_url": api_base,
            "api_key": api_key,
        }

    def _get_ollama_config(self):
        model_name = self.model_name or "llama3.1:latest"
        ollama_base_url = self.ollama_base_url or "http://localhost:11434"
        if not OLLAMA_AVAILABLE:
            raise ImportError("使用Ollama需要安装autogen_ext包。请使用 pip install autogen_ext 安装。")
        ollama_client = OllamaChatCompletionClient(
            model=model_name,
            base_url=ollama_base_url
        )
        return {
            "model": model_name,
            "client": ollama_client
        }

    def _get_config_list(self, role="default"):
        if self.model_type == "api":
            return [self._get_api_config(role=role)]
        elif self.model_type == "ollama":
            return [self._get_ollama_config()]
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}。请使用 'api' 或 'ollama'。")

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
        return counselor, patient

    def _create_patient_agent(self, prompt):
        temperature = self.patient_temp if self.patient_temp is not None else 0.15
        return AssistantAgent(
            name="patient", 
            code_execution_config={"work_dir": "coding", "use_docker": False},
            llm_config={
                "config_list": self.patient_config_list,
                "temperature": temperature,
                "extra_body":{"chat_template_kwargs": {
                "enable_thinking": False  # 关闭思考模式。改为 True 则开启。
            }}
            },
            human_input_mode="NEVER",
            system_message=prompt,
        )

    def _create_counselor_agent(self, prompt):
        temperature = self.counselor_temp if self.counselor_temp is not None else 0.1
        return UserProxyAgent(
            name="counselor", 
            llm_config={
                "config_list": self.counselor_config_list,
                "temperature": temperature,
                "extra_body":{"chat_template_kwargs": {
                "enable_thinking": False  # 关闭思考模式。改为 True 则开启。
            }}
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
                # max_turns是一个list，每次随机选择list其中的一个值
                max_turns=random.choice(self.max_turns)
            )

    def _sample_max_turns(self):
        if isinstance(self.max_turns, int):
            return self.max_turns
        if isinstance(self.max_turns, list) and self.max_turns:
            return random.choice(self.max_turns)
        return 10

    def _chat_with_client(self, client: OpenAI, system_prompt: str, messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int = 512):
        if self.test_mode:
            return "测试模式回复"
        request_messages = [{"role": "system", "content": system_prompt}] + messages
        response = client.chat.completions.create(
            model=self.model_name,
            messages=request_messages,
            stream=False,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {
                "enable_thinking": False
            }}
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""

    def _history_to_string(self, history):
        return "\n".join([f"{item['role']}: {item['content']}" for item in history if item.get("content")])

    def run_conversation_c_group(self, patient_prompt, counselor_prompt, supervisor_prompt, test_mode=False):
        if test_mode:
            return {
                "chat_history": [],
                "history_string": "",
                "supervisor_feedback_summary": "测试模式批评内容",
                "turn_trace": []
            }

        context_history = []
        chat_history = []
        turn_trace = []
        max_turns = self._sample_max_turns()

        init_item = {"role": "Therapist", "content": self.init_message}
        context_history.append(init_item)
        chat_history.append(init_item)

        init_patient_messages = [{"role": "user", "content": self._history_to_string(context_history)}]
        init_patient_reply = self._chat_with_client(
            self.patient_client,
            patient_prompt,
            init_patient_messages,
            temperature=self.patient_temp if self.patient_temp is not None else 0.15,
            max_tokens=512
        )
        context_history.append({"role": "Client", "content": init_patient_reply})
        chat_history.append({"role": "Client", "content": init_patient_reply})

        for _ in range(max_turns):
            counselor_messages = [{"role": "user", "content": self._history_to_string(context_history)}]
            counselor_draft = self._chat_with_client(
                self.counselor_client,
                counselor_prompt,
                counselor_messages,
                temperature=self.counselor_temp if self.counselor_temp is not None else 0.1,
                max_tokens=512
            )

            history_for_supervisor = self._history_to_string(context_history + [{"role": "TherapistDraft", "content": counselor_draft}])
            supervisor_feedback, _ = self.get_critic_content(history_for_supervisor, supervisor_prompt, test_mode=False)
            #print(f"history_for_supervisor:\n{history_for_supervisor}\n")

            revise_instruction = (
                "Rewrite your response for this turn based on the content below. "
                "Output only the final revised therapist reply in 1-2 sentences.\n"
                f"Original reply: {counselor_draft}\n"
                f"Supervisor feedback: {supervisor_feedback}\n"
                "Provide the revised version:"
            )
            revised_messages = [
                {"role": "user", "content": self._history_to_string(context_history)},
                {"role": "user", "content": revise_instruction}
            ]
            counselor_revised = self._chat_with_client(
                self.counselor_client,
                counselor_prompt,
                revised_messages,
                temperature=self.counselor_temp if self.counselor_temp is not None else 0.1,
                max_tokens=512
            )

            if self.keep_c_internal_history:
                chat_history.append({"role": "TherapistDraft", "content": counselor_draft})
                chat_history.append({"role": "Supervisor", "content": supervisor_feedback})

            context_history.append({"role": "Therapist", "content": counselor_revised})
            chat_history.append({"role": "Therapist", "content": counselor_revised})

            patient_messages = [{"role": "user", "content": self._history_to_string(context_history)}]
            patient_reply = self._chat_with_client(
                self.patient_client,
                patient_prompt,
                patient_messages,
                temperature=self.patient_temp if self.patient_temp is not None else 0.15,
                max_tokens=512
            )
            context_history.append({"role": "Client", "content": patient_reply})
            chat_history.append({"role": "Client", "content": patient_reply})

            turn_trace.append({
                "counselor_draft": counselor_draft,
                "supervisor_feedback": supervisor_feedback,
                "counselor_revised": counselor_revised,
                "patient_reply": patient_reply
            })

        history_string = self._history_to_string(chat_history)
        supervisor_feedback_summary = "\n\n".join(
            [f"Turn {idx + 1}: {item['supervisor_feedback']}" for idx, item in enumerate(turn_trace)]
        )
        return {
            "chat_history": chat_history,
            "history_string": history_string,
            "supervisor_feedback_summary": supervisor_feedback_summary,
            "turn_trace": turn_trace
        }

    def process_chat_history(self, chat_result,test_mode=False):
        chat_history = []
        history_string = ""
        if test_mode:
            return chat_history, history_string
        for i, message in enumerate(chat_result.chat_history):
            sender_name = message["name"]
            content = message["content"].strip()
            role = "Therapist" if sender_name == "counselor" else "Client"
            history_string += f"{role}: {content}\n"
            chat_history.append({"role": role, "content": content})
        return chat_history, history_string

    def save_chat_history(self, uuid, patient_id,mistake_id, chat_history, history_string, critic_content, critic_reasoning=None, classify_result=None, file_path=None, turn_trace=None, group=None):
        file_path = file_path or self.paths['output_path']
        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
        content = {
            "uuid": uuid,
            "patient_id": patient_id,
            "mistake_id": mistake_id,
            "group": group or self.group,
            "history": chat_history,
            "history_string": history_string,
            "origin_critic": critic_content,
            "turn_trace": turn_trace or []
        } # , "origin_critic_reasoning": critic_reasoning, "classify_result": classify_result
        with open(file_path, "a", encoding="utf-8") as f:
            json.dump(content, f, ensure_ascii=False) #  indent=4
            f.write("\n")
        return file_path

    def get_critic_content(self, history, supervisor_user_prompt, test_mode=False):
        supervisor_user_prompt = supervisor_user_prompt.format(history=history)
        if test_mode:
            return "测试模式批评内容", "测试模式批评推理内容"
        try:
            response = self.supervisor_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": supervisor_user_prompt}],
                    stream=False,
                    temperature=0,
                    max_tokens=1024,
                    extra_body={"chat_template_kwargs": {
                "enable_thinking": False  # 关闭思考模式。改为 True 则开启。
            }}
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
            response = self.supervisor_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": mistake_classify_prompt}],
                    stream=False,
                    temperature=0,
                    max_tokens=1024,
                    extra_body={"chat_template_kwargs": {
                "enable_thinking": False  # 关闭思考模式。改为 True 则开启。
            }}
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
                turn_trace = []
                critic_reasoning = None
                if self.group == "C":
                    c_result = await loop.run_in_executor(
                        None,
                        self.run_conversation_c_group,
                        patient_prompt,
                        counselor_prompt,
                        supervisor_prompt,
                        self.test_mode
                    )
                    chat_history = c_result["chat_history"]
                    history_string = c_result["history_string"]
                    critic_content = c_result["supervisor_feedback_summary"]
                    turn_trace = c_result["turn_trace"]
                else:
                    counselor, patient = await loop.run_in_executor(None, self.create_agents, patient_prompt, counselor_prompt)
                    chat_result = await loop.run_in_executor(None, self.start_conversation, counselor, patient, self.test_mode)
                    chat_history, history_string = await loop.run_in_executor(None, self.process_chat_history, chat_result, self.test_mode)

                    if not self.test_mode:
                        critic_content, critic_reasoning = await loop.run_in_executor(
                            None,
                            self.get_critic_content,
                            history_string,
                            supervisor_prompt,
                            self.test_mode
                        )
                    else:
                        critic_content, critic_reasoning = "测试模式批评内容", "测试模式批评推理内容"
                
                # 使用任务特定的client实例进行错误分类
                mistake_classify_result = None

                return {
                    "uuid": uuid,
                    "patient_id": patient_id,
                    "mistake_id": mistake_id,
                    "chat_history": chat_history,
                    "history_string": history_string,
                    "critic_content": critic_content,
                    "critic_reasoning": critic_reasoning,
                    "mistake_classify_result": mistake_classify_result,
                    "turn_trace": turn_trace,
                    "group": self.group,
                }
            except Exception as e:
                print(f"处理 {key} 第 {sample_idx+1} 次采样失败: {e}")
                return None

    async def run_async(self):
        if not self.prompts:
            raise ValueError("没有生成任何提示词，请检查患者信息和错误模板文件。")
        semaphore = asyncio.Semaphore(self.max_concurrent)
        pbar = tqdm(total=len(self.prompts)*self.sample_num, desc="生成对话", unit="条")
        batch_results = []
        tasks = []
        for idx, prompt in enumerate(self.prompts):
            for i in range(self.sample_num):
                tasks.append(self.process_single_prompt(prompt, i, semaphore))
        # 按批次执行并保存
        for i in range(0, len(tasks), self.max_concurrent):
            batch = tasks[i:i+self.max_concurrent]
            results = await asyncio.gather(*batch)
            batch_results.extend([r for r in results if r is not None])
            pbar.update(len(batch))
            # 批量保存
            if batch_results:
                for item in batch_results:
                    self.save_chat_history(
                        item["uuid"],
                        item["patient_id"],
                        item["mistake_id"],
                        item["chat_history"],
                        item["history_string"],
                        item["critic_content"],
                        item["critic_reasoning"],
                        item["mistake_classify_result"],
                        turn_trace=item.get("turn_trace"),
                        group=item.get("group", self.group)
                    )
                batch_results.clear()
        pbar.close()

def parse_arguments():
    parser = argparse.ArgumentParser(description='生成医生和患者之间的对话')
    parser.add_argument('--model-type', type=str, default='api',
                        choices=['api', 'ollama'],
                        help='模型类型: api 或 ollama (默认: api)')
    parser.add_argument('--model-name', type=str, default="/WX24061/lzy/Models/qwen3-8b-instruct",  # deepseek官网：deepseek-chat  goachao：deepseek-v3-0324
                        help='模型名称')

    #         api_key="sk-4137dec89d004743a28fd056b1b4b7c7",
    #         base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    # https://api.ai-gaochao.cn/v1  sk-ttfaXnv5KA5BfF2pF5E2A57c3f514e348cB47b9dA14f35B5
    # https://api.deepseek.com/v1
    # sk-960253e600d141de9a0577ae5eb65ba0
    parser.add_argument('--api-base', type=str, default="http://0.0.0.0:8177/v1",
                        help='API基础URL (仅用于API模式)')
    parser.add_argument('--api-key', type=str,default="",
                        help='API密钥 (仅用于API模式)')
    parser.add_argument('--patient-api-base', type=str, default="http://0.0.0.0:8177/v1",
                        help='患者角色API基础URL (为空则回退到 --api-base)')
    parser.add_argument('--patient-api-key', type=str, default="",
                        help='患者角色API密钥 (为空则回退到 --api-key)')
    parser.add_argument('--counselor-api-base', type=str, default="http://0.0.0.0:8177/v1",
                        help='咨询师角色API基础URL (为空则回退到 --api-base)')
    parser.add_argument('--counselor-api-key', type=str, default="",
                        help='咨询师角色API密钥 (为空则回退到 --api-key)')
    parser.add_argument('--supervisor-api-base', type=str, default="http://0.0.0.0:8177/v1",
                        help='督导角色API基础URL (为空则回退到 --api-base)')
    parser.add_argument('--supervisor-api-key', type=str, default="",
                        help='督导角色API密钥 (为空则回退到 --api-key)')
    parser.add_argument('--ollama-url', type=str, default=None,
                        help='Ollama服务器URL (仅用于Ollama模式)')
    parser.add_argument('--patient_start_num', type=int, default=0,
                        help='患者信息起始编号 (默认: 0)')
    parser.add_argument('--patient_end_num', type=int, default=20,
                        help='患者信息结束编号 (默认: 1)')
    
    parser.add_argument('--counselor_start_num', type=int, default=0,
                        help='错误模板起始编号 (默认: 0)')
    parser.add_argument('--counselor_end_num', type=int, default=15)
    parser.add_argument('--sample_num', type=int, default=1,
                        help='采样次数 (默认: 1)')
    parser.add_argument('--max_turns', type=int, nargs='+', default=[10],
                        help='最大对话轮次 (默认: 5)')
    parser.add_argument('--patient_info_path', type=str, default='',
                        help='患者信息文件路径')
    parser.add_argument('--mistake_path', type=str, default='',
                        help='错误模板文件路径')
    parser.add_argument('--output_path', type=str, default='')
    parser.add_argument('--counselor_temp', type=float, default=0.3)
    parser.add_argument('--patient_temp', type=float, default=0.3,
                        help='患者智能体的temperature参数')
    parser.add_argument('--test_mode', type=bool, default=False, help='启用测试模式')
    parser.add_argument('--max_concurrent', type=int, default=30, help='最大并发线程数')
    parser.add_argument('--group', type=str, default='C', choices=['B', 'C'],
                        help='对话分组模式: B(正常咨询师) 或 C(易犯错咨询师+督导修正)')
    parser.add_argument('--keep_c_internal_history', action='store_true',
                        help='仅在 C 组生效：将 TherapistDraft/Supervisor 也写入 history')
    return parser.parse_args()

def main():
    args = parse_arguments()
    try:
        data_maker = MakeData(
            model_type=args.model_type,
            model_name=args.model_name,
            api_base=args.api_base,
            api_key=args.api_key,
            patient_api_base=args.patient_api_base or None,
            patient_api_key=args.patient_api_key or None,
            counselor_api_base=args.counselor_api_base or None,
            counselor_api_key=args.counselor_api_key or None,
            supervisor_api_base=args.supervisor_api_base or None,
            supervisor_api_key=args.supervisor_api_key or None,
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
            max_concurrent=args.max_concurrent,
            group=args.group,
            keep_c_internal_history=args.keep_c_internal_history
        )
        asyncio.run(data_maker.run_async())
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()