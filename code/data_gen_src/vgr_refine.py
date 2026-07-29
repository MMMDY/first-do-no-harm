import os
import re
import json
import argparse
import asyncio
from typing import Tuple, Optional, List, Dict, Any
from tqdm import tqdm
from openai import OpenAI


class CritiqueRefiner:

    # 模板1: 用于首次独立验证 origin_critic
    INITIAL_VALIDATE_PROMPT_TEMPLATE = """### Role: You are a quality assurance expert for psychological counseling training systems.
    
### Task
Your task is to evaluate whether the provided feedback (`critic`) meets professional standards based on the trainee-counselor conversation and a specified mistake type.

### Note
Use the following checklist to judge whether the `critic` meets the professional standards. For each item, return either true or false.

### Checklist
"1. Does the `critic` provide step-by-step, gradual feedback that guides the trainee to correct the mistake progressively?"
"2. Is the feedback specific and detailed, avoiding vague or overly general suggestions?"
"3. Is the `critic` aligned with ethical and professional standards in psychological counseling?"
"4. Is the tone of the feedback constructive, respectful, and non-judgmental?"

### Return Format:[true,false,true,false]

### Input
mistake_type: '''{mistake_type}'''

conversation: '''{conversation}'''

critic: '''{critic}'''
"""

    # 模板2: 用于精炼 critique
    REFINE_PROMPT_TEMPLATE = """### Role
    You are a professional psychological counseling expert, proficient in various counseling techniques, skilled in maintaining listening at appropriate points, actively building counseling relationships, flexibly using silence strategies, accurately interpreting visitors' expressions, and deeply empathizing with their inner feelings.
    
### Task
Give you a counseling dialogue history between a novice counselor and a visitor, because the novice counselor make some mistakes, so there are some criticism for the counselor.
You are tasked with refining the original criticism based on the conversation between the novice counselor and the visitor, output a more detailed and professional criticism for the counselor. Your refined criticism should serve as feedback for the novice, guiding them to recognize and correct their mistakes in a step-by-step, progressive manner. It needs to be more detailed and specific than the original, with clear, concrete suggestions that avoid vagueness or over-generality. Ensure your feedback strictly aligns with ethical and professional standards in psychological counseling, maintaining a constructive, respectful, and non-judgmental tone throughout. Structure your observations to move from surface-level observations to deeper, more nuanced insights, helping the counselor build awareness gradually and develop practical skills for improvement.

### Note
1. Provide you three things: 1. counselor mistake information. 2. the original criticism. 3. the counseling dialogue.
2. Only output the refined criticism, do not output any other content.

### Mistake information
Novice counselor make mistake type is: '''{mistake_type}'''. 
The description of the mistake is: '''{mistake_content}'''

### Counseling Dialogue: '''{conversation}'''

### Original Criticism: '''{origin_critic}'''

### Refined Criticism:"""

    # 模板3: 用于比较 origin_critic 和 refined_critic
    COMPARATIVE_VALIDATE_PROMPT_TEMPLATE = """### Role: You are a quality assurance expert for psychological counseling training systems.
    ### Objective
    Your task is to evaluate whether the revised feedback (`refined_critic`) is an improved version of the original feedback (`origin_critic`) based on a trainee-counselor conversation and a specified mistake type.
    
    ### Note
    Use the following checklist to judge whether the `refined_critic` meets the professional standards. For each item, return either true or false. Only if all items are true, the `refined_critic` can be accepted as valid.
    
    ### Checklist
    "1. Does the `critic` provide step-by-step, gradual feedback that guides the trainee to correct the mistake progressively?"
    "2. Is the `refined_critic` more detailed and specific than the `origin_critic`?"
    "3. Is the `refined_critic` aligned with ethical and professional standards in psychological counseling?"
    "4. Is the tone of the feedback constructive, respectful, and non-judgmental?"

### Return Format:[true,false,true,false]

### Input
mistake_type: '''{mistake_type}'''

conversation: '''{conversation}'''

origin_critic: '''{origin_critic}'''

refined_critic: '''{refined_critic}'''
"""
    
    def __init__(self, api_key: str, api_base: str):
        if not api_key or not api_base:
            raise ValueError("API key and base URL are required.")
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        print("✅ CritiqueRefiner initialized successfully.")

    def _call_llm(self, user_prompt: str) -> str:
        print(f"\n LLM CALL ".center(50, "-"))

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o", 
                messages=[{"role": "user", "content": user_prompt}],
                stream=False,
                temperature=0,
                max_tokens=8192,
                extra_headers={"lora_id": "0"},
                extra_body={"show_ref_label": True}
            )
            content = response.choices[0].message.content
            print("INFO: LLM call successful.")
            return content
        except Exception as e:
            print(f"❌ ERROR: An error occurred during the LLM API call: {e}")
            return ""

    async def _call_llm_async(self, user_prompt: str) -> str:
        print(f"\n LLM CALL ".center(50, "-"))

        try:
            response = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": user_prompt}],
                    stream=False,
                    temperature=0,
                    max_tokens=8192,
                    extra_headers={"lora_id": "0"},
                    extra_body={"show_ref_label": True}
                )
            )
            content = response.choices[0].message.content
            print("INFO: LLM call successful.")
            return content
        except Exception as e:
            print(f"❌ ERROR: An error occurred during the LLM API call: {e}")
            return ""

    def _parse_validation_output(self, output_str: str) -> List[bool]:
        try:
            matches = re.findall(r'true|false', output_str.lower())
            if not matches:
                print(f"⚠️ Warning: Validator LLM returned no boolean values. Output: '{output_str}'")
                return [False]
            return [val == 'true' for val in matches]
        except Exception as e:
            print(f"Error parsing validation output: {e}")
            return [False]

    def process_critique(
        self,
        mistake_type: str,
        mistake_content: str,
        conversation: str,
        origin_critic: str,
        N_retry: int = 3
    ) -> Tuple[Optional[str], Optional[str]]:
        # 1. 首次验证 (Initial Validation)
        print(f"\n{'='*20} STEP 1: Initial Validation of Original Critique {'='*20}")
        initial_prompt = self.INITIAL_VALIDATE_PROMPT_TEMPLATE.format(
            mistake_type=mistake_type,
            conversation=conversation,
            critic=origin_critic
        )
        print(f"📝 Initial Validation Checklist Prompt: {initial_prompt}")
        initial_validation_str = self._call_llm(initial_prompt)
        print(f"📝 Initial Validation Checklist Output: {initial_validation_str}")
        initial_validation_result = self._parse_validation_output(initial_validation_str)
        print(f"📝 Initial Validation Checklist Result: {initial_validation_result}")

        # 2. 判断与分流 (Decision Point)
        if all(initial_validation_result):
            print("\n✅ Original critique is already valid. No refinement needed.")
            return (origin_critic, "no_refine_needed")
        
        print("\n❌ Original critique failed validation. Starting refinement process...")
        
        # 3. 迭代精炼循环 (Refinement Loop)
        current_critic_to_refine = origin_critic
        for k in range(N_retry):
            print(f"\n{'='*20} STEP 2: Refinement Attempt {k+1}/{N_retry} {'='*20}")
            
            # 3a. Refine Stage
            refine_prompt = self.REFINE_PROMPT_TEMPLATE.format(
                mistake_type=mistake_type,
                mistake_content=mistake_content,
                conversation=conversation,
                origin_critic=current_critic_to_refine
            )
            refined_critic = self._call_llm(refine_prompt)
            if not refined_critic:
                print("❌ Refinement failed due to API error. Retrying...")
                continue
            print(f"✅ Refined Critique Generated: '{refined_critic}'")

            # 3b. Comparative Validate Stage
            compare_prompt = self.COMPARATIVE_VALIDATE_PROMPT_TEMPLATE.format(
                mistake_type=mistake_type,
                conversation=conversation,
                origin_critic=origin_critic,
                refined_critic=refined_critic
            )
            validation_output_str = self._call_llm(compare_prompt)
            validation_result = self._parse_validation_output(validation_output_str)
            print(f"📝 Comparative Validation Checklist Result: {validation_result}")
            
            # 3c. Check for success
            if all(validation_result):
                print(f"🎉 Validation successful on attempt {k+1}! Returning the refined critique.")
                return (refined_critic, "refine_success")
            else:
                print(f"❌ Validation failed on attempt {k+1}. Retrying...")
                current_critic_to_refine = refined_critic

        # 4. 所有尝试都失败了 (Failure)
        print(f"\n🛑 All {N_retry} refinement attempts failed. Flagging for manual review.")
        return (None, "need_manual_check")

    async def process_critique_async(
        self,
        mistake_type: str,
        mistake_content: str,
        conversation: str,
        origin_critic: str,
        N_retry: int = 3
    ) -> Tuple[Optional[str], Optional[str]]:
        # Convert the synchronous process to async
        print(f"\n{'='*20} STEP 1: Initial Validation of Original Critique {'='*20}")
        
        initial_prompt = self.INITIAL_VALIDATE_PROMPT_TEMPLATE.format(
            mistake_type=mistake_type,
            conversation=conversation,
            critic=origin_critic
        )
        
        initial_validation_str = await self._call_llm_async(initial_prompt)
        initial_validation_result = self._parse_validation_output(initial_validation_str)

        if all(initial_validation_result):
            return (origin_critic, "no_refine_needed")
            
        current_critic_to_refine = origin_critic
        for k in range(N_retry):
            refine_prompt = self.REFINE_PROMPT_TEMPLATE.format(
                mistake_type=mistake_type,
                mistake_content=mistake_content,
                conversation=conversation,
                origin_critic=current_critic_to_refine
            )
            
            refined_critic = await self._call_llm_async(refine_prompt)
            if not refined_critic:
                continue

            compare_prompt = self.COMPARATIVE_VALIDATE_PROMPT_TEMPLATE.format(
                mistake_type=mistake_type,
                conversation=conversation,
                origin_critic=origin_critic,
                refined_critic=refined_critic
            )
            
            validation_output_str = await self._call_llm_async(compare_prompt)
            validation_result = self._parse_validation_output(validation_output_str)
            
            if all(validation_result):
                return (refined_critic, "refine_success")
            current_critic_to_refine = refined_critic

        return (None, "need_manual_check")


# ==============================================================================
# 步骤 2: 定义文件处理的辅助函数和主逻辑
# ==============================================================================

def format_conversation(history: List[Dict[str, str]]) -> str:

    if not history:
        return "No conversation history provided."
    
    # 将 "counselor" 和 "client" 首字母大写以提高可读性
    return "\n".join([f"{turn['role'].capitalize()}: {turn['content']}" for turn in history])


def process_jsonl_file(input_path: str, output_path: str, mistake_path: str, refiner: CritiqueRefiner, 
                      start_line: int = 0, end_line: Optional[int] = None):

    print(f"🚀 Starting processing of '{input_path}'...")
    print(f"📍 Processing range: from line {start_line} to {end_line if end_line else 'end'}")
    
    try:
        with open(mistake_path, 'r', encoding='utf-8') as mistake_file, \
             open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            mistakes = json.load(mistake_file)
            

            for _ in range(start_line):
                next(infile)
            
            for i, line in enumerate(infile, start=start_line):
        
                if end_line is not None and i >= end_line:
                    break
                    
                try:
                    data_record = json.loads(line)
                    uuid = data_record.get("uuid", f"unknown_uuid_{i+1}")
                    print(f"\n\n{'='*25} Processing record {i+1} (UUID: {uuid}) {'='*25}")

                    mistake_id = data_record.get("mistake_id")
                    mistake_id = int(mistake_id)

                    mistake_item = mistakes[mistake_id-1]
                    mistake_type = mistake_item["mistakeType"]
                    mistake_content = mistake_item['mistakeContent']
                    origin_critic = data_record.get("origin_critic", "")
                    conversation_str = data_record.get("history_string", [])

                    if not all([mistake_type, origin_critic, conversation_str]):
                        print(f"⚠️  Skipping record {uuid} due to missing critical data.")
                        continue

                    final_critique, flag = refiner.process_critique(
                        mistake_type=mistake_type,
                        mistake_content=mistake_content,
                        conversation=conversation_str,
                        origin_critic=origin_critic
                    )

                    data_record['refined_critique'] = final_critique
                    data_record['refine_status'] = flag
                    
                    outfile.write(json.dumps(data_record, ensure_ascii=False) + '\n')

                except json.JSONDecodeError:
                    print(f"❌ ERROR: Could not decode JSON from line {i+1}. Skipping.")
                except Exception as e:
                    print(f"❌ ERROR: An unexpected error occurred while processing line {i+1}: {e}")

        print(f"\n\n✅✅✅ Processing complete. Output saved to '{output_path}'.")

    except FileNotFoundError:
        print(f"❌ ERROR: Input file not found at '{input_path}'")


async def process_jsonl_file_async(input_path: str, output_path: str, mistake_path: str, 
                                 refiner: CritiqueRefiner, start_line: int = 0, 
                                 end_line: Optional[int] = None, max_concurrent: int = 10):
    
    print(f"🚀 Starting processing of '{input_path}'...")
    print(f"📍 Processing range: from line {start_line} to {end_line if end_line else 'end'}")
    
    try:
        with open(mistake_path, 'r', encoding='utf-8') as mistake_file:
            mistakes = json.load(mistake_file)
            
        with open(input_path, 'r', encoding='utf-8') as infile:
            lines = list(infile)[start_line:end_line]
            
        async def process_line(line: str, line_num: int):
            try:
                data_record = json.loads(line)
                uuid = data_record.get("uuid", f"unknown_uuid_{line_num}")
                print(f"\n\n{'='*25} Processing record {line_num} (UUID: {uuid}) {'='*25}")
                
                mistake_id = int(data_record.get("mistake_id"))
                mistake_item = mistakes[mistake_id-1]
                mistake_type = mistake_item["mistakeType"]
                mistake_content = mistake_item['mistakeContent']
                origin_critic = data_record.get("origin_critic", "")
                conversation_str = data_record.get("history_string", [])
                
                if not all([mistake_type, origin_critic, conversation_str]):
                    return None
                    
                final_critique, flag = await refiner.process_critique_async(
                    mistake_type=mistake_type,
                    mistake_content=mistake_content,
                    conversation=conversation_str,
                    origin_critic=origin_critic
                )
                
                data_record['refined_critique'] = final_critique
                data_record['refine_status'] = flag
                return data_record
                
            except Exception as e:
                print(f"❌ ERROR processing line {line_num}: {e}")
                return None
                
        semaphore = asyncio.Semaphore(max_concurrent)
        async def bounded_process_line(line: str, line_num: int):
            async with semaphore:
                return await process_line(line, line_num)
                
        tasks = [bounded_process_line(line, i) for i, line in enumerate(lines, start=start_line)]
        
        with tqdm(total=len(tasks), desc="Processing records") as pbar:
            results = []
            for batch in range(0, len(tasks), max_concurrent):
                batch_tasks = tasks[batch:batch + max_concurrent]
                batch_results = await asyncio.gather(*batch_tasks)
                results.extend([r for r in batch_results if r is not None])
                pbar.update(len(batch_tasks))
                
                # Write batch results
                with open(output_path, 'a', encoding='utf-8') as outfile:
                    for result in results:
                        json.dump(result, outfile, ensure_ascii=False)
                        outfile.write('\n')
                results.clear()
                
        print(f"\n\n✅✅✅ Processing complete. Output saved to '{output_path}'.")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        raise


# ==============================================================================
# 步骤 3: 脚本执行入口
# ==============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description='Refine counseling critique data')
    parser.add_argument('--input-file', type=str, 
                        default="",
                        help='Input JSONL file path')
    parser.add_argument('--output-file', type=str,
                        default="",
                        help='Output JSONL file path')
    parser.add_argument('--mistake-file', type=str,
                        default="",
                        help='Mistake types file path')
    parser.add_argument('--start-line', type=int, default=100,
                        help='Starting line number to process')
    parser.add_argument('--end-line', type=int, default=200,
                        help='Ending line number to process')
    parser.add_argument('--max-concurrent', type=int, default=50,
                        help='Maximum number of concurrent tasks')
    parser.add_argument('--api-key', type=str,
                        default="s",
                        help='API key for the LLM service')
    parser.add_argument('--api-base', type=str,
                        default="",
                        help='Base URL for the LLM service')
    return parser.parse_args()


async def main_async():
    args = parse_arguments()
    
    try:
        refiner_instance = CritiqueRefiner(api_key=args.api_key, api_base=args.api_base)
        await process_jsonl_file_async(
            args.input_file,
            args.output_file,
            args.mistake_file,
            refiner_instance,
            start_line=args.start_line,
            end_line=args.end_line,
            max_concurrent=args.max_concurrent
        )
    except ValueError as e:
        print(f"Initialization failed: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during execution: {e}")

if __name__ == '__main__':
    asyncio.run(main_async())