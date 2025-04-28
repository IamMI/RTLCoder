import argparse
import json
import os
from tqdm import*
from multiprocessing import Pool
import time
import re
import sys
import torch.multiprocessing as mp
sys.path.append("/deltadisk/huangjiayi/demo/verilog-eval/verilog_eval")
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import openai

from execution import check_correctness

def load_json(filename):
    des_data = []
    with open(filename, 'r') as f:
        for line in f:
            data = json.loads(line)
            des_data.append(data)
    return des_data

def Process(des_data, input_data, model_name, prompt=False):
    # Create client
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),   
    )
    
    try:
        dic_list = []
        for data in des_data:
            # Input
            dic = {}
            dic['task_id'] = data['task_id']
            dic['description'] = data['detail_description']

            for j in range(len(input_data)):
                if input_data[j]['task_id'] == dic['task_id']:                
                    dic['prompt'] = input_data[j]['prompt']
                    break
            prompt = dic['description'] + '\n' + dic['prompt'] + '\n'
            
            if not prompt:
                requirement = "You're a Verilog designer. Based on the input requirements, you will write Verilog code with the following instructions: \
                            1. The input includes a module functionality description followed by the module header. You are to write the rest of the module based on this header. \
                            2. In your response, start directly from the given module header and end with endmodule. Do not include any other explanation or description.\
                            3. In your response, please do not include the given function header. Just write the body of the program directly, and end it with 'endmodule'."
            else:
                requirement = "You are a Verilog programmer. You will be given a series of Verilog-related problems in the following format: \
                            1. First, you will receive a problem description. You need to carefully read and understand the problem. \
                            2. Then, you will be provided with a function header. Your code should start directly from this header and end with endmodule. \
                            Please follow these steps to write the Verilog code: \
                            1. Begin by analyzing the problem and output your thought process. \
                            2. Break the problem down into smaller sub-tasks. \
                            3. Based on your analysis, write the complete Verilog code. \
                            Your Verilog code block should follow the format below: \
                            1. The function header must match the one provided exactly. \
                            2. The code must end with endmodule. "
                                        
            # API call
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {'role': 'system', 'content': requirement},
                    {'role': 'user', 'content': prompt}],
            )
            
            outputs1 = completion.model_dump()['choices'][0]['message']['content']
            
            # post process
            if not prompt:
                if(len(outputs.split("top_module", 1))!=1):
                    outputs = outputs.split(";", 1)[1]
                if(len(outputs.split("```", 1))!=1):
                    outputs = outputs.split("```", 2)[1][7:]
            else:
                outputs2 = re.findall(r'```verilog(.*?)```', outputs1, re.DOTALL)[-1]
                outputs3 = outputs2.split(";", 1)[1]

            dic.update({
                'completion' : outputs3,
            })
            dic_list.append(dic)
    except Exception as e:
        print(outputs1)
        print('='*30)
        print(outputs2)
        print('='*30)
        print(outputs3)
        print(data['task_id'])
        # print(outputs)
        print(f"Error occur: {e}")
        
    return dic_list

def localModel(des_data, input_data, args):
    id = 1
    while id <= len(des_data):
        gen_batch_size = 16
        tmp_list = []
        dic_list = []
        for ite in range(gen_batch_size):
            item = des_data[id - 1]
            dic = {}
            dic['task_id'] = item['task_id']
            dic['description'] = item['detail_description']

            for j in range(len(input_data)):
                if input_data[j]['task_id'] == dic['task_id']:                
                    dic['prompt'] = input_data[j]['prompt']
                    break

            prompt = dic['description'] + '\n' + dic['prompt'] + '\n'
            tmp_list.append(prompt)
            dic_list.append(dic)
            id = id + 1
            if id > len(des_data):
                flag = 1
                break
        
        inputs = tokenizer(tmp_list, return_tensors="pt", padding='longest').to(args.gpu_name)
        outputs = model.generate(inputs=inputs.input_ids, max_length=len(inputs[0]) + 1024, do_sample=True, temperature=args.temperature, top_p=0.95,
        attention_mask=inputs.attention_mask)

        for res_i, output in enumerate(outputs):
            s_full = tokenizer.decode(output[len(inputs[0]):].cpu().squeeze(), skip_special_tokens=True)
            # please note that the RTLCoder-deepseek-v1.1 version requires a different extraction method
            if len(s_full.split('endmodulemodule', 1)) == 2:
                s = s_full.split('endmodulemodule', 1)[0] + "\n" + "endmodule"
            else:
                s = s_full.rsplit('endmodule', 1)[0] + "\n" + "endmodule"
            if s.find('top_module') != -1:
                s = s.split('top_module', 1)[0]
                s = s.rsplit('endmodule', 1)[0] + "\n" + "endmodule"
            index = s.rfind('tb_module')
            if index == -1:
                index = s.find('testbench')
            if index != -1:
                s_tmp = s[:index]
                s = s_tmp.rsplit("endmodule", 1)[0] + "\n" + "endmodule"
            
            # Record
            dic_list[res_i].update({
                "completion": s,
            })
        
        output_file = f"{args.model}_eval{args.bench_type}.json"
        with open(os.path.join(args.output_dir, output_file),'a') as f:
            for dic_item in dic_list:
                ob = json.dumps(dic_item)
                f.write(ob)
                f.write('\n')

def CloudModel(des_data, input_data, args, model_name, num_workers=20):
    chunk_size = (len(des_data) + num_workers - 1) // num_workers
    chunks = [des_data[i:min(i + chunk_size, len(des_data))] for i in range(0, len(des_data), chunk_size)]

    pool = Pool(processes=num_workers)
    results = []

    for chunk in chunks:
        result = pool.apply_async(Process, args=(chunk, input_data, model_name, args.prompt))
        results.append(result)

    print("Begin to call API!")
    start = time.time()
    pool.close()
    pool.join()
    end = time.time()
    print(f"End of calling! Spend {(end-start)/60} min!")
    
    all_results = []
    for r in results:
        all_results.extend(r.get())

    # Save
    output_file = f"{args.model}_eval{args.bench_type}.json"
    with open(os.path.join(args.output_dir, output_file),'a') as f:
        for dic in all_results:
            json_line = json.dumps(dic)
            f.write(json_line + '\n')          

def react_worker(des_data, input_data, args, gpu_name, model, tokenizer, react_correct):
    maxiter = 4
    
    # LLM advice
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_base = os.getenv("OPENAI_BASE_URL")

    requirement = "You are a Verilog inspector. The user will provide a question and a piece \
of erroneous code. Your task is to analyze the question, then review the code based \
on the issue described. Finally, return a suggestion within 100 words, clearly \
identifying the problem and offering modification advice. Note: you must not \
directly output the corrected code — only provide suggestions."
    
    for item in des_data:
        last_history = ""
        iters=0
        # prepare data
        dic = {}
        dic['task_id'] = item['task_id']
        dic['description'] = item['detail_description']
        for j in range(len(input_data)):
            if input_data[j]['task_id'] == dic['task_id']:                
                dic['prompt'] = input_data[j]['prompt']
                dic['test'] = input_data[j]['test']
                break
        prompt = dic['description'] + '\n' + dic['prompt'] + '\n'
        last_history = prompt
        
        # React
        while iters < maxiter:
            # Generate
            inputs = tokenizer(last_history, return_tensors="pt", padding='longest').to(gpu_name)
            outputs = model.generate(inputs=inputs.input_ids, max_length=len(inputs[0]) + 1024, do_sample=True, temperature=args.temperature, top_p=0.95,
            attention_mask=inputs.attention_mask, pad_token_id=tokenizer.pad_token_id )
            outputs = outputs[:, inputs.input_ids.shape[-1]:] 
            
            # Decode
            s_full = tokenizer.decode(outputs.cpu().squeeze(), skip_special_tokens=True)
            s = s_full.split('endmodule')[0]+ '\nendmodule'
            
            # Check correctness
            result = check_correctness(dic, s, 30, item['task_id'])
            
            if result['passed']:
                print(f"Task: {item['task_id']} pass! Use {iters+1} totally!")
                react_correct[iters] += 1
                break
            
            # Obtain suggestion of correcting
            completion = openai.ChatCompletion.create(
                model='gpt-4o',
                messages=[
                    {'role': 'system', 'content': requirement},
                    {'role': 'user', 'content': 'Problem description and wrong code: '+prompt+s}],
            )
            
            advice = completion['choices'][0]['message']['content']

            # Add into history
            last_history = dic['description']
            last_history += "\nNext I will give you a wrong code and advice, please follow advice to correct it carefully.\n"
            last_history += '\n'+dic['prompt'] 
            last_history += s 
            last_history += '\nAdvice: '+advice
            last_history += '\n'+dic['prompt']
            
            iters += 1
           
def parallel_react(des_data, input_data, args):
    start = time.time()
    with mp.Manager() as manager:
        correct = manager.Value('i', 0)  
        react_correct = manager.list([0 for _ in range(4)])

        num_chunks = 4
        des_data_chunks = [des_data[i::num_chunks] for i in range(num_chunks)]
        processes = []
        
        for i in range(num_chunks):
            gpu_name = f'cuda:{i}' 
            
            # Model and tokenizer
            model_path = os.path.join(os.environ['HOME'], "pretrain_model", args.model)
            tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map=gpu_name,)
            model.eval()
            
            
            
            p = mp.Process(target=react_worker, args=(des_data_chunks[i], input_data, args, gpu_name, model, tokenizer, react_correct))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
        
        end = time.time()
        
        with open("react.json", "w") as f:
            f.write(f"Total accuracy: {react_correct[0]+react_correct[1]+react_correct[2]+react_correct[3]}/{len(des_data)}.\n")
            f.write(f"React accuracy: {react_correct[1]+react_correct[2]+react_correct[3]}/{len(des_data)}.\n")
            f.write(f"React iter{0}: {react_correct[0]}.\n")
            f.write(f"React iter{1}: {react_correct[1]}.\n")
            f.write(f"React iter{2}: {react_correct[2]}.\n")
            f.write(f"React iter{3}: {react_correct[3]}.\n")
            f.write(f"Time cost: {(end-start)/3600} h!\n")
        return correct.value, react_correct.value
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some test.')
    parser.add_argument('--model', type=str)
    parser.add_argument('--temperature', type=float)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--bench_type', type=str)
    parser.add_argument('--gpu_name', type=int)
    parser.add_argument('--n', type=int) 
    parser.add_argument('--prompt', type=bool)
    args = parser.parse_args()

    descri_path = '/deltadisk/huangjiayi/demo/verilog-eval/descriptions/VerilogDescription_' + args.bench_type + '.jsonl'
    input_path = '/deltadisk/huangjiayi/demo/verilog-eval/data/VerilogEval_' + args.bench_type + '.jsonl'

    des_data = load_json(descri_path)
    input_data = load_json(input_path)
    tmp_list = des_data
    des_data = []
    for i in range(args.n):
        des_data += tmp_list

    mp.set_start_method('spawn', force=True)

    # load model
    if args.model in ["RTLCoder-DS"]:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
        )
        
        # model_path = os.path.join(os.environ['HOME'], "pretrain_model", args.model)
        # tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        # model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map=args.gpu_name,)
        # model.eval()
        parallel_react(des_data, input_data, args)
        # localModel(des_data, input_data, progress_bar, args)
        
    # elif args.model in ["Qwen2.5-Coder-14B"]:
    #     from openai import OpenAI
        
    #     # Model name
    #     model_name_dic = {
    #         "Qwen2.5-Coder-14B" : "qwen2.5-14b-instruct",
    #     }
    #     model_name = model_name_dic[args.model]
    #     assert model_name, "Model Name is not supported now! Please enter its name on platform!"
        
    #     CloudModel(des_data, input_data, args, client, model_name)

    elif args.model in ["GPT4", "GPT4o", "GPT4o-mini", "Gemini2.0flash", "Qwen2.5-Coder-14B"]:
        from openai import OpenAI
        
        model_name_dic = {
            "GPT4.1" : "gpt-4.1",
            "GPT4o-mini" : "gpt-4o-mini",
            "GPT4o" : "gpt-4o",
            "Gemini2.0flash" : "gemini-2.0-flash",
            "Qwen2.5-Coder-14B" : "qwen2.5-14b-instruct",
        }
        model_name = model_name_dic[args.model]
        assert model_name, "Model Name is not supported now! Please enter its name on platform!"
        
        CloudModel(des_data, input_data, args, model_name)
    else:
        print("Your chose model is not supported!")