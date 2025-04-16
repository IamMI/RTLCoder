import argparse
import json
import os
from tqdm import*


def load_json(filename):
    des_data = []
    with open(filename, 'r') as f:
        for line in f:
            data = json.loads(line)
            des_data.append(data)
    return des_data


def localModel(des_data, input_data, progress_bar, args):
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
            #please note that the RTLCoder-deepseek-v1.1 version requires a different extraction method
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
            
            #If the RTLCoder version is based on Mistral, just use the following extraction method.
            ####
            # s = s_full.rsplit('endmodule', 1)[0] + "\n" + "endmodule"

            # # the model may output testbench after the design code
            # index = s.rfind('tb_module')
            # if index == -1:
            #     index = s.find('testbench')
            # if index != -1:
            #     s_tmp = s[:index]
            #     s = s_tmp.rsplit('endmodule', 1)[0] + "\n" + "endmodule"
            #####
        output_file = f"{args.model}_eval{args.benchtype}.json"
        with open(os.path.join(args.output_dir, output_file),'a') as f:
            for dic_item in dic_list:
                ob = json.dumps(dic_item)
                f.write(ob)
                f.write('\n')
        progress_bar.update(len(dic_list))

def CloudModel(des_data, input_data, args, client, model_name):
    
    for data in tqdm(des_data):
        # Input
        dic = {}
        dic['task_id'] = data['task_id']
        dic['description'] = data['detail_description']

        for j in range(len(input_data)):
            if input_data[j]['task_id'] == dic['task_id']:                
                dic['prompt'] = input_data[j]['prompt']
                break
        prompt = dic['description'] + '\n' + dic['prompt'] + '\n'
        
        requirement = "You're a Verilog designer. Based on the input requirements, you will write Verilog code with the following instructions: \
                       1. The input includes a module functionality description followed by the module header. You are to write the rest of the module based on this header. \
                       2. In your response, start directly from the given module header and end with endmodule. Do not include any other explanation or description."
        
        # API call
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {'role': 'system', 'content': requirement},
                {'role': 'user', 'content': prompt}],
        )
        
        outputs = completion.model_dump()['choices'][0]['message']['content']

        dic.update({
            'completion' : outputs,
        })
        
        output_file = f"{args.model}_eval{args.bench_type}.json"
        with open(os.path.join(args.output_dir, output_file),'a') as f:
            ob = json.dumps(dic)
            f.write(ob)
            f.write('\n')
        




parser = argparse.ArgumentParser(description='Process some test.')
parser.add_argument('--model', type=str)
parser.add_argument('--temperature', type=float)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--bench_type', type=str) # it can be Machine or Human
parser.add_argument('--gpu_name', type=int)
parser.add_argument('--n', type=int) # 'n' represent how many code candidates generated for each instruction
parser.add_argument('--api', type=str)
args = parser.parse_args()


descri_path = '/deltadisk/huangjiayi/demo/verilog-eval/descriptions/VerilogDescription_' + args.bench_type + '.jsonl'
input_path = '/deltadisk/huangjiayi/demo/verilog-eval/data/VerilogEval_' + args.bench_type + '.jsonl'

des_data = load_json(descri_path)
input_data = load_json(input_path)
progress_bar = tqdm(total=len(des_data) * args.n)
tmp_list = des_data
des_data = []
for i in range(args.n):
    des_data += tmp_list

# load model
if args.model in ["Qwen2.5-Coder-7B", "RTLCoder-DS"]:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )
    model_path = os.path.join(os.environ['HOME'], "pretrain_model", args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map=args.gpu_name,)
    model.eval()
    localModel(des_data, input_data, progress_bar, args)
    
elif args.model in ["Qwen2.5-Coder-14B", "GPT3.5", "GPT4", "GPT4o"]:
    from openai import OpenAI
    # API call
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=args.api if args.api else os.getenv("DASHSCOPE_API_KEY"), 
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    # Model name
    model_name_dic = {
        "Qwen2.5-Coder-14B" : "qwen2.5-14b-instruct",
    }
    model_name = model_name_dic[args.model]
    assert model_name, "Model Name is not supported now! Please enter its name on platform!"
    
    CloudModel(des_data, input_data, args, client, model_name)
