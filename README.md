RTLCoder Modified by MI1353140938@qq.com

RTLCoder Modified by MI1353140938@qq.com.
# Prepare
- You should download pretrain model "ishorn5/RTLCoder-Deepseek-v1.1" from HuggingFace into "pretrain_model" folder defaultly, or anywhere you want, but please change run.sh to point out your model_path.
- You should clone VerilogEval to local, switch the branch to "release/1.0.0", configurate following its hints.

# Infer
Read run.sh and change input params to point out:
- "--model" : Pretrain model params path (like "pretrain_model/")
- "--n" : The number of generated code for each problem
- "--temperature" : Model output params
- "--gpu_name" : GPU to use
- "--output_dir" : Folder to log generated code
- "--output_file" : File to log generated code, always with "--bench_type" suffix
- "--bench_type" : Choose from "Machine" and "Human"

## Run default run.sh
When you do not change the default run.sh and run it. The output file will appear at "results/rtlcoder_temp0.2_evalHuman.json",which contains
- "task_id" : Task id for each problem
- "description" : Problem description
- "prompt" : Generated code


# Todo
- Support more open source model like Qwen
- Support more close source model like ChatGPT, which demands API call.
