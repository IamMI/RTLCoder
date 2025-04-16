RTLCoder Modified by MI1353140938@qq.com

# Prepare
- Firstly you should determine which model do you want to use. When you choose open source model, please read the second point, otherwise, please read the third point.
- For open source model like llama, you should download pretrain model from HuggingFace and place it into certain folder, change run.sh to point out your model_path using "--model_path". 
- For close source model like ChatGPT or Qwen, you should configurate your environment variable "DASHSCOPE_API_KEY" to your api_key.

# Infer
Read run.sh and change input params to point out:
- "--model" : Pretrain model name (like "Qwen2.5-Coder-14B")
- "--model_path" : Pretrain model path
- "--n" : The number of generated code for each problem
- "--temperature" : Model output params
- "--gpu_name" : GPU to use
- "--output_dir" : Folder to log generated code
- "--bench_type" : Choose from "Machine" and "Human"

# Todo
- [] Support more open source model like Qwen
- [x] Support more close source model like ChatGPT, which demands API call.
- [] Perfect README.md
