RTLCoder Modified by MI1353140938@qq.com

# Prepare
- Firstly you should determine which model do you want to use. When you choose open source model, please read the second point, otherwise, please read the third point.
- For open source model like llama, you should download pretrain model from HuggingFace and place it into certain folder.
- For close source model like ChatGPT or Qwen, you should configurate your environment variable:
    - OPENAI_API_KEY=Your api key
    - OPENAI_BASE_URL=Model base url

# Infer
Read run.sh and change input params to point out:
- "--model" : Model name
- "--n" : The number of generated code for each problem
- "--temperature" : Model output params
- "--gpu_name" : GPU to use
- "--output_dir" : Folder to log generated code
- "--bench_type" : Choose from "Machine" and "Human"
- "--prompt" : LLM will analyse problem firstly and write down code finally iff "True", otherwise, LLM will write down code directly

# Todo

- [x] Support more open source model like Qwen
- [x] Support more close source model like ChatGPT, which demands API call.
- [ ] Perfect README.md
