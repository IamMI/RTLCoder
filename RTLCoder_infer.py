import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# Prompt
prompt = "Please act as a professional verilog designer and provide a half adder. \nmodule half_adder\n(input a, \ninput b, \noutput sum, \n output carry);\n"

# Load model and tokenizer
# With multiple gpus, you can specify the GPU you want to use as gpu_name (e.g. int(0)).
gpu_name = 0
# tokenizer = AutoTokenizer.from_pretrained("./pretrain_model/")
# model = AutoModelForCausalLM.from_pretrained("./pretrain_model", torch_dtype=torch.float16, device_map=gpu_name)
# model.eval()
# # Sample
# input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(gpu_name)
# sample = model.generate(input_ids, max_length=512, temperature=0.5, top_p=0.9, do_sample=True)
# s_full = tokenizer.decode(sample[0])
# The RTLCoder-Deepseek-v1.1 may not stop even when the required output text is finished.
# We need to extract the required part from the output sequence based on a keyword "endmodulemodule".

# if len(s_full.split('endmodulemodule', 1)) == 2:
#     s = s_full.split('endmodulemodule', 1)[0] + "\n" + "endmodule"
# else:
#     s = s_full.rsplit('endmodule', 1)[0] + "\n" + "endmodule"
# if s.find('top_module') != -1:
#     s = s.split('top_module', 1)[0]
#     s = s.rsplit('endmodule', 1)[0] + "\n" + "endmodule"
# index = s.rfind('tb_module')
# if index == -1:
#     index = s.find('testbench')
# if index != -1:
#     s_tmp = s[:index]
#     s = s_tmp.rsplit("endmodule", 1)[0] + "\n" + "endmodule"
# print(s)

#For "ishorn5/RTLCoder-v1.1", it will stop generating tokens after completing the coding task.
#But you can still use the keyword "endmodule" to extract the code part.
# tokenizer = AutoTokenizer.from_pretrained("ishorn5/RTLCoder-v1.1")
# model = AutoModelForCausalLM.from_pretrained("ishorn5/RTLCoder-v1.1", torch_dtype=torch.float16, device_map=gpu_name)
# model.eval()
# # Sample
# input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(gpu_name)
# sample = model.generate(input_ids, max_length=512, temperature=0.5, top_p=0.9)
# print(tokenizer.decode(sample[0]))