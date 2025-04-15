import json

data = []
with open("results/rtlcoder_temp0.2_evalmachine.json", "r") as f:
    data.append(json.loads(f))
    
print(data)