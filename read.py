# import json

# data = []
# with open("results/rtlcoder_temp0.2_evalmachine.json", "r") as f:
#     data.append(json.loads(f))
    
# print(data)

import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_BASE_URL")

completion = openai.ChatCompletion.create(
            model='gpt-4o',
            messages=[
                {'role': 'user', 'content': 'Hello'}],
        )
