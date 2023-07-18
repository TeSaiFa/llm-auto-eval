from llm_inference import get_prompt_by_typ
import json
data = json.load(open('../dataset/test_samples.json'))
for d in data:
    prompt = get_prompt_by_typ(d)
    print(prompt)