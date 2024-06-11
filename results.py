
import json
import pandas as pd

all_results = {}
with open('model_results.json','r') as file: # 12:36 
    try:
        all_results = json.load(file)
    except json.JSONDecodeError as e:
        all_results = {}

df = pd.DataFrame(all_results).T
print(df['test_scores'])
