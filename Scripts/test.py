import json
import pandas as pd

jsonl_file_path = r'D:\ML_Projects\AI_Tech_ChatBot\Data\ChatGPT_chatlogs\ChatGPT_chatlogs.jsonl'

data = []
with open(jsonl_file_path, 'r') as file:
    for line in file:
        data.append(json.loads(line))

df = pd.DataFrame(data)

excel_file_path = r'D:\ML_Projects\AI_Tech_ChatBot\Data\ChatGPT_chatlogs\ChatGPT_chatlogs.xlsx'
df.to_excel(excel_file_path, index=False)

print(f"Data successfully saved to {excel_file_path}")
