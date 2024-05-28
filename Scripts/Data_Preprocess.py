import numpy as np
import pandas as pd
import joblib

#df = pd.read_csv(r'D:\ML_Projects\AI_Tech_ChatBot\Data\UbuntuDialogueText\UbuntuDialogueText.csv')

import json
jsonl_file_path = r'D:\ML_Projects\AI_Tech_ChatBot\Data\ChatGPT_chatlogs\ChatGPT_chatlogs.jsonl'

data = []
with open(jsonl_file_path, 'r') as file:
    for line in file:
        data.append(json.loads(line))

#df = pd.DataFrame(data)
#csv_file_path = r'D:\ML_Projects\AI_Tech_ChatBot\Data\ChatGPT_chatlogs\ChatGPT_chatlogs.csv'
#df.to_csv(csv_file_path, index=False)

df = pd.DataFrame(data, columns=['Values'])
df.to_excel(r'D:\ML_Projects\AI_Tech_ChatBot\Data\ChatGPT_chatlogs\ChatGPT_chatlogs.xlsx', index=False)

#joblib.dump(data, r'D:\ML_Projects\AI_Tech_ChatBot\Data\ChatGPT_chatlogs\ChatGPT_chatlogs.pkl')

print(f"Data successfully saved")

