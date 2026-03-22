import pandas as pd

df = pd.read_excel("tagalog-sms.xlsx")
df.to_csv("tagalog-sms.csv", index=False)