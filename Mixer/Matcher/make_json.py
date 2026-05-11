import csv
import json

csv_path = "FBC.csv"
json_path = "FBC.json"

result = {}

with open(csv_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        fg = row["Foreground"].strip()
        bg1 = row["Background_1"].strip()
        result[fg] = bg1

with open(json_path, "w", encoding='utf-8') as jsonfile:
    json.dump(result, jsonfile, indent=4)

print(f"Saved mapping to {json_path}")