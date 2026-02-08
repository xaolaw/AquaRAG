import json
import os
import sys

sys.path.append(r"D:\Studia\AquaRAG\backend")
from main import agent

with open("./data/questions.json", "r", encoding="utf-8") as f:
    data = json.load(f)["questions"]
f.close()

config = {"configurable": {"user_id": "1", "thread_id": "1"}}

for item in data:
    print(round(int(item["id"]) * 100 / len(data), 2), "% done")
    if item["id"] > 0:
        input_data = {
            "messages": item["question"],
            "mode": "multi_cross_parent_mistral",
        }
        agent.invoke(input=input_data, config=config)

        with open("./output/tmp.json", "r", encoding="utf-8") as f:
            llm_response = json.load(f)
        f.close()
        llm_response["id"] = item["id"]
        llm_response["reference"] = item["answer"]
        with open(f"./output/{item['id']}.json", "w", encoding="utf-8") as output_file:
            json.dump(llm_response, output_file, indent=4, ensure_ascii=False)

        output_file.close()

        open("./output/tmp.json", "w").close()
