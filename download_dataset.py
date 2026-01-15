import json
import ir_datasets
from tqdm import tqdm

dataset = ir_datasets.load("msmarco-passage-v2")

with open("msmarco_v2_passage_only.jsonl", "w", encoding="utf-8") as f:
    for doc in tqdm(dataset.docs_iter()):
        text = doc.text.replace("\n", " ")
        json.dump({"text": text}, f, ensure_ascii=False)
        f.write("\n")
