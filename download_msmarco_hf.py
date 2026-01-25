import argparse
import json
import random

from datasets import load_dataset
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["v1.1", "v2.1"], default="v1.1")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--neg_per_pos", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = load_dataset(
        "microsoft/ms_marco",
        args.version,
        split=args.split,
        streaming=args.streaming,
    )

    rng = random.Random(args.seed)

    with open(args.output, "w", encoding="utf-8") as f:
        count = 0
        for item in tqdm(dataset):
            query = item.get("query", "")
            passages = item.get("passages", {})
            texts = passages.get("passage_text", [])
            labels = passages.get("is_selected", [])
            pos = []
            neg = []
            for passage, label in zip(texts, labels):
                if int(label) == 1:
                    pos.append(passage)
                else:
                    neg.append(passage)

            if not pos:
                continue

            max_neg = args.neg_per_pos * len(pos)
            if len(neg) > max_neg:
                neg = rng.sample(neg, max_neg)

            for passage in pos:
                row = {
                    "query": query,
                    "passage": passage,
                    "label": 1,
                }
                json.dump(row, f, ensure_ascii=False)
                f.write("\n")
                count += 1
                if args.max_rows is not None and count >= args.max_rows:
                    return

            for passage in neg:
                row = {
                    "query": query,
                    "passage": passage,
                    "label": 0,
                }
                json.dump(row, f, ensure_ascii=False)
                f.write("\n")
                count += 1
                if args.max_rows is not None and count >= args.max_rows:
                    return


if __name__ == "__main__":
    main()
