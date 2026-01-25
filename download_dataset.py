import argparse
import json
import ir_datasets
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        choices=["v1", "v2"],
        default="v2",
    )
    parser.add_argument(
        "--output",
        default=None,
    )
    args = parser.parse_args()

    dataset_name = "msmarco-passage-v2" if args.version == "v2" else "msmarco-passage"
    output_path = args.output or f"msmarco_{args.version}_passage_only.jsonl"

    dataset = ir_datasets.load(dataset_name)

    with open(output_path, "w", encoding="utf-8") as f:
        for doc in tqdm(dataset.docs_iter()):
            text = doc.text.replace("\n", " ")
            json.dump({"text": text}, f, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    main()
