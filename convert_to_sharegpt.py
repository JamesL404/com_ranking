import argparse
import json
import random


PROMPT_TEMPLATE = (
    "You are a relevance classifier. Your task is to determine if the following Passage is relevant to the given Query.\n\n"
    "Relevance is defined as: The passage provides an answer to the query, or provides useful context that helps answer the query.\n\n"
    "Input Data:\n"
    "Query: {{QUERY}}\n"
    "Passage: {{PASSAGE}}\n\n"
    "Instructions:\n"
    "- If the passage is relevant, output: 1\n"
    "- If the passage is NOT relevant, output: 0\n"
    "- Output ONLY the integer (0 or 1). Do not provide explanations or extra text."
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--shuffle_buffer", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    def to_sharegpt(line):
        obj = json.loads(line)
        query = obj.get("query", "")
        passage = obj.get("passage", "")
        label = obj.get("label", "")
        prompt = PROMPT_TEMPLATE.replace("{{QUERY}}", str(query)).replace("{{PASSAGE}}", str(passage))
        sharegpt = {
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": str(label)},
            ]
        }
        return json.dumps(sharegpt, ensure_ascii=False)

    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        if not args.shuffle:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                fout.write(to_sharegpt(line))
                fout.write("\n")
            return

        buffer = []
        for line in fin:
            line = line.strip()
            if not line:
                continue
            item = to_sharegpt(line)
            if len(buffer) < args.shuffle_buffer:
                buffer.append(item)
                continue
            idx = rng.randrange(len(buffer))
            fout.write(buffer[idx])
            fout.write("\n")
            buffer[idx] = item

        rng.shuffle(buffer)
        for item in buffer:
            fout.write(item)
            fout.write("\n")


if __name__ == "__main__":
    main()
