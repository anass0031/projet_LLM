# =========================
# 1. IMPORTS
# =========================
from datasets import load_dataset, concatenate_datasets
import json

# =========================
# 2. LOAD DATASETS
# =========================
print("Loading datasets...")

med_dataset = load_dataset(
    "lavita/medical-qa-datasets",
    "pubmed-qa"
)["train"]

drug_dataset = load_dataset(
    "lavita/medical-qa-datasets",
    "medical_meadow_medical_flashcards"
)["train"]


# =========================
# 3. REDUCE SIZE (IMPORTANT)
# =========================
# med_dataset = med_dataset.select(range(10000))
# drug_dataset = drug_dataset.select(range(10000))


# =========================
# 4. FORMAT DATA
# =========================
def format_med(example):
    question = example.get("QUESTION", "")
    contexts = example.get("CONTEXTS", [])
    context_text = " ".join(contexts[:2])
    answer = example.get("LONG_ANSWER", "")

    return {
        "text": f"Question: {question}\nContext: {context_text}\nAnswer: {answer}"
    }


def format_drug(example):
    return {
        "text": f"Instruction: {example.get('instruction','')}\nInput: {example.get('input','')}\nAnswer: {example.get('output','')}"
    }


print("Formatting datasets...")

med_dataset = med_dataset.map(format_med, remove_columns=med_dataset.column_names)
drug_dataset = drug_dataset.map(format_drug, remove_columns=drug_dataset.column_names)


# =========================
# 5. MERGE DATASETS
# =========================
dataset = concatenate_datasets([med_dataset, drug_dataset])


# =========================
# 6. CLEAN DATA
# =========================
dataset = dataset.filter(lambda x: len(x["text"]) > 20)


# =========================
# 7. SHUFFLE
# =========================
dataset = dataset.shuffle(seed=42)


# =========================
# 8. TEST OUTPUT
# =========================
print("Sample:")
print(dataset[0])


# =========================
# 9. SAVE DATASET
# =========================
# dataset.save_to_disk("medical_dataset")

# print("Dataset saved successfully!")

# =========================
# 10. EXPORT TO JSONL
# =========================
output_file = "medical_dataset.jsonl"

print("Saving JSONL...")

with open(output_file, "w", encoding="utf-8") as f:
    for item in dataset:
        json_line = json.dumps(item, ensure_ascii=False)
        f.write(json_line + "\n")

print(f"Saved to {output_file} successfully!")
