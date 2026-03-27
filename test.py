import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_from_disk, load_dataset
import evaluate

model_id = "google/gemma-3-1b-it"
adapter_path = "./gemma_lora_model"
dataset_path = "medical_dataset"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model and test data using {device}...")
tokenizer = AutoTokenizer.from_pretrained(adapter_path)
base_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base_model, adapter_path).to(device)

# dataset = load_from_disk(dataset_path)
dataset = load_dataset(
    "json",
    data_files="dataset/medical_dataset.jsonl"
)["train"]
test_data = dataset.select(range(min(100, len(dataset)))) 

rouge = evaluate.load("rouge")

predictions = []
references = []

print("Starting Evaluation...")

for i, entry in enumerate(test_data):
    prompt = entry['text'].split("Response:")[0] + "Response:" 
    ground_truth = entry['text'].split("Response:")[-1].strip() 
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
    
    gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prediction = gen_text.replace(prompt, "").strip() 
    
    predictions.append(prediction)
    references.append(ground_truth)
    
    if i % 10 == 0:
        print(f"Tested {i}/{len(test_data)} items...")

results = rouge.compute(predictions=predictions, references=references)

precision_score = results['rougeL'] * 100 

print("\n" + "="*30)
print(f"FINAL PRECISION SCORE: {precision_score:.2f}%")
print("="*30)