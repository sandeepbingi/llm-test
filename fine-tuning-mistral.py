import torch
import json
import os
from datasets import Dataset as HFDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from transformers import DataCollatorForSeq2Seq
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
from transformers import EarlyStoppingCallback
import deepspeed

# Initialize Accelerator
accelerator = Accelerator()

# Initialize distributed processing
torch.distributed.init_process_group(backend='nccl')

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", torch_dtype=torch.float16)

# Move model to device and prepare with accelerator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model = accelerator.prepare(model)

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_length=512):
        with open(json_file, "r") as f:
            self.data = [json.loads(line) for line in f.readlines()]
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        prompt = f"Instruction: {example['instruction']}\nInput: {example['input']}\nOutput: {example['output']}"
        tokenized = self.tokenizer(prompt, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        return {key: val.squeeze() for key, val in tokenized.items()}

# Load dataset
dataset = CustomDataset("dataset.json", tokenizer)
train_size = int(0.9 * len(dataset))
eval_size = len(dataset) - train_size
train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])

# DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=2)

# Prepare data loaders with accelerator
train_dataloader, eval_dataloader = accelerator.prepare(train_dataloader, eval_dataloader)

# LoRA configuration
lora_config = LoraConfig(
    r=16, 
    lora_alpha=32, 
    lora_dropout=0.05, 
    target_modules=["q_proj", "v_proj"],
    bias="none"
)
model = get_peft_model(model, lora_config)

# Training setup
training_args = TrainingArguments(
    output_dir="./mistral_finetuned",
    evaluation_strategy="steps",
    save_strategy="steps",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    fp16=True,
    deepspeed="zero3.json",  # DeepSpeed config
    save_total_limit=2,
    ddp_find_unused_parameters=False,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    report_to="none",  # Disable reporting to external services
)

# Data Collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# Fine-tune model
trainer.train()

# Save the fine-tuned model
trainer.save_model("./mistral_finetuned")
tokenizer.save_pretrained("./mistral_finetuned")
