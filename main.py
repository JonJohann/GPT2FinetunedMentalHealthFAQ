import re
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments,AutoModelWithLMHead, pipeline

# Function for cleaning up the data into a  single-lined text file
def clean_up_data(df, file_path):
    f = open(file_path, 'w')
    df['Combined'] = "Question: " + df['Questions'] + " Answer: " + df['Answers']
    current = ''
    for texts in df['Combined']:
        combined = str(texts).strip()
        combined = re.sub(r"\s", " ", combined)
        combined = re.sub(r"  ", " ", combined)
        combined = re.sub(r"   ", " ", combined)
        current += combined + "  "
    f.write(current)

mental_health_data = pd.read_csv('./mental.csv')
train_path = 'train_dataset.txt'
test_path = 'test_dataset.txt'

train, test = train_test_split(mental_health_data,test_size=0.15) 
clean_up_data(train,train_path)
clean_up_data(test,test_path)

# Fetching Huggingface's tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
def load_dataset(train_path,test_path,tokenizer):
    train_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=train_path,
          block_size=128)
     
    test_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=test_path,
          block_size=128)   
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return train_dataset,test_dataset,data_collator

train_dataset,test_dataset,data_collator = load_dataset(train_path,test_path,tokenizer)
model = AutoModelWithLMHead.from_pretrained("gpt2")
training_args = TrainingArguments(
    output_dir="./gpt2",
    overwrite_output_dir=True, 
    num_train_epochs=20, 
    per_device_train_batch_size=32, 
    per_device_eval_batch_size=64,
    eval_steps = 400,
    save_steps=800,
    warmup_steps=500,
    prediction_loss_only=True,
    )

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Training and saving the model
trainer.train()
trainer.save_model()

# Creating models for generating text on both the original GPT-2 and our fine-tuned one
generate_finetuned = pipeline('text-generation',model='./gpt2', tokenizer='gpt2',config={'max_length':1200})
generate_original = pipeline('text-generation',model='gpt2', tokenizer='gpt2',config={'max_length':1200})

finetuned = []
original = []

for question in test["Questions"]:
  finetuned.append(generate_finetuned(f"Question: {question} Answer:")[0]['generated_text'].split("Answer:")[1])
  original.append(generate_original(f"Question: {question} Answer:")[0]['generated_text'].split("Answer:")[1])
test["Original"] = original
test["Finetuned"] = finetuned

test[["Questions", "Original", "Finetuned"]].to_csv("results.csv")