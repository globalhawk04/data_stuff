from transformers import BertTokenizer, BertForMaskedLM
import torch
import pandas as pd 
import csv
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
import math
from transformers import AutoModelForMaskedLM
from tensorflow.python.ops.numpy_ops import np_config

#dataset = load_dataset("csv", data_files={"train": ["geo_bert_train.csv"], "test": "geo_bert_test.csv",column_names=['sentence','label']})


dataset = load_dataset('csv', data_files={'train': 'mit_train.csv','test':'mit_test.csv'},column_names=['sentence'])

#print(dataset['train'][10])


tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")



def preprocess_function(examples):
    return tokenizer([" ".join(x) for x in examples["sentence"]])



tokenized_data = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=dataset["train"].column_names,

)


block_size = 25

def group_texts(examples):

    # Concatenate all texts.

    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}

    total_length = len(concatenated_examples[list(examples.keys())[0]])

    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can

    # customize this part to your needs.

    if total_length >= block_size:

        total_length = (total_length // block_size) * block_size

    # Split by chunks of block_size.

    result = {

        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]

        for k, t in concatenated_examples.items()

    }

    result["labels"] = result["input_ids"].copy()

    return result


lm_dataset = tokenized_data.map(group_texts, batched=True, num_proc=4)

tokenizer.pad_token = tokenizer.eos_token



model = AutoModelForMaskedLM.from_pretrained("distilroberta-base")


training_args = TrainingArguments(
    output_dir="mit_masked_model",

    evaluation_strategy="epoch",

    learning_rate=2e-5,
    
    num_train_epochs=10,

    weight_decay=0.01,

    push_to_hub=False,         
)


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=lm_dataset["train"],

    eval_dataset=lm_dataset["test"],

    data_collator=data_collator,

)

trainer.train()



eval_results = trainer.evaluate()

print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

try:
    trainer.save()


except Exception as e:
    print(e)
    print('save model did not work')
     
     
     


dataset = load_dataset('csv', data_files={'train': 'cal-tech_train.csv','test':'cal-tech_test.csv'},column_names=['sentence'])

#print(dataset['train'][10])


tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")



def preprocess_function(examples):
    return tokenizer([" ".join(x) for x in examples["sentence"]])



tokenized_data = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=dataset["train"].column_names,

)


block_size = 25

def group_texts(examples):

    # Concatenate all texts.

    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}

    total_length = len(concatenated_examples[list(examples.keys())[0]])

    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can

    # customize this part to your needs.

    if total_length >= block_size:

        total_length = (total_length // block_size) * block_size

    # Split by chunks of block_size.

    result = {

        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]

        for k, t in concatenated_examples.items()

    }

    result["labels"] = result["input_ids"].copy()

    return result


lm_dataset = tokenized_data.map(group_texts, batched=True, num_proc=4)

tokenizer.pad_token = tokenizer.eos_token



model = AutoModelForMaskedLM.from_pretrained("distilroberta-base")


training_args = TrainingArguments(
    output_dir="cal-tech_masked_model",

    evaluation_strategy="epoch",

    learning_rate=2e-5,

    num_train_epochs=10,
    
    weight_decay=0.01,

    push_to_hub=False,         
)


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=lm_dataset["train"],

    eval_dataset=lm_dataset["test"],

    data_collator=data_collator,

)

trainer.train()



eval_results = trainer.evaluate()

print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

try:
    trainer.save()


except Exception as e:
    print(e)
    print('save model did not work')
    
    


dataset = load_dataset('csv', data_files={'train': 'artsci_train.csv','test':'artsci_test.csv'},column_names=['sentence'])

#print(dataset['train'][10])


tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")



def preprocess_function(examples):
    return tokenizer([" ".join(x) for x in examples["sentence"]])



tokenized_data = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=dataset["train"].column_names,

)


block_size = 25

def group_texts(examples):

    # Concatenate all texts.

    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}

    total_length = len(concatenated_examples[list(examples.keys())[0]])

    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can

    # customize this part to your needs.

    if total_length >= block_size:

        total_length = (total_length // block_size) * block_size

    # Split by chunks of block_size.

    result = {

        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]

        for k, t in concatenated_examples.items()

    }

    result["labels"] = result["input_ids"].copy()

    return result


lm_dataset = tokenized_data.map(group_texts, batched=True, num_proc=4)

tokenizer.pad_token = tokenizer.eos_token



model = AutoModelForMaskedLM.from_pretrained("distilroberta-base")


training_args = TrainingArguments(
    output_dir="artsci_masked_model",

    evaluation_strategy="epoch",

    learning_rate=2e-5,
    
    num_train_epochs=10,
    
    weight_decay=0.01,

    push_to_hub=False,         
)


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=lm_dataset["train"],

    eval_dataset=lm_dataset["test"],

    data_collator=data_collator,

)

trainer.train()



eval_results = trainer.evaluate()

print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

try:
    trainer.save()


except Exception as e:
    print(e)
    print('save model did not work')
