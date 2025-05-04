import os
import json
import random
import math
import traceback # For detailed error printing
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)

# --- [ Evaluation Metrics Imports ] ---
from rouge_score import rouge_scorer, scoring # For ROUGE
from bert_score import score as bert_score_metric # For BERTScore
from nltk.tokenize import sent_tokenize # For compression ratio
import nltk # For downloading punkt tokenizer data

# Ensure punkt is downloaded (only downloads if needed)
try:
    nltk.data.find('tokenizers/punkt')
except (nltk.downloader.DownloadError, LookupError): # Added LookupError
    print("NLTK 'punkt' resource not found. Downloading...")
    nltk.download('punkt', quiet=True) # Added quiet=True for cleaner logs

# Environment Configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # Helps with memory fragmentation

# ----- [ Settings ] -----
file_path = 'all_cleaned_texas_leg_data.json'
ratio = 0.2
random_seed = 42
source_text_col = "enrolled_text"
target_text_col = "summary_text"
prefix = "summarize: "
model_name = "t5-small"
# Main directory for hyperparameter search outputs
base_output_dir_prefix = "./hyp_search_t5_small_texas"
max_length_source = 4979
max_length_target = 752 # Target length (Note: Longer targets increase generation memory)
num_epochs = 3 # Number of training epochs per HPO run

# --- [ Device Configuration ] ---
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
n_gpu = torch.cuda.device_count()
print(f"Using device: {device}") # Corrected f-string

# Set random seeds for reproducibility
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

# --- [ Helper Functions ] -----
def load_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        exit()
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        exit()


def create_dataframe(data, source_col, target_col):
    try:
        df = pd.DataFrame(data)
        df = df[[source_col, target_col]]
        # Basic cleaning
        df = df.dropna() # Ensure no NaN values
        df = df[df[source_col].str.len() > 50] # Remove very short sources
        df = df[df[target_col].str.len() > 10] # Remove very short targets
        df = df.reset_index(drop=True) # Reset index after dropping rows
        return df
    except KeyError:
        print(f"Error: Columns '{source_col}' or '{target_col}' not found in the data.")
        exit()
    except Exception as e:
        print(f"Error creating DataFrame: {e}")
        exit()


# --- [ Load and Prepare Data ] ---
print(f"Loading data from: {file_path}") # Corrected f-string
data = load_data(file_path)
df = create_dataframe(data, source_text_col, target_text_col)
print(f"Data loaded. Filtered examples: {len(df)}") # Corrected f-string

# Perform train-test split
train_df, test_df = train_test_split(df, test_size=ratio, random_state=random_seed)
print(f"Split into {len(train_df)} training examples and {len(test_df)} test examples.")

# Create the huggingface dataset (using split data directly)
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# --- [ Tokenizer ] ---
print(f"Loading tokenizer: {model_name}") # Corrected f-string
tokenizer = AutoTokenizer.from_pretrained(model_name)

# --- [ Preprocessing Function ] ---
def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples[source_text_col]]
    model_inputs = tokenizer(inputs, max_length=max_length_source, truncation=True) # Padding applied by collator

    # Setup the tokenizer for targets
    labels = tokenizer(text_target=examples[target_text_col], max_length=max_length_target, truncation=True) # Padding applied by collator

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# --- [ Create Tokenized Datasets ] ---
print("Tokenizing datasets...")
num_proc = max(1, os.cpu_count() // 2) # Use half cores, or at least 1
print(f"Using {num_proc} cores for tokenization") # Corrected f-string

tokenized_train = train_dataset.map(preprocess_function, batched=True, num_proc=num_proc,
                                    remove_columns=train_dataset.column_names) # Remove original columns dynamically
tokenized_test = test_dataset.map(preprocess_function, batched=True, num_proc=num_proc,
                                   remove_columns=test_dataset.column_names) # Remove original columns dynamically

print("Tokenization complete.")

# --- [ Model and Data Collator ] ---
print(f"Loading base model: {model_name}") # Corrected f-string
base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
base_model.to(device) # Move base model to device

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=base_model, padding=True) # Padding=True is essential

# --- [ Evaluation Metrics Functions ] ---
def compute_rouge_metrics(predictions, references):
    """Calculates ROUGE scores."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()
    skipped_count = 0

    for pred, ref in zip(predictions, references):
        pred_clean = pred.strip()
        ref_clean = ref.strip()
        if pred_clean and ref_clean:
            try:
                # Ensure inputs are strings
                aggregator.add_scores(scorer.score(str(ref_clean), str(pred_clean)))
            except Exception as e:
                # Limit printing warnings for frequent errors if needed
                if skipped_count < 5: # Print only first few errors
                     print(f"Warning: ROUGE calculation error. Skipping pair. Error: {e}") # Corrected f-string
                skipped_count += 1
        else:
            skipped_count += 1

    if skipped_count > 0:
        print(f"Warning: Skipped {skipped_count} pairs in ROUGE calculation (empty or error).")

    if not aggregator._scores: # Check if any scores were successfully added
       print("Warning: No valid prediction/reference pairs processed for ROUGE calculation.")
       return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    result = aggregator.aggregate()
    return {
        "rouge1": round(result["rouge1"].mid.fmeasure * 100, 2) if "rouge1" in result else 0.0,
        "rouge2": round(result["rouge2"].mid.fmeasure * 100, 2) if "rouge2" in result else 0.0,
        "rougeL": round(result["rougeL"].mid.fmeasure * 100, 2) if "rougeL" in result else 0.0,
    }

def compute_bertscore(predictions, references):
    """Calculates BERTScore, optimized for lower memory."""
    valid_preds = [str(p).strip() for p, r in zip(predictions, references) if p and str(p).strip() and r and str(r).strip()]
    valid_refs = [str(r).strip() for p, r in zip(predictions, references) if p and str(p).strip() and r and str(r).strip()]

    if not valid_preds or not valid_refs:
        print("Warning: No valid prediction/reference pairs found for BERTScore calculation after filtering.")
        return None

    try:
        # **ADJUSTMENT FOR 8GB GPU:** Reduced BERTScore batch_size
        bertscore_batch_size = 4 # Lower batch size for BERTScore on 8GB GPU
        print(f"Calculating BERTScore with batch size: {bertscore_batch_size}...")
        P, R, F1 = bert_score_metric(
            valid_preds,
            valid_refs,
            lang='en',
            verbose=False, # Set to True for more detailed progress if needed
            device=device,
            batch_size=bertscore_batch_size # Use lower batch size
        )
        print("BERTScore calculation finished.")
        return {
            "bertscore_precision": round(P.mean().item() * 100, 2),
            "bertscore_recall": round(R.mean().item() * 100, 2),
            "bertscore_f1": round(F1.mean().item() * 100, 2),
        }
    except Exception as e:
        print(f"Warning: BERTScore calculation error: {e}. Check VRAM usage. Returning None for BERTScore.") # Corrected f-string
        return None

def compute_compression_ratio(source_texts, predictions):
    """Calculates the average compression ratio (sentence-based)."""
    ratios = []
    skipped_count = 0
    for source, pred in zip(source_texts, predictions):
        source_clean = str(source).strip()
        pred_clean = str(pred).strip()
        if not source_clean or not pred_clean:
            skipped_count += 1
            continue
        try:
            source_sents = sent_tokenize(source_clean)
            pred_sents = sent_tokenize(pred_clean)
            if len(source_sents) > 0:
                 ratio = len(pred_sents) / len(source_sents)
                 ratios.append(ratio)
            else:
                 skipped_count +=1 # Source had 0 sentences after tokenization
        except Exception as e:
             if skipped_count < 5 : # Limit warnings
                print(f"Warning: Error tokenizing sentence for compression ratio. Skipping. Error: {e}") # Corrected f-string
             skipped_count += 1

    if skipped_count > 0:
        print(f"Warning: Skipped {skipped_count} pairs in compression ratio calculation (empty or error).")

    return {"compression_ratio": round(np.mean(ratios) * 100, 2) if ratios else 0.0}

def compute_loss_only(eval_pred):
    """Used only for loss-based checkpointing during training."""
    return {} # Trainer calculates 'eval_loss' internally

def prepare_predictions_and_labels(model, tokenizer, eval_dataset, device, generation_config):
    """Generates predictions and collects labels for detailed evaluation."""
    model.eval()
    predictions = []
    labels = []

    # **ADJUSTMENT FOR 8GB GPU:** Lower eval_batch_size for generation
    eval_batch_size = 2
    print(f"Preparing dataloader for generation with batch size: {eval_batch_size}")

    # Ensure dataset columns are available or use collator correctly
    # Use `remove_columns=False` in map if original columns needed, or handle in collator
    data_collator_eval = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset.with_format("torch"),
        batch_size=eval_batch_size,
        collate_fn=data_collator_eval, # Use separate collator if needed, or ensure main one works
        shuffle=False # No need to shuffle for evaluation
    )

    print("Generating predictions for detailed evaluation...")
    from tqdm.auto import tqdm # Add progress bar
    for batch in tqdm(eval_dataloader, desc="Generating Predictions"):
        # Move batch to device
        batch_on_device = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        label_ids_batch = batch_on_device['labels'] # Get labels for decoding reference text

        with torch.no_grad():
            try:
                generated_ids = model.generate(
                    input_ids=batch_on_device['input_ids'],
                    attention_mask=batch_on_device['attention_mask'],
                    **generation_config
                )
            except RuntimeError as e:
                 if "out of memory" in str(e):
                     print("\nCUDA out of memory during generation! Try reducing eval_batch_size further or max_target_length.")
                     torch.cuda.empty_cache()
                     # Option: Skip this batch or try with batch size 1? For now, re-raise.
                     raise e
                 else:
                     raise e # Re-raise other runtime errors

        # Decode predictions
        decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        predictions.extend(decoded_preds)

        # Decode labels (move to CPU before numpy conversion)
        label_ids_np = label_ids_batch.cpu().numpy()
        label_ids_np = np.where(label_ids_np == -100, tokenizer.pad_token_id, label_ids_np)
        decoded_labels = tokenizer.batch_decode(label_ids_np, skip_special_tokens=True)
        labels.extend(decoded_labels)

        # Optional: Clear cache periodically if memory issues persist
        # if len(predictions) % 100 == 0: # Example: every 100 batches
        #    torch.cuda.empty_cache()

    print(f"Generated {len(predictions)} predictions.")
    return predictions, labels

def calculate_composite_metric(metrics, perplexity):
    """Calculates a composite metric for comparing runs."""
    bertscore_f1 = metrics.get("bertscore_f1", 0.0) if metrics.get("bertscore_f1") is not None else 0.0
    rouge1 = metrics.get("rouge1", 0.0)
    rouge2 = metrics.get("rouge2", 0.0)
    rougeL = metrics.get("rougeL", 0.0)

    # Example weights (adjust based on importance)
    # Giving ROUGE scores more weight here
    composite_score = (
        -perplexity * 0.05 + # Modest penalty for high perplexity
        rouge1 * 0.25 +
        rouge2 * 0.35 + # Slightly more weight on ROUGE-2
        rougeL * 0.25 +
        bertscore_f1 * 0.10 # Weight BERTScore less heavily
    )
    return composite_score

# --- [ Hyperparameter Search Setup ] ---
# Define hyperparameter space
learning_rates = [3e-5]
weight_decays = [0.01, 0.05, .001]
# **ADJUSTMENT FOR 8GB GPU:** Suggest higher GAS values if effective batch size needs increasing
gradient_accumulation_steps_list = [8] # Start with 8 (effective batch size 1*8=8). Consider [8, 16] if memory allows.

best_composite_metric = -float('inf')
best_run_params = None
best_model_path = None
best_metrics_dict = None
all_run_results = []

print("\n--- Starting Hyperparameter Search ---")

# Iterate through hyperparameter combinations
for lr in learning_rates:
    for wd in weight_decays:
        for gas in gradient_accumulation_steps_list:
            # ** FIX: Create unique output directory for this run **
            run_output_dir = f"{base_output_dir_prefix}_lr{lr}_wd{wd}_gas{gas}"
            run_log_dir = f"./tensorboard_logs/run_lr{lr}_wd{wd}_gas{gas}"

            # ** FIX: Corrected f-string formatting **
            print("\n" + "="*50)
            print(f"Starting Training Run: LR={lr}, WD={wd}, GAS={gas}")
            print(f"Output Directory: {run_output_dir}")
            print("="*50)

            # Model re-use: Assuming Trainer handles resets or using fresh state if needed.
            # If not reloading model, ensure no state persists inappropriately between runs.
            current_model = base_model # Using the same initial model structure

            # --- [ Training Arguments (tuned for 8GB GPU) ] ---
            training_args = Seq2SeqTrainingArguments(
                output_dir=run_output_dir,
                eval_strategy="epoch",
                save_strategy="epoch",
                learning_rate=lr,
                per_device_train_batch_size=1, # ** Keep 1 for 8GB GPU **
                # ** ADJUSTMENT FOR 8GB GPU: Lower eval batch size **
                per_device_eval_batch_size=2,
                gradient_accumulation_steps=gas, # ** Crucial for effective batch size **
                gradient_checkpointing=True, # ** Saves significant memory **
                weight_decay=wd,
                save_total_limit=1, # Keep only the best checkpoint
                num_train_epochs=num_epochs,
                # ** Use FP16 only if CUDA available**
                fp16=torch.cuda.is_available(),
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss", # Using loss for checkpoint selection
                greater_is_better=False,
                optim="adafactor", # Memory-efficient optimizer
                seed=random_seed,
                data_seed=random_seed,
                predict_with_generate=True, # Generate text for metrics calculation (affects evaluate)
                generation_max_length=max_length_target, # Affects memory during predict_with_generate steps
                generation_num_beams=4, # Beam search uses more memory
                logging_steps=max(10, len(train_dataset) // (gas * n_gpu * 10) if len(train_dataset) > 0 and gas > 0 else 50), # Log ~10 times per epoch
                logging_dir=run_log_dir,
                report_to="tensorboard",
                # Helps prevent some OOM errors when generation happens during eval_strategy="epoch"
                # evaluation_accumulation_steps=2 # Accumulate grads during eval if needed (use if eval OOMs occur) - requires more recent transformers version
            )

            # Generation config dict for detailed evaluation function
            generation_config = {
                "max_length": max_length_target,
                "num_beams": 4,
                "early_stopping": True,
                # "length_penalty": 0.8, # Example: encourage shorter summaries
                # "no_repeat_ngram_size": 3 # Example: prevent phrase repetition
            }


            # --- [ Trainer ] ---
            trainer = Seq2SeqTrainer(
                model=current_model, # Pass the model instance for this run
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_test,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_loss_only, # Using loss only during Trainer's eval steps
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.01)]
            )

            # --- [ Train ] ---
            print(f"Starting training for run LR={lr}, WD={wd}, GAS={gas}...")
            try:
                train_result = trainer.train()
                trainer.save_model() # Saves the best model according to load_best_model_at_end
                print("Training finished successfully.")
            except RuntimeError as e:
                 if "out of memory" in str(e):
                     print("\nCUDA out of memory during TRAINING! Try increasing gradient_accumulation_steps,")
                     print("reducing max_target_length, or checking if gradient_checkpointing is active.")
                     torch.cuda.empty_cache()
                     print("Skipping to next hyperparameter combination...")
                     continue # Skip evaluation if training failed
                 else:
                     print(f"An unexpected error occurred during training: {e}")
                     traceback.print_exc()
                     continue # Skip evaluation if training failed
            except Exception as e:
                print(f"An error occurred during training: {e}") # Corrected f-string
                traceback.print_exc()
                continue

            # --- [ Evaluate - Detailed Metrics (Post-Training) ] ---
            print("Starting detailed evaluation on the best checkpoint for this run...")
            try:
                # Ensure model is on the correct device after potential loading
                best_model_for_run = trainer.model.to(device)
                best_tokenizer_for_run = trainer.tokenizer

                # Calculate perplexity on test set using the best model
                print("Calculating perplexity...")
                eval_metrics = trainer.evaluate(eval_dataset=tokenized_test, metric_key_prefix="final_eval")
                perplexity = math.exp(eval_metrics['final_eval_loss'])
                print(f"Perplexity of best checkpoint for this run: {perplexity:.4f}")

                # Generate Predictions using the best model
                predictions, labels = prepare_predictions_and_labels(
                    best_model_for_run,
                    best_tokenizer_for_run,
                    tokenized_test,
                    device,
                    generation_config
                )

                # Compute Detailed Metrics
                print("Calculating detailed metrics (ROUGE, BERTScore, Compression)...")
                rouge_results = compute_rouge_metrics(predictions, labels)
                print(f"ROUGE Results: {rouge_results}") # Corrected f-string

                bertscore_results = compute_bertscore(predictions, references=labels) # Pass references
                if bertscore_results:
                    print(f"BERTScore Results: {bertscore_results}") # Corrected f-string
                else:
                    print("BERTScore calculation skipped or failed.")

                # Get original source texts for compression ratio
                original_source_texts = test_df[source_text_col].tolist()
                compression_results = compute_compression_ratio(original_source_texts, predictions)
                print(f"Compression Ratio Result: {compression_results}") # Corrected f-string

                # Combine all metrics
                detailed_metrics = {**rouge_results, **compression_results}
                if bertscore_results:
                    detailed_metrics.update(bertscore_results)
                detailed_metrics["perplexity"] = round(perplexity, 4) # Add perplexity

                print(f"\nDetailed metrics for run (LR={lr}, WD={wd}, GAS={gas}):") # Corrected f-string
                for key, value in detailed_metrics.items():
                     print(f"  {key}: {value}") # Corrected f-string

                # Calculate Composite Metric
                current_composite_metric = calculate_composite_metric(detailed_metrics, perplexity)
                print(f"Composite Metric for this run: {current_composite_metric:.4f}") # Corrected f-string

                # Store results
                run_summary = {
                    "params": {"lr": lr, "wd": wd, "gas": gas},
                    "metrics": detailed_metrics,
                    "composite_metric": current_composite_metric,
                    "output_dir": run_output_dir
                }
                all_run_results.append(run_summary)

                # Check if this run is the best so far
                if current_composite_metric > best_composite_metric:
                    best_composite_metric = current_composite_metric
                    best_run_params = {"lr": lr, "wd": wd, "gas": gas}
                    best_model_path = run_output_dir # Directory where the best model was saved
                    best_metrics_dict = detailed_metrics
                    print(f"*** New Best Model Found! *** Composite Metric: {best_composite_metric:.4f}") # Corrected f-string

            except RuntimeError as e:
                 if "out of memory" in str(e):
                     print("\nCUDA out of memory during DETAILED EVALUATION! Try reducing eval_batch_size in")
                     print("prepare_predictions_and_labels, max_target_length, or BERTScore batch size.")
                     torch.cuda.empty_cache()
                     print("Skipping detailed metrics for this run.")
                 else:
                     print(f"Evaluation failed for run LR={lr}, WD={wd}, GAS={gas}: {e}") # Corrected f-string
                     traceback.print_exc()
            except Exception as e:
                print(f"Evaluation failed for run LR={lr}, WD={wd}, GAS={gas}: {e}") # Corrected f-string
                traceback.print_exc()

            # Manual cleanup between HPO runs might help memory stability
            del trainer
            del best_model_for_run # Ensure reference is removed
            # del current_model # Careful if current_model points to base_model
            if 'predictions' in locals(): del predictions # Clear large lists
            if 'labels' in locals(): del labels
            if 'original_source_texts' in locals(): del original_source_texts
            torch.cuda.empty_cache() # Clear GPU cache


print("\n--- Hyperparameter Search Complete ---")

# --- [ Final Results ] ---
if best_run_params:
    print("\n" + "="*50)
    print("Overall Best Performing Model Parameters:")
    print(f"  Learning Rate: {best_run_params['lr']}") # Corrected f-string
    print(f"  Weight Decay: {best_run_params['wd']}") # Corrected f-string
    print(f"  Gradient Accumulation Steps: {best_run_params['gas']}") # Corrected f-string
    print(f"  Best Composite Metric: {best_composite_metric:.4f}") # Corrected f-string
    print(f"  Best Model saved in directory: {best_model_path}") # Corrected f-string
    print("\n  Best Model Metrics:")
    if best_metrics_dict:
        for key, value in best_metrics_dict.items():
             print(f"    {key}: {value}") # Corrected f-string
    print("="*50)

    # Save all run results to a JSON file
    try:
        results_file = "hyperparameter_search_results.json"
        # Convert numpy types to standard types for JSON serialization
        serializable_results = []
        for run in all_run_results:
            serializable_metrics = {k: (float(v) if isinstance(v, (np.float32, np.float64)) else v) for k, v in run['metrics'].items()}
            run['metrics'] = serializable_metrics
            if isinstance(run['composite_metric'], (np.float32, np.float64)):
                 run['composite_metric'] = float(run['composite_metric'])
            serializable_results.append(run)

        with open(results_file, 'w', encoding='utf-8') as f: # Ensure encoding
             json.dump(serializable_results, f, indent=4)
        print(f"Saved detailed results for all runs to: {results_file}") # Corrected f-string
    except Exception as e:
        print(f"Error saving all run results: {e}") # Corrected f-string

else:
    print("Hyperparameter search completed, but no successful runs were recorded or no best model found.")

print("\nScript finished.")
