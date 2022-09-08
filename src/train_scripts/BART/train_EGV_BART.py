import pandas as pd
from datasets import load_metric, Dataset
from transformers import(
    BartForConditionalGeneration, 
    BartTokenizer, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)
import torch, glob, gc, time, os

model_name = 'facebook/bart-base'
batch_size = 2
epochs = 100

metric = load_metric('rouge')

def get_rouge_metrics(preds, refs):
    """Computes the rouge metrics.
    Args:
        preds: list. The model predictions.
        refs: list. The references.
    Returns:
        The rouge metrics.
    """
    rouge_outputs = metric.compute(predictions=preds, references=refs, use_stemmer=True) # stemmer extract the base form of the words
    rouge1 = rouge_outputs["rouge1"].mid
    rouge2 = rouge_outputs["rouge2"].mid
    rougeLsum = rouge_outputs["rougeLsum"].mid

    metrics = {
        "rouge1_precision": round(rouge1.precision, 4)*100,
        "rouge1_recall": round(rouge1.recall, 4)*100,
        "rouge1_fmeasure": round(rouge1.fmeasure, 4)*100,
        "rouge2_precision": round(rouge2.precision, 4)*100,
        "rouge2_recall": round(rouge2.recall, 4)*100,
        "rouge2_fmeasure": round(rouge2.fmeasure, 4)*100,
        "rougeLsum_precision": round(rougeLsum.precision, 4)*100,
        "rougeLsum_recall": round(rougeLsum.recall, 4)*100,
        "rougeLsum_fmeasure": round(rougeLsum.fmeasure, 4)*100,
    }
    print(metrics)
    return metrics

def compute_metrics(pred):
    """Computes the rouge metrics for the evaluation phase.
    Args:
        pred: The model prediction.
    Returns:
        The rouge metrics.
    """
    preds_ids = pred.predictions
    preds_str = tokenizer.batch_decode(preds_ids, skip_special_tokens=True)
    labels_ids = pred.label_ids
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    labels_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    return get_rouge_metrics(preds=preds_str, refs=labels_str)

tokenizer = BartTokenizer.from_pretrained(model_name)
# Define the maximum length for both sources and summaries.
max_len_source = 1024
max_len_summary = 256

def preprocess_function(batch):
    """ Prepare inputs model """
    # tokenize the input text 
    inputs = tokenizer(batch["event"], padding="max_length", max_length=max_len_source, truncation=True)
    # tokenize the target summary
    outputs = tokenizer(batch["event_mention"], padding="max_length", max_length=max_len_summary, truncation=True)
    # adds to the dataset some colums for parameters of the model
    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["labels"] = outputs.input_ids
    # excludes tokens that are padding
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in batch["labels"]]

    return batch

# Load datasets
train_dataset = Dataset.from_pandas(pd.read_csv('../../../data/datasets/train.tsv', sep='\t'))
val_dataset = Dataset.from_pandas(pd.read_csv('../../../data/datasets/validation.tsv', sep='\t'))
test_dataset = Dataset.from_pandas(pd.read_csv('../../../data/datasets/test.tsv', sep='\t'))

train_dataset_model_input = train_dataset.map(
    preprocess_function, 
    batched=True, 
    batch_size=batch_size,
    remove_columns=["event", "event_mention"]
)

train_dataset_model_input.set_format(
 type="torch",
 columns=["input_ids", "attention_mask", "labels"]
)

val_dataset_model_input = val_dataset.map(
    preprocess_function, 
    batched=True, 
    batch_size=batch_size,
    remove_columns=["event", "event_mention"]
)

val_dataset_model_input.set_format(
 type="torch",
 columns=["input_ids", "attention_mask", "labels"]
)

# Utility functions
def load_model():
  checkpoints = glob.glob('../../../data/model_data/EGV/BART/model_checkpoints/checkpoint-*')
  if len(checkpoints) > 0:
    model = BartForConditionalGeneration.from_pretrained("./" + checkpoints[0])
    print("Loaded checkpoint from " + checkpoints[0])
  else:
    model = BartForConditionalGeneration.from_pretrained(model_name)
    print("Loaded HuggingFace model: " + model_name)
  return model

model = load_model().to("cuda:1" if torch.cuda.is_available() else "cpu")

args = Seq2SeqTrainingArguments(
    output_dir = "../../../data/model_data/EGV/BART/model_checkpoints/", # where the model predictions and checkpoints will be written
    evaluation_strategy = "epoch",  # to adopt during training
    save_strategy = "steps",
    per_device_train_batch_size = batch_size*4,
    do_train = True,
    do_eval = True,
    per_device_eval_batch_size = batch_size*4,
    weight_decay = 0.01,
    num_train_epochs = epochs,
    logging_steps = 3866, # 3866 means every epoch
    save_steps = 3866,
    eval_steps = 3866,
    warmup_steps = 1000,
    gradient_accumulation_steps = batch_size,
    eval_accumulation_steps = batch_size,
    predict_with_generate=True,
    overwrite_output_dir=True,
    save_total_limit=1,
    fp16=True # risparmia spazio dimezzando la precisione dei calcoli
)

trainer = None
train_dataset = None
val_datataset = None
test_dataset = None
torch.cuda.empty_cache()
gc.collect()

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=train_dataset_model_input,
    eval_dataset=val_dataset_model_input,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
trainer.train()
print("Training finished")