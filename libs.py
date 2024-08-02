import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import seaborn as sns

from datasets import load_dataset, Dataset
from datasets import DatasetDict
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import TrainerCallback


import evaluate
import os
from tqdm import tqdm
from tabulate import tabulate

import datetime
import pyperclip

accuracy = evaluate.load("accuracy")


id2label = {0: '3', 1: '4', 2: '5', 3: '6'}
label2id = {'3': 0, '4': 1, '5': 2, '6': 3}


import numpy as np


# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     predictions = np.argmax(predictions, axis=1)
#     return accuracy.compute(predictions=predictions, references=labels) 

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    # # Debugging lines to understand the structure
    # print(f"Predictions type: {type(predictions)}")
    # print(f"Predictions content: {predictions}")
    
    # print(f"Labels type: {type(labels)}")
    # print(f"Labels content: {labels}")
    if isinstance(predictions, tuple):
        predictions = predictions[0]  # Assuming the first element contains the logits
    

    # Ensure predictions is a numpy array
    predictions = np.array(predictions)
    
    try:
        predictions = np.argmax(predictions, axis=1)
    except ValueError as e:
        print(f"ValueError: {e}")
        print("Predictions array has inconsistent shapes. Debugging...")
        print(predictions)
        raise e

    accuracy = (predictions == labels).mean()
    return {'accuracy': accuracy}

def compute_top_k_accuracy(preds, labels, k=1):
    
    if isinstance(preds, tuple):
        preds = preds[0]
        
    top_k_preds = np.argsort(preds, axis=1)[:, -k:]
    top_k_accuracy = np.any(top_k_preds == np.expand_dims(labels, axis=1), axis=1).mean()
    return top_k_accuracy

def preprocess_function(examples, tokenizer):
    return tokenizer(examples["Question"], truncation=True, padding = 'max_length', max_length=512)

# class LoggingCallback(TrainerCallback):
#     def __init__(self):
#         self.train_acc = []
#         self.eval_acc_asdiv = []
#         self.eval_acc_mcas = []
        
#     def on_log(self, args, state, control, logs=None, **kwargs):
#         if logs is not None:
#             print(f"Logs: {logs}")  # Print the logs to see what they contain