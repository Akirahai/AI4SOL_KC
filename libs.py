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


import evaluate
import argparse
import os
from tqdm import tqdm
from tabulate import tabulate

import datetime

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
    
    # Debugging lines to understand the structure
    print(f"Predictions type: {type(predictions)}")
    print(f"Predictions content: {predictions}")
    
    print(f"Labels type: {type(labels)}")
    print(f"Labels content: {labels}")
    if isinstance(predictions, tuple):
        predictions = predictions[0]  # Assuming the first element contains the logits
    
    print(f"Extracted Predictions shape: {predictions.shape}")  # Debugging line
    print(f"Labels shape: {labels.shape}")  # Debugging line

    # Ensure predictions is a numpy array
    predictions = np.array(predictions)
    
    # Check if the predictions array is consistent in shape
    try:
        # Assuming your predictions are logits or probabilities
        predictions = np.argmax(predictions, axis=1)
    except ValueError as e:
        print(f"ValueError: {e}")
        print("Predictions array has inconsistent shapes. Debugging...")
        print(predictions)
        raise e

    accuracy = (predictions == labels).mean()
    return {'accuracy': accuracy}
