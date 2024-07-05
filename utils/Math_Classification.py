import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer, AutoModelForSequenceClassification


def tokenize_function(example):
    start_prompt = """
    You are a professional teacher adhering to the Common Core standards, teaching Mathematics to students from Grade 1 to Grade 6. 
    Your task is to identify the minimum grade level required to answer the given question.
    
    Question:
    """
    end_prompt = '\n\nGrade classification: '
    prompt = [start_prompt + question + end_prompt for question in example["Question"]]
    example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
    example['attention_mask'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").attention_mask
    return example