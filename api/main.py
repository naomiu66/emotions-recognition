from transformers import BertForSequenceClassification, AutoTokenizer
import torch
import os

root_path = os.getcwd()

model = BertForSequenceClassification.from_pretrained(f'{root_path}/models/model')
tokenizer = AutoTokenizer.from_pretrained(f'{root_path}/models/tokenizer')