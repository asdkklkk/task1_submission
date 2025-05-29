import os

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification



# 加载模型和分词器
model_name = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

print("模型加载成功")


