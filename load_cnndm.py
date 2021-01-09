import datasets
from datasets import load_dataset

cnndm = load_dataset("cnn_dailymail", "3.0.0")

print(cnndm)