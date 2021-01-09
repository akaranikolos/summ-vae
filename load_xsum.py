import datasets
from datasets import load_dataset

xsum = load_dataset("xsum")

print(len(xsum))
print(xsum[0])
print(xsum.shape)


