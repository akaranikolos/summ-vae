import datasets
from datasets import load_dataset

gigaword = load_dataset("gigaword")

print(len(gigaword))
print(gigaword[0])
print(gigaword.shape)


