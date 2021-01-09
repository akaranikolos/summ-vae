import datasets
from datasets import load_dataset

gigaword = load_dataset("gigaword")

print(gigaword)
# train: 3,803,957  validation:189,651 test: 1,951 
#[document, summary]

print(sum(len(article) for article in gigaword['train']['document']) / len(gigaword['train']['document']))
 # 181.53 chars
print(sum(len(article) for article in gigaword['train']['summary']) / len(gigaword['train']['summary']))
 # 51.06 chars
print(sum(len(summary) for summary in gigaword['validation']['document']) / len(gigaword['validation']['document']))
 # 181.50 chars
print(sum(len(summary) for summary in gigaword['validation']['summary']) / len(gigaword['validation']['summary']))
# 51.81 chars






