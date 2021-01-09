import datasets
from datasets import load_dataset

xsum = load_dataset("xsum")

print(xsum)
# train: 204,045 validation: 11,332  test: 11,334
# three features [document, summary, id]

print(sum(len(article) for article in xsum['train']['document']) / len(xsum['train']['document']))
 # 2,202.11
 print(sum(len(article) for article in xsum['train']['summary']) / len(xsum['train']['summary']))
 # 125.46
 print(sum(len(summary) for summary in xsum['validation']['document']) / len(xsum['validation']['document']))
 # 2,173.75
 print(sum(len(summary) for summary in xsum['validation']['summary']) / len(xsum['validation']['summary']))
# 125.58

