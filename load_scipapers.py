import datasets
from datasets import load_dataset

scipapers = load_dataset("scientific_papers", 'arxiv')

print(scipapers)
# train: 203,037 validation: 6,436  test: 6,640
# three features [article, abstract, section_names]
print(scipapers['train']['article'][0])
# len: 26,092
print(scipapers['train']['summary'][0])
# len: 932

 print(sum(len(article) for article in scipapers['train']['article']) / len(scipapers['train']['article']))
 # 33,431.85
 print(sum(len(article) for article in scipapers['validation']['article']) / len(scipapers['validation']['article']))
 # 32,622.98
 print(sum(len(summary) for summary in scipapers['train']['abstract']) / len(scipapers['train']['abstract']))
 # 1,624.39
 print(sum(len(summary) for summary in scipapers['validation']['abstract']) / len(scipapers['validation']['abstract']))
# 961.71

