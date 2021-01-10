import datasets
from datasets import load_dataset

arxiv = load_dataset("scientific_papers", 'arxiv')
arxiv.save_to_disk("dataset/arxiv")

print(arxiv)

# train: 203,037 validation: 6,436  test: 6,640
# three features [article, abstract, section_names]
#print(arxiv['train']['article'][0])
# len: 26,092
#print(arxiv['train']['abstract'][0])
# len: 932

