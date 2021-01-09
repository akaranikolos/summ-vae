from transformers import BertTokenizerFast
from datasets import load_from_disk

cnndm = load_from_disk("dataset/cnndm")
train_data = cnndm['train']
valid_data = cnndm['validation']
test_data = cnndm['test']

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def map_to_length(x):
  x["article_len"] = len(tokenizer(x["src"]).input_ids)
  x["article_longer_512"] = int(x["article_len"] > tokenizer.max_len)
  x["summary_len"] = len(tokenizer(x["trg"]).input_ids)
  x["summary_longer_64"] = int(x["summary_len"] > 64)
  x["summary_longer_128"] = int(x["summary_len"] > 128)
  return x

sample_size = 10000
data_stats = train_data.select(range(sample_size)).map(map_to_length, num_proc=4)

def compute_and_print_stats(x):
  if len(x["article_len"]) == sample_size:
    print(
        "Article Mean: {}, %-Articles > 512:{}, Summary Mean:{}, %-Summary > 64:{}, %-Summary > 128:{}".format(
            sum(x["article_len"]) / sample_size,
            sum(x["article_longer_512"]) / sample_size, 
            sum(x["summary_len"]) / sample_size,
            sum(x["summary_longer_64"]) / sample_size,
            sum(x["summary_longer_128"]) / sample_size,
        )
    )

output = data_stats.map(
  compute_and_print_stats, 
  batched=True,
  batch_size=-1,
)



