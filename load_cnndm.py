import datasets
from datasets import load_dataset

cnndm = load_dataset('json', data_files={'train': 'data/cnn_dm/cnndm_train.json','validation': 'data/cnn_dm/cnndm_valid.json','test': 'data/cnn_dm/cnndm_test.json'})
cnndm.save_to_disk("dataset/cnndm")

train_data = cnndm['train']
#print(len(cnndm))
# train: 287,112 validation: 13,368  test: 11,490
# two features [src, trg]

#print(sum(len(article) for article in cnndm['train']['src']) / len(cnndm['train']['src']))
 # 4,034.67 chars
#print(sum(len(article) for article in cnndm['train']['trg']) / len(cnndm['train']['trg']))
 # 297.76 chars
#print(sum(len(summary) for summary in cnndm['validation']['src']) / len(cnndm['validation']['src']))
 # 3,924.83 chars
#print(sum(len(summary) for summary in cnndm['validation']['trg']) / len(cnndm['validation']['trg']))
# 329.03 chars


