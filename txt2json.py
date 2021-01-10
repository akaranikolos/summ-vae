import json
for name in ['train', 'valid', 'test']:
    with open(f'./data/cnndm/{name}.source', 'r') as f:
        src = f.readlines()
    with open(f'./data/cnndm/{name}.target', 'r') as f:
        trg = f.readlines()
    with open(f'./data/cnndm/{name}.json', 'w') as f:
        for s, t in zip(src, trg):
            example = {'src': s, 'trg': t}
            json.dump(example, f)
            f.write('\n')

            