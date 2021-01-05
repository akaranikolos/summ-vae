import json
for name in ['train', 'dev', 'test']:
    with open(f'./data/{name}.source', 'r') as f:
        src = f.readlines()
    with open(f'./data/{name}.target', 'r') as f:
        trg = f.readlines()
    with open(f'./data/gigaword/{name}.json', 'w') as f:
        for s, t in zip(src, trg):
            example = {'src': s, 'trg': t}
            json.dump(example, f)
            f.write('\n')

            