import math
import os
import torch.nn as nn
import torch.functional as F 
import torch.optim as optim
import numpy as np 
from filedataset import Gigaword_Dataset
from model import RNN_VAE
import argparse
import random
import time

parser = argparse.ArgumentParser(
	description='Controlable Text Generation'
	)
parser.add_argument('--gpu', default=True, action='store_true',
					help='whether to run in the GPU')
parser.add_argument('--model', default='ctext', metavar='')
args = parser.parse_args()

batch_size = 32
z_dim = 20
h_dim = 64
lr = 1e-3
c_dim = 2
dataset = Giagaword_Dataset()
model = RNN_VAE(dataset.n_vocab, h_dim, z_dim, c_dim, p_word_dropout=0,3,
	pretrained_embeddings=dataset.get_vocab_vectors(), freeze_embeddings=True,
	gpu=args.gpu)
if args.gpu:
	model.load_state_dict(torch.load('models/{}.bin'.format(args.model)))
else:
	model.load_state_dict(torch.load('models/{}'.bin'.format(args.model), map_location=lambda storage, loc: storage))
z = model.sample_z_prior(1)
c = model.sample_c_prior(1)










