import os
import json
import time
import torch
import argparse
import numpy as np
from multiprocessing import cpu_count
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict

from utils import experiment_name
from seq2seq import *

def train(model, iterator, optimizer, criterion, clip):    
    model.train()    
    epoch_loss = 0    
    for i, batch in enumerate(iterator):        
        src, src_len = batch.src
        trg = batch.trg        
        optimizer.zero_grad()        
        output, _ = model(src, src_len, trg)        
        loss = criterion(output[1:].view(-1, output.shape[2]), trg[1:].view(-1))        
        loss.backward()        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)        
        optimizer.step()        
        epoch_loss += loss.item()                
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):    
    model.eval()    
    epoch_loss = 0    
    with torch.no_grad():    
        for i, batch in enumerate(iterator):
            src, src_len = batch.src
            trg = batch.trg
            output, _ = model(src, src_len, trg, 0) #turn off teacher forcing
            loss = criterion(output[1:].view(-1, output.shape[2]), trg[1:].view(-1))
            epoch_loss += loss.item()        
    return epoch_loss / len(iterator)


def main(args):    
    input_dim = len(SRC.vocab)
	output_dim = len(TRG.vocab)
	enc_emb_dim = args.embedding_size
	dec_emb_dim = args.embedding_size
	hid_dim = args.hidden_size
	enc_dropout = args.embedding_dropout
	dec_dropout = args.embedding_dropout
	sos_idx = TRG.vocab.stoi['<sos>']

	attn = Attention(hid_dim)
	enc = Encoder(input_dim, enc_emb_dim, hid_dim, enc_dropout)
	dec = Decoder(output_dim, dec_emb_dim, hid_dim, dec_dropout, attn)
	model = Seq2Seq(enc, dec, sos_idx, device).to(device)

	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	pad_idx = TRG.vocab.stoi['<pad>']
	criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
	n_epochs = args.ep
	clip = 1
	save_dir = args.save_model_path
	model_file_path = os.path.join(save_dir, 'model.pt')
	best_valid_loss = float('inf')
	if not os.path.isdir(f'{save_dir}'):
    	os.makedirs(f'{save_dir}')
	for epoch in range(n_epochs):    
    	train_loss = train(model, train_iterator, optimizer, criterion, clip)
    	valid_loss = evaluate(model, valid_iterator, criterion)    
    	if valid_loss < best_valid_loss:
        	best_valid_loss = valid_loss
        	torch.save(model.state_dict(), model_file_path)    
    	print(f'| Epoch: {epoch+1:03} | Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f} |

if __name__ == '__main__':
    parser = argparse.ArgumentParser()   
    parser.add_argument('-ep', '--epochs', type=int, default=10)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5)
    parser.add_argument('-tb', '--tensorboard_logging', action='store_true')
    parser.add_argument('-log', '--logdir', type=str, default='logs')
    parser.add_argument('-bin', '--save_model_path', type=str, default='bin')
    args = parser.parse_args()    
    main(args)

