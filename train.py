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


from ptb import PTB
from utils import to_var, idx2word, expierment_name
from seq2seq import *

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
SOS_IDX = TRG.vocab.stoi['<sos>']

attn = Attention(HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, DEC_DROPOUT, attn)
model = Seq2Seq(enc, dec, SOS_IDX, device).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
pad_idx = TRG.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

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


N_EPOCHS = 10
CLIP = 1
SAVE_DIR = 'models'
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'model.pt')

best_valid_loss = float('inf')
if not os.path.isdir(f'{SAVE_DIR}'):
    os.makedirs(f'{SAVE_DIR}')

for epoch in range(N_EPOCHS):    
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)    
    print(f'| Epoch: {epoch+1:03} | Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f} |
