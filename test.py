import os
import json
import torch
import argparse
import math
import matplotlib.pyplot as plt
from seq2seq import *


def translate_sentence(sentence):   
    tokenized = tokenize_src(sentence) #
    tokenized = ['<sos>'] + [t.lower() for t in tokenized] + ['<eos>'] 
    numericalized = [SRC.vocab.stoi[t] for t in tokenized] 
    sentence_length = torch.LongTensor([len(numericalized)]).to(device) #need sentence length for masking
    tensor = torch.LongTensor(numericalized).unsqueeze(1).to(device) #convert to tensor and add batch dimension
    translation_tensor_probs, attention = model(tensor, sentence_length, None, 0) #pass through model to get translation probabilities
    translation_tensor = torch.argmax(translation_tensor_probs.squeeze(1), 1) #get translation from highest probabilities
    translation = [TRG.vocab.itos[t] for t in translation_tensor][1:] #ignore the first token, just like we do in the training loop
    return translation, attention[1:] #ignore first attention array


def display_attention(candidate, translation, attention):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    translation = translation[:translation.index('<eos>')] #cut translation after first <eos> token
    attention = attention[:len(translation)].squeeze(1).cpu().detach().numpy() #cut attention to same length as translation
    cax = ax.matshow(attention, cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + ['<sos>'] + [t.lower() for t in tokenize_src(candidate)] + ['<eos>'], rotation=90)
    ax.set_yticklabels([''] + translation)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()

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

    if not os.path.exists(args.load_checkpoint):
        raise FileNotFoundError(args.load_checkpoint)

    model.load_state_dict(torch.load(args.load_checkpoint))
    print("Model loaded from %s" % args.load_checkpoint)

    if torch.cuda.is_available():
        model = model.cuda()
    
    test_loss = evaluate(model, test_iterator, criterion)
    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

    candidate = ' '.join(vars(train_data.examples[0])['src'])
    candidate_translation = ' '.join(vars(train_data.examples[0])['trg'])
    print(candidate)
    print(candidate_translation)
    translation, attention = translate_sentence(candidate)
    print(translation)
    display_attention(candidate, translation, attention)
    
    candidate = ' '.join(vars(valid_data.examples[0])['src'])
    candidate_translation = ' '.join(vars(valid_data.examples[0])['trg'])
    print(candidate)
    print(candidate_translation)
    translation, attention = translate_sentence(candidate)
    print(translation)
    display_attention(candidate, translation, attention)

    candidate = ' '.join(vars(test_data.examples[0])['src'])
    candidate_translation = ' '.join(vars(test_data.examples[0])['trg'])
    print(candidate)
    print(candidate_translation)
    translation, attention = translate_sentence(candidate)
    print(translation)
    display_attention(candidate, translation, attention)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--load_checkpoint', type=str, default='bin/model.pt')  
    args = parser.parse_args()
    main(args)
