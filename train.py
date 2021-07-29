import argparse


from skip_gram import Skipgram
from trainer import Trainer

import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.vocab import build_vocab_from_iterator

from utils import yield_tokens, create_contexts_target

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--train_fn', required=True)
    p.add_argument('--hidden_size', type=int, default=100)
    p.add_argument('--sample_size', type=int, default=5)
    p.add_argument('--window_size', type=int, default=2)
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--n_epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--verbose', type=int, default=2)
    
    config = p.parse_args()
    return config

def main(config):
    vocab = build_vocab_from_iterator(yield_tokens(config.train_fn), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])

    if config.gpu_id >= 0:
        device = torch.device('cuda:%d'%config.gpu_id)
    else:
        device = torch.device('cpu')

    cs, labels = create_contexts_target(config.train_fn,
                                        config.window_size,
                                        vocab,
                                        config.sample_size)

    cs, labels = cs.to(device), labels.to(device)

    model = Skipgram(len(vocab), config.hidden_size).to(device)
    optimizer = optim.Adam(model.parameters())
    crit = nn.BCEWithLogitsLoss().to(device)
    trainer = Trainer(model, optimizer, crit)
    
    model = trainer.train(cs, labels, config)

    torch.save({
        'model':model.state_dict(),
        'config':config,
        'vocab':vocab
    }, config.model_fn)



if __name__ == "__main__":
    config = define_argparser()
    main(config)