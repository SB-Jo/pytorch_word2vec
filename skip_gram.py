import torch
import torch.nn as nn

class Skipgram(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        self.in_embedding = nn.Embedding(vocab_size, hidden_size)
        self.out_embedding = nn.Embedding(vocab_size, hidden_size)

    def forward(self, x):
        # |x| = (batch_size, 1+window_size*2)
        
        target, context = x[:, 0].unsqueeze(1), x[:, 1:]
        # |target| = (bs, 1)
        # |context| = (bs, window_size*2)

        target = self.in_embedding(target)
        # |target| = (bs, 1, hs)

        context = self.out_embedding(context)
        # |context| = (bs, window-size*2, hs)        
        
        x = torch.bmm(target, context.transpose(1,2))
        # |x| = (bs, 1, window_size*2)

        x = torch.mean(x, dim=-1)

        return x

