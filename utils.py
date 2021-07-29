import io
import re
import numpy as np
import torch
from torchtext.data.utils import get_tokenizer

def yield_tokens(file_path):
    tokenizer = get_tokenizer('basic_english')
    with io.open(file_path, encoding='utf-8') as f:
        for line in f:
            _, text = line.split('\t')
            text= re.sub('<br />', "", text)
            text = re.sub('[^a-zA-Z]', " ", text)
            yield tokenizer(text)
            

def create_contexts_target(file_path, window_size, vocab, sample_size):
    tokenizer = get_tokenizer('basic_english')
    corpus = []
    vocab_size = len(vocab)
    p = np.zeros(vocab_size)
    
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            _, text = line.split('\t')
            text= re.sub('<br />', "", text)
            text = re.sub('[^a-zA-Z]', " ", text)
            text = vocab(tokenizer(text))
            for t in text:
                p[t] += 1
            corpus.append(text)
    
    targets = []
    contexts = []
    labels = []

    p = np.power(p, 0.75)
    p /= np.sum(p)

    n_sampler = NegativeSampler(vocab_size, window_size, sample_size, p)

    print('문맥-타겟 단어 생성 및 네거티브 샘플링 데이터 생성 시작')
    for c in corpus:
        for idx in range(window_size, len(c)-window_size):
            cs = []
            for t in range(-window_size, window_size+1):
                if t == 0:
                    continue
                # c[idx] : target
                # c[idx+t] : context
                cs.append(c[idx+t])
            targets.append(c[idx])
            contexts.append(cs)
            labels.append(1)
            negative_sample = n_sampler.get_negative_sample(c[idx])
            targets += [c[idx]]*sample_size
            contexts += negative_sample.tolist()
            labels += [0]*sample_size
    print(targets[:10])
    print(contexts[:10])
    cs = torch.cat([torch.LongTensor(targets).unsqueeze(1), torch.LongTensor(contexts)], dim=-1)
    labels = torch.FloatTensor(labels)
    print('문맥-타겟 단어 생성 및 네거티브 샘플링 데이터 생성 끝')
    return cs, labels


class NegativeSampler:
    def __init__(self, vocab_size, window_size, sample_size, p):
        self.vocab_size = vocab_size
        self.window_size = window_size
        self.sample_size = sample_size
        self.p = p

    def get_negative_sample(self, target):
        p = self.p.copy()
        p[target] = 0
        p /= p.sum()
        negative_sample = np.random.choice(self.vocab_size, size=(self.sample_size, self.window_size*2), replace=False, p=p)
        return negative_sample