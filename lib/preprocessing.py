import numpy as np
from nltk.tokenize import WordPunctTokenizer
from subword_nmt.apply_bpe import BPE


simple_tokenizer = WordPunctTokenizer()


def tokenize(x):
    return ' '.join(simple_tokenizer.tokenize(x.lower()))


def preprocess(test):
    bpe = BPE(open('./lib/bpe_rules'))
    
    return np.array([bpe.process_line(tokenize(line)) for line in test['description']])
        