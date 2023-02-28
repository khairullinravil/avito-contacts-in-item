import numpy as np
from nltk.tokenize import WordPunctTokenizer
from subword_nmt.apply_bpe import BPE


simple_tokenizer = WordPunctTokenizer()


def tokenize(x):
    return ' '.join(simple_tokenizer.tokenize(x.lower()))


def preprocess(test):
    test['description'].apply(tokenize).to_csv('./test_description', index=False, header=False)
    bpe = BPE(open('./lib/bpe_rules'))
#     with open('test.bpe', 'w') as f_out:
#         for line in open('test_description'):
#             f_out.write(bpe.process_line(line.strip()) + '\n')
#     return np.array(open('./test.bpe').read().split('\n')[:-1])
    return np.array([bpe.process_line(tokenize(line)) for line in test['description']])
        