import collections
import json
from typing import Tuple

from sberbank_baseline.utils import singleton


class NgramManager:
    def __init__(self):
        self.word2num, self.gram_freq = self.load_all()

    @singleton
    def load_all(self):
        word2num = None
        with open('data/word2num.json', 'r', encoding='utf-8') as fin:
            word2num = json.load(fin)

        gram_freq = collections.defaultdict(int)
        for filename in ['data/2grams-3.txt', 'data/3grams-3.txt']:
            with open(filename, 'r', encoding='utf-8') as fin:
                for line in fin:
                    tokens = line.strip().split('\t')
                    gram_freq[tuple([word2num[e] for e in tokens[1:]])] = int(tokens[0])

        return word2num, gram_freq

    def get_freq(self, n_gram: Tuple[str, ...]):
        return self.gram_freq[tuple([self.word2num.get(g, -1) for g in n_gram])]
