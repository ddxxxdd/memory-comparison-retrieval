"""方案中的原文 BM25 和混合检索对照，不参与描述选择主流程。"""

from collections import Counter
import math
import re

import numpy as np


def tokenize(text):
    # 中文按字，拉丁文字按词；固定词法分词，不引入额外语义模型。
    return re.findall(r"[\u3400-\u9fff]|[^\W_\u3400-\u9fff]+", text.lower(), flags=re.UNICODE)


class BM25:
    def __init__(self, contents):
        self.docs = [Counter(tokenize(text)) for text in contents]
        self.lengths = [sum(doc.values()) for doc in self.docs]
        self.average = max(sum(self.lengths) / len(self.docs), 1)
        self.df = Counter(term for doc in self.docs for term in doc)

    def score(self, query):
        scores = np.zeros(len(self.docs), dtype=np.float64)
        count = len(self.docs)
        for word in set(tokenize(query)):
            frequency = self.df.get(word, 0)
            idf = math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for i, doc in enumerate(self.docs):
                tf = doc.get(word, 0)
                denominator = tf + 1.2 * (1 - 0.75 + 0.75 * self.lengths[i] / self.average)
                scores[i] += idf * tf * 2.2 / denominator
        return scores


def rank(scores, memories):
    return sorted(range(len(memories)), key=lambda i: (-float(scores[i]), memories[i].memory_id))


def reciprocal_rank_fusion(first, second, memories):
    scores = np.zeros(len(memories), dtype=np.float64)
    for values in (first, second):
        for position, index in enumerate(rank(values, memories), 1):
            scores[index] += 1 / (60 + position)
    return scores
