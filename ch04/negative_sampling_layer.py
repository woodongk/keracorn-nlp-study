# coding: utf-8
import sys
sys.path.append('..')
import numpy as np
from common.layers import Embedding, SigmoidWithLoss
import collections

class EmbeddingDot:
    def __init__(self, W):
        # 총 4개의 인스턴스 변수
        self.embed = Embedding(W)  # Embedding 계층
        self.params = self.embed.params  # 매개변수 저장
        self.grads = self.embed.grads  # 기울기 저장
        self.cache = None  # 순전파 시의 계산 결과를 잠시 유지하기 위해 사용되는 변수

    # 순전파 메서드에서는 은닉층 뉴런과 단어 ID의 넘파이 배열(미니배치)을 받는다.
    def forward(self, h, idx):
        target_W = self.embed.forward(idx)  # embedding 계층의 forward(idx)를 호출하여 idx에 해당하는 행 추출
        out = np.sum(target_W * h, axis=1)  # 내적 계산 이후 행마다 더하여 최종결과 out 반환

        self.cache = (h, target_W)
        return out

    def backward(self, dout):
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0], 1)

        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        dh = dout * target_W
        return dh


class UnigramSampler:
    # 초기화 시에 3개의 인수를 받는다
    # 단어 ID 목록, 확률분포에 제곱할 값, 부정적 예시 샘플링할 개수
    def __init__(self, corpus, power, sample_size):
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None

        # 단어 빈도 산출
        counts = collections.Counter()
        for word_id in corpus:
            counts[word_id] += 1

        vocab_size = len(counts)
        self.vocab_size = vocab_size

        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i] = counts[i]

        # 단어의 빈도 기준 확률 분포 산출
        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)

    # target으로 지정한 단어를 긍정적 예로 해석하고, 그 외의 단어 ID를 샘플링
    def get_negative_sample(self, target):
        batch_size = target.shape[0]
        GPU = False
        if not GPU:
            negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)

            for i in range(batch_size):
                p = self.word_p.copy()
                target_idx = target[i]
                p[target_idx] = 0
                p /= p.sum()
                negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)
        else:
            # GPU(cupy）로 계산할 때는 속도를 우선한다.
            # 부정적 예에 타깃이 포함될 수 있다.
            negative_sample = np.random.choice(self.vocab_size, size=(batch_size, self.sample_size),
                                               replace=True, p=self.word_p)

        return negative_sample


class NegativeSamplingLoss:
    # 출력 가중치 W, 말뭉치 ID 리스트, 확률분포에 제곱할 값, 샘플링 횟수
    def __init__(self, W, corpus, power=0.75, sample_size=5):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)

        # 원하는 계층을 리스트로 보관
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)]  # 부정적 예시(sample_size) + 긍정적 예시 (1)
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]

        self.params, self.grads = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, h, target):
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target)  # 부정적 예를 샘플링하여 변수에 저장

        # 긍정적 예 순전파. 0번째 계층
        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype=np.int32)  # 1
        loss = self.loss_layers[0].forward(score, correct_label)

        # 부정적 예 순전파
        negative_label = np.zeros(batch_size, dtype=np.int32)  # 0
        for i in range(self.sample_size):
            negative_target = negative_sample[:, i]
            score = self.embed_dot_layers[1 + i].forward(h, negative_target)
            loss += self.loss_layers[1 + i].forward(score, negative_label)

        return loss

    def backward(self, dout=1):
        dh = 0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = l0.backward(dout)
            dh += l1.backward(dscore)

        return dh