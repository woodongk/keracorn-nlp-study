import sys
sys.path.append("..")
import numpy as np
from common.functions import softmax
from ch06_게이트가_추가된_RNN.rnnlm import Rnnlm
from ch06_게이트가_추가된_RNN.better_rnnlm import BetterRnnlm

class RnnlmGen(Rnnlm):
    # 문장 생성 수행. 100 단어까지
    def generate(self, start_id, skip_ids=None, sample_size=100):
        ''' start_id : 최초로 주는 단어의 ID
            skip_ids : 해당 리스트에 속하는 단어 ID는 샘플링 되지 않도록 방지. (전처리된 단어 등)
        '''
        word_ids = [start_id]

        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(-1, 1)  # 미니배치 처리때문에 x는 2차원. 1X1로 변형.
            score = self.predict(x)  # 각 단어의 점수 출력
            p = softmax(score.flatten())  # softmax 함수 통해 정규화하여 확률분포 p 얻기

            sampled = np.random.choice(len(p), size=1, p=p)  # p로부터 다음 단어 샘플링
            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(x))

        return word_ids

    def get_state(self):
        return self.lstm_layer.h, self.lstm_layer.c

    def set_state(self, state):
        self.lstm_layer.set_state(*state)

class BetterRnnlmGen(BetterRnnlm):
    def generate(self, start_id, skip_ids=None, sample_size=100):
        word_ids = [start_id]

        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1, 1)
            score = self.predict(x).flatten()
            p = softmax(score).flatten()

            sampled = np.random.choice(len(p), size=1, p=p)
            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(x))

        return word_ids

    def get_state(self):
        states = []
        for layer in self.lstm_layers:
            states.append((layer.h, layer.c))
        return states

    def set_state(self, states):
        for layer, state in zip(self.lstm_layers, states):
            layer.set_state(*state)