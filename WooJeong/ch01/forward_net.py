# 계층으로 클래스화 및 순전파 구현

import numpy as np

class Sigmoid: # 시그모이드 계층
    def __init__(self):
        self.params = [] # 매개변수들은 params 인스턴스 변수에 보관한다., sigmoid는 학습하는 매개변수 없으므로 빈 리스트로 초기화

    def forward(self, x):
        return 1 / (1 + np.exp(-x))


class Affine:
    def __init__(self,W,b):
        self.params = [W,b] # 초기화 시에 가중치와 편향을 받음. Affine 계층의 매개변수

    def forward(self,x):
        W,b = self.params
        out = np.matmul(x,W) + b # input * weight + bias
        return out

#class TwolayerNet:
#    def _init__(self, input_size, hidden_size, output_size):
