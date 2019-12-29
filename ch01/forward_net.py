# 계층으로 클래스화 및 순전파 구현

import numpy as np

# 모든 계층은 forward 와 backward 메서드를 가진다.
# 모든 계층은 인스턴트 변수인 params와 grads 를 가진다.

# 매개변수 보관 장소. list params [], 하나의 리스트에 보관하면 매개변수 갱신과 매개변수 저장을 손쉽게 처리할 수 있다

class Sigmoid: # 시그모이드 계층
    def __init__(self):

        self.params = [] # sigmoid는 학습하는 매개변수 없으므로 빈 리스트로 초기화

    def forward(self, x):
        return 1 / (1 + np.exp(-x))


class Affine: # 완전 연결 계층
    def __init__(self,W,b):

        # 초기화 시에 가중치와 편향 받음
        self.params = [W,b] # 신경망이 학습될 때 수시로 갱신되는 Affine 계층의 매개변수.

    def forward(self,x):
        W,b = self.params
        out = np.matmul(x,W) + b # input * weight + bias
        return out

class TwolayerNet:
    def __init__(self, input_size, hidden_size, output_size):

        I, H, O = input_size, hidden_size, output_size

        # 가중치와 편향 초기화
        W1 = np.random.randn(I,H) #가우시안 분포의 난수
        b1 = np.random.randn(H)
        W2 = np.random.randn(H,O)
        b2 = np.random.randn(O)

        # 3개의 계층 생성
        self.layers = [
            Affine(W1,b1),
            Sigmoid(),
            Affine(W2,b2)
        ]

        # 모든 가중치를 리스트에 모은다.
        self.params = []
        for layer in self.layers:
            self.params += layer.params

    def predict(self,x):
        for layer in self.layers:
            x = layer.forward(x)
        return x


x = np.random.randn(10,2)
model = TwolayerNet(2,4,3)
s = model.predict(x)