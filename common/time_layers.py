import numpy as np
from common.layers import *
from common.functions import softmax, sigmoid


class RNN:
    def __init__(self, Wx, Wh, b):  # 가중치 2개와 편향 1개 인수로
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]  # numpy.zeros_like : shape 유지하고 0으로 초기화
        self.cache = None  # *** 역전파 계산 시 사용하는 중간 데이터 담는 곳

    def forward(self, x, h_prev):
        Wx, Wh, b = self.params
        t = np.matmul(h_prev, Wh) + np.matmul(x, Wx) + b  # Main 식
        h_next = np.tanh(t)  # 다음 시각 계층으로의 입력

        self.cache = (x, h_prev, h_next)
        return h_next

    def backward(self, dh_next):
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache

        dt = dh_next * (1 - h_next ** 2)  # tanh 미분
        db = np.sum(dt, axis=0)
        dWh = np.matmul(h_prev.T, dt)
        dh_prev = np.matmul(dt, Wh.T)
        dWx = np.matmul(x.T, dt)
        dx = np.matmul(dt, Wx.T)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev
