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


class TimeRNN:
    def __init__(self, Wx, Wh, b, stateful=False):  # stateful : 은닉상태 인계 받을지 여부
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None  # T 개의 RNN 계층 리스트로 저장하는 용도

        # h : forward() 메서드 이후 마지막 RNN 계층의 은닉상태 저장
        # dh : backward() 메서드 이후 하나의 앞 블록의 은닉 상태의 기울기 저장
        self.h, self.dh = None, None
        # True : 아무리 긴 시계열 데이터여도 순전파를 끊지 않고 전파
        # False : 은닉 상태를 영행렬 (모든 요소가 0 행렬)로 초기화
        self.stateful = stateful

    def set_state(self, h):  # 은닉상태 설정
        self.h = h

    def reset_state(self):  # 은닉상태 초기화
        self.h = None

    # 순전파에서 입력 xs를 받는다
    # xs : T 개 분량의 시계열 데이터를 하나로 모은 것
    def forward(self, xs):
        Wx, Wh, b = self.params
        # 미니배치크기 N, 시계열 데이터 T개, 입력 벡터 차원수 D
        N, T, D = xs.shape
        D, H = Wx.shape

        self.layers = []
        # 출력값 담을 그릇
        hs = np.empty((N, T, H), dtype='f')

        # "stateful이 false" 이거나 "처음 호출 " 일때 영행렬로 초기화
        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')

        # RNN 계층이 각 시간 t의 은닉 상태 h를 계산하고 이를 hs에 저장
        for t in range(T):
            layer = RNN(*self.params)
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)

        # forward가 처음 호출되면 h에는 마지막 RNN 계층의 은닉 상태가 저장됨
        # 다음번 forward 호출 시 stateful이 True면 먼저 저장된 h 값이 그대로 이용되고 False면 영행렬로 초기화
        return hs

    # 역전파
    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D, H = Wx.shape

        dxs = np.empty((N, T, D), dtype='f')
        dh = 0
        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            # RNN 계층의 순전파에서는 출력이 2개로 분기되어 역전파에서 각 기울기가 합산되어 전해짐
            dx, dh = layer.backward(dhs[:, t, :] + dh)  # --> 합산된 기울기
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh

        return dxs