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
            # dxs : 하류로 흘려보낼 기울기를 담을 그릇.. dx 여러개가 담김.
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            # 각 RNN 계층의 가중치 기울기를 합산하여 최종 결과를 멤버 변수 self.grads에 덮어씀
            self.grads[i][...] = grad
        self.dh = dh

        return dxs

# 순전파 시에 T 개의 Embedding 계층을 준비하고 각 Embedding 계층이 각 시각의 데이터를 처리
class TimeEmbedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers = None
        self.W = W

    def forward(self, xs):
        N, T = xs.shape
        V, D = self.W.shape

        out = np.empty((N, T, D), dtype='f')
        self.layers = []

        for t in range(T):
            layer = Embedding(self.W)
            out[:, t, :] = layer.forward(xs[:, t])
            self.layers.append(layer)

        return out

    def backward(self, dout):
        N, T, D = dout.shape
        grad = 0

        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:, t, :])
            grad += layer.grads[0]

        self.grads[0][...] = grad
        return None


class TimeAffine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        N, T, D = x.shape
        W, b = self.params

        rx = x.reshape(N * T, -1)
        out = np.dot(rx, W) + b

        self.x = x

        return out.reshape(N, T, -1)

    def backward(self, dout):
        x = self.x
        N, T, D = x.shape
        W, b = self.params

        dout = dout.reshape(N * T, -1)
        rx = x.reshape(N * T, -1)

        db = np.sum(dout, axis=0)
        dW = np.dot(rx.T, dout)
        dx = np.dot(dout, W.T)
        dx = dx.reshape(*x.shape)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx

# 시계열 버전의 Softmax
class TimeSoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.ignore_label = -1

    def forward(self, xs, ts):
        N, T, V = xs.shape

        if ts.ndim == 3:  # 정답 레이블이 원핫 벡터인 경우
            ts = ts.argmax(axis=2)

        mask = (ts != self.ignore_label)

        # 배치용과 시계열용을 정리(reshape)
        xs = xs.reshape(N * T, V)
        ts = ts.reshape(N * T)
        mask = mask.reshape(N * T)

        ys = softmax(xs)
        ls = np.log(ys[np.arange(N * T), ts])
        ls *= mask  # ignore_label에 해당하는 데이터는 손실을 0으로 설정
        loss = -np.sum(ls)
        loss /= mask.sum()

        self.cache = (ts, ys, mask, (N, T, V))
        return loss

    def backward(self, dout=1):
        ts, ys, mask, (N, T, V) = self.cache

        dx = ys
        dx[np.arange(N * T), ts] -= 1
        dx *= dout
        dx /= mask.sum()
        dx *= mask[:, np.newaxis]  # ignore_label에 해당하는 데이터는 기울기를 0으로 설정

        dx = dx.reshape((N, T, V))

        return dx