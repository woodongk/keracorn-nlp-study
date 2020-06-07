import numpy as np
import sys

sys.path.append("../")
import matplotlib.pyplot as plt
import seaborn as sns

from common.optimizer import SGD
from dataset import ptb
from simple_rnnlm import SimpleRnnlm

# 하이퍼파라미터
batch_size = 10
wordvec_size = 100
hidden_size = 100  # RNN의 은닉 상태 벡터의 원소 수
time_size = 5  # Truncated BPTT가 한 번에 펼치는 시간 크기
lr = 0.1
max_epoch = 100

# 학습 데이터 읽기
corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_size = 1000
corpus = corpus[:corpus_size]
vocab_size = int(max(corpus) + 1)

xs = corpus[:-1]  # 입력
ts = corpus[1:]  # 출력(정답 레이블) - xs보다 한 단어 앞
data_size = len(xs)  # 999
print('말뭉치 크기: %d, 어휘 수: %d' % (corpus_size, vocab_size))

# 학습 시 사용하는 변수
max_iters = data_size // (batch_size * time_size)
time_idx = 0
total_loss = 0
loss_count = 0
ppl_list = []

# 모델 생성
model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)

########################################
# 1 - 각 미니배치에서 샘플을 읽을 위치를 계산
jump = (corpus_size - 1) // batch_size
offsets = [i * jump for i in range(batch_size)]
# [0, 99, 198, 297, 396, 495, 594, 693, 792, 891] -> 오프셋의 각 원소에 데이터를 읽는 시작 위치가 담긴다!
########################################

for epoch in range(max_epoch):
    for iter in range(max_iters):
        ########################################
        # 2 - 미니배치 획득
        batch_x = np.empty((batch_size, time_size), dtype='i')  # 그릇 준비
        batch_t = np.empty((batch_size, time_size), dtype='i')  ##그릇 준비
        for t in range(time_size):  # time idx를 순차적으로 늘리면서 말뭉치에서 time idx위치의 데이터를 확보한다
            for i, offset in enumerate(offsets):
                batch_x[i, t] = xs[(offset + time_idx) % data_size]
                batch_t[i, t] = ts[(offset + time_idx) % data_size]
            time_idx += 1
        ########################################

        # 기울기를 구하여 매개변수 갱신
        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)
        total_loss += loss
        loss_count += 1

    ########################################
    # 3 - 에포크마다 퍼플렉시티 평가
    ppl = np.exp(total_loss / loss_count)
    print("| epoch %d | perplexity %.2f" % (epoch + 1, ppl))
    ppl_list.append(float(ppl))
    total_loss, loss_count = 0, 0
    ########################################

plt.figure(figsize=(7,5))
sns.lineplot(list(range(100)),ppl_list)
plt.show()