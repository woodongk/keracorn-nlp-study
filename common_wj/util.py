import numpy as np

# 말뭉치 단어 별로 아이디로 변환해주는 함수
def preprocess(text):
    # 간단한 전처리
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')

    # 단어를 Id로
    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word

def create_co_matrix(corpus, vocab_size, window_size=1):
    '''단어 ID로 이루어진 리스트로부터 동시 발생 행렬 만들어주는 함수

        :param corpus: 단어 아이디의 리스트
        :param vocab_size : 어휘 수
        :return: 동시 발생 행렬
    '''

    corpus_size = len(corpus)  # 단어의 숫자
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i  # 윈도우 사이즈만큼 왼쪽으로
            right_idx = idx + i  # 윈도우 사이즈만큼 오른쪽으로

            if left_idx >= 0:  # 문장 빠져나가지 않을 때까지
                left_word_id = corpus[left_idx]  # index를 통해 단어의 id로 구성된 corpus를 통해 id 받아냄
                co_matrix[word_id][left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id][right_word_id] += 1

    return co_matrix

# 코사인 유사도 구하기
def cos_similarity(x, y, eps=1e-8):
    # epsilon을 추가하여 divide by zero 오류 방지
    nx = x / np.sqrt(np.sum(x**2) + eps) # x의 정규화
    ny = y / np.sqrt(np.sum(y**2) + eps) # y의 정규화
    return np.dot(nx,ny)


# 동시발생 행렬을 PPMI 행렬로 변환하는 함수
def ppmi(C, verbose=False, eps=1e-8):
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)  # return scalar
    S = np.sum(C, axis=0)  # return array, row 단위로 합치기
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2((C[i, j] * N) / (S[i] * S[j]) + eps)
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                k = total // 100  # integer division
                if (k != 0) & (cnt % k == 0):
                    print("%.1f%% 완료" % (100 * cnt / total), end='\r')
    return M

def create_contexts_target(corpus, window_size=1):
    target = corpus[window_size:-window_size]  # 맥락의 개수가 채워지지 않는 양 끝 단어는 제외
    contexts = []

    for idx in range(window_size, len(corpus) - window_size):
        cs = []  # context_per_target
        for t in range(-window_size, window_size + 1):  # target=0을 기준으로 window_size만큼 좌우
            if t == 0:
                continue
            cs.append(corpus[idx + t])

        contexts.append(cs)

    return np.array(contexts), np.array(target)


def convert_one_hot(corpus, vocab_size):
    '''one-hot encoding 으로 변환

    param corpus: 단어 ID목록(1차원 혹은 2차원의 NumPy배열)
    param vocab_size: 어휘수(unique)
    :return:one-hot표현(2차원 혹은 3차원의 NumPy배열)
    '''
    N = corpus.shape[0]

    if corpus.ndim == 1:  # target
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)  # unique한 어휘 개수로 one hot length 부여됨.
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1  # 한 단어 당 하나의 one-hot

    elif corpus.ndim == 2:  # contexts
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):  # word_id 개수만큼 다시 반복분돌기
                one_hot[idx_0, idx_1, word_id] = 1

    return one_hot

def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate