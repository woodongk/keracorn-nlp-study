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

# 단어 ID로 이루어진 리스트로부터 동시 발생 행렬 만들어주는 함수
def create_co_matrix(corpus, vocab_size, window_size=1):
    # corpus : 단어 아이디의 리스트
    # vocab_size : 어휘 수

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