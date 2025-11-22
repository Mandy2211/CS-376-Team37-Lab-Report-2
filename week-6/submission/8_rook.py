import numpy as np

def rook_net(sz=8, n_it=100, thr=0, n_r=8):
    n = sz * sz
    wt = np.zeros((n, n))
    for i in range(sz):
        for j in range(sz):
            for k in range(sz):
                for l in range(sz):
                    if i == k and j != l:
                        wt[i * sz + j, k * sz + l] = -1
                    if j == l and i != k:
                        wt[i * sz + j, k * sz + l] = -1

    st = np.zeros(n, dtype=int)
    rpos = np.random.choice(n, n_r, replace=False)
    for p in rpos:
        st[p] = 1

    print("Initial State:")
    print(st.reshape(sz, sz))

    for it in range(n_it):
        ust = st.copy()
        for i in range(n):
            ni = np.dot(wt[i], ust)
            ust[i] = 1 if ni > thr else 0
        if np.array_equal(st, ust):
            break
        st = ust

    print("Final State:")
    print(st.reshape(sz, sz))
    return st.reshape(sz, sz)

np.random.seed(42)
sol = rook_net()