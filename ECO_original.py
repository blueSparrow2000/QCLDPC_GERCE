import numpy as np
import numba

''' 
ECO - GERCE에 사용되는 ECO 방법
the original method
1. col swap to get 1
2. row swap to get 1
3. col elimination

-
'''

@numba.jit(nopython=True, parallel=True)
def ECO_original(M,Q, BGCE = True):
    m,n = M.shape # we know m>n (Q is nxn)
    # L = np.identity(m,dtype=np.uint8) # left multiplication matrix (dont need)
    # print(m,n) m: 2000, n: 1000
    i=0
    j=0
    while i < m and j < n:
        if M[i, j] == 0: # need to fetch 1
            k1 = np.argmax(M[i,j:]) + j
            if M[i,k1] == 0: # no column 1 found
                k2 = np.argmax(M[i:,j]) + i
                if M[k2,j] == 0: # no row 1 found
                    i+=1 # We are making LTM matrix => i is incremented
                    continue
                # row swap
                temp = np.copy(M[k2])
                M[k2] = M[i]
                M[i] = temp

                # Q is nxn and n<m (index out of range) so row operation is applied on other matrix L
                # temp = np.copy(L[k2])
                # L[k2] = L[i]
                # L[i] = temp

            else: # col 1 found!
                # col swap
                temp = np.copy(M[:,k1])
                M[:,k1] = M[:,j]
                M[:,j] = temp

                temp = np.copy(Q[:,k1])
                Q[:,k1] = Q[:,j]
                Q[:,j] = temp

        # Column elimination
        row = np.copy(M[i])
        row[j] = 0
        if not BGCE:
            row[:j] =0

        # for jx in range(n):
        for jx in numba.prange(n):
            if row[jx]:
                M[i:,jx] ^= M[i:,j]
                Q[:,jx] ^= Q[:,j]

        i += 1
        j += 1

    return M, Q # returns result and ECO matrix








