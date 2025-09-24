import numpy as np
from util import *

# a = np.array([[1,0,0],[0,0,0] ,[1,1,1] ])
# b = np.array([0,0,0])
#
# # print(b.tolist() in a.tolist())
#
# x = np.where(np.all(a==b,axis=1))
#
# if x[0].size>0:
#     idx = x[0][0]
#     c = np.delete(a, (idx), axis=0)
#     print(c)

####### save to an image of the recovered H matrix
H_final = read_matrix("H_recovered")
A = read_matrix("noisy_codeword")
A_error_free = read_matrix('error_free_codeword')
save_image_data(H_final, "recovered_qc_test_save")


