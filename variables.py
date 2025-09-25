'''
Some variables to input
'''
import numpy as np

############### LDPC variable ###############
mb = 4                   # number of parity check blocks in a column
nb = 32                  # number of blocks in a column
Z = 64                   # lifting factor = block size (choose 64,128,256,...)
pooling_factor = 10      # amount of codeword generated (= codeword length * pooling factor)

# variables used for generating codewords
skip_generation = True   # skip generation (use already made files in H_saves folder)
error_free = False       # generate codewords with no errors
noise_level= 10          # set to 10 if you want to use NOISE_PROB
NOISE_PROB = 0.0001      # BER
np.random.seed(seed=0)   # seed for generation of QC LDPC code

# options for GERCE operation
BGCE = True                # without BGCE, one may get more dual vectors but they are likely to be erronous
get_all_block_shifts = True  # get block shifts of recovered dual vector -> ONLY IF BLOCK SIZE Z IS KNOWN!
use_gauss_elim = False     # if you want to use gauss elimination (ERO) instead of ECO, set this to True
load_H_recovered = False   # If you already have an H_recovered file, just set this to True
remove_duplicate_block_shift = True # option to remove the parity check vector obtained by block shift from H_recovered
recovered_data_mode = 'a'  # 'w': write mode / 'a': append mode, add recovered PCM into existing H file, appending after finding a new parity check block

sampling_rate = 0.5        # what proportion of columns do you want to sample from the n bit codeword
Nmin = 50                  # amount of GERCE operation in one main loop (can be determined by v, n and BER)
GERCE_iter = 1             # number permutation operation in GERCE (1 => dont permute)
total_iteration = 7#10     # full iteration number - decoding is done this amount
w_threshold = 13+2         # threshold value for number of ones in each recovered parity check vectors. used for sparsification

############################################ DO NOT CHANGE ###################################################
target_rate = 1 - mb / nb
codeword_len = nb * Z
databit_num = codeword_len - mb * Z

density = 0.15  # number of ones in a P matrix
if codeword_len > 500:
    density = 0.08
elif codeword_len >= 2000:
    density = 0.05

LARGE_CODE = True if codeword_len > 50 else False
parity_num = codeword_len-databit_num

threshold = round(((pooling_factor-1)*codeword_len)*0.325)  # suggested beta coeff on the paper
if not BGCE: # if GCE, higher the threshold
    threshold = round((pooling_factor*codeword_len)*0.325) # codeword_amount/2 * beta_opt (0.65, based on experience)

PRINT_VAR_SETTING = True
# print variable settings
if PRINT_VAR_SETTING:
    print('#' * 5, "variable settings", '#' * 5)
    print("(n, k) = ({}, {})".format(codeword_len, databit_num))
    print("Number of code words: ", pooling_factor*codeword_len)
    print("Threshold: ", threshold)
    print('#' * 30)
############################################ DO NOT CHANGE ###################################################