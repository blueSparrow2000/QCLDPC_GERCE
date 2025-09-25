'''
Implementation of LDPC PCM recovering method of paper:
A fast reconstruction of the parity check matrices of LDPC codes in a noisy environment (2021)
'''
from QCLDPC_sampler import *
from submatrix_sampler import *
from block_recover import *
from verifier import *
from formatter import *
from util import *
from bit_flip_decoder_sequential import ldpc_bitflip_seqdecode_numba
from GERCE import GERCE
from sparsifier import *
import time
start_time = time.time()

# random seed for recovering process
np.random.seed(2)

######################## generating / loading QC LDPC code & H ########################
if error_free:
    noise_level = 0

if skip_generation:
    H = read_matrix("H_true")
    A = read_matrix("noisy_codeword")
    A_error_free = read_matrix('error_free_codeword')
else:
    # 1. sample LDPC code word
    H, B, S = generate_nand_qc_ldpc(mb=mb, nb=nb, Z=Z, target_rate=target_rate, rng=1234, sparse=True)  #
    save_matrix(H, 'H_true')
    save_image_data(H, filename="n_{}_k_{}".format(codeword_len, databit_num))
    H_diag = diag_format(H, databit_num)
    generator = get_generator(H, databit_num)
    A = get_codewords(generator, codeword_len, databit_num, pooling_factor=pooling_factor, noise_level=noise_level,
                      save_noise_free=True)
    A_error_free = read_matrix('error_free_codeword')
    save_matrix(A, filename='noisy_codeword')

total_codewords = pooling_factor * codeword_len
correct_codewords = compare_matrix(A_error_free, A)
print("Correct / Total = {} / {}".format(correct_codewords, total_codewords))

print("Elapsed time: %s seconds" % round(time.time() - start_time, 3))
#############################################################################################
'''
H_recovered: GERCE로 직접적으로 얻은 벡터 -> 이걸로 linearly independent vector를 체킹하는 용도. block shift다 구해서 복원한 뒤 하면 느리기 때문에
newerly_recovered_vector: H_recovered의 일부. block shift하려고 모아둔 벡터. main loop돌때마다 리셋된다. 이걸로 block shift해서 H_final을 구한다
H_final: 모든 block shift를 다 구한 최종 벡터. 최종 복운된 H matrix이다.
'''
H_recovered = None  # currently found dual vectors - L.I check하기 위해 소량만 들고있는거
H_final = None      # H_recover의 모든 row에 block shift를 취하여 불린 모든 parity check vector
decoding_codeword_matrix = np.copy(A)
col_sample_amount = round(codeword_len * sampling_rate)
recovered_vector_count = 0           # num_dual_vectors

# append mode initialization
if load_H_recovered and recovered_data_mode == 'a':
    H_final = read_matrix('H_recovered')  # currently found dual vectors
    H_recovered = np.copy(H_final)
    if not (H_final is None) and H_final.size == 0:  # initial step but append mode
        H_final = None
        H_recovered = None

mainIter = 0
while mainIter < total_iteration:
    print('='*20)
    print("{}th main loop".format(mainIter+1))
    print('='*20)

    # this vector is used for only block shift purposes
    newerly_recovered_vec = None  # keep track of number of recovered vectors in a total loop

    for i in range(Nmin):
        # 1. Sample col indices
        col_indices = sample_col_indices(decoding_codeword_matrix, col_sample_amount)
        Mv = decoding_codeword_matrix[:, col_indices]

        if use_gauss_elim:
            # gauss elim method
            Gs = gf2elim(Mv)
            mv, _ = Gs.shape
            # zero pad Hv to form PCM in dim n space
            G_recovered = np.zeros((mv, codeword_len),
                                   dtype=np.uint8)
            G_recovered[:, col_indices] = Gs

            # Below is the work that needs to be done after formatting!
            P = G_recovered[:, databit_num:]
            P_t = np.transpose(P)
            H_pad = np.concatenate((P_t, np.identity(parity_num, dtype=np.uint8)), axis=1)
            mv = parity_num

        else:
            # 2. GERCE
            Hv = GERCE(Mv, Niter=GERCE_iter, QCLDPC=True, use_ECO=True)

            # 3. sparsification
            H_sparse = general_sparsifier(Hv, databit_num, w_th=w_threshold, use_row_combinations=False)
            mv, nv = H_sparse.shape

            # 4. zero pad Hv to form PCM in dim n space
            H_pad = np.zeros((mv, codeword_len), dtype=np.uint8)
            H_pad[:, col_indices] = H_sparse

        # 5. remove vectors that does not satisfy block constraint(weight 1 in a row per block etc.)
        H_candidate = get_block_candidates(H_pad, codeword_len, Z)
        if not H_candidate:
            mv = 0
        else:
            H_candidate = np.array(H_candidate)
            mv,_ = H_candidate.shape

        # 6. check L.I.
        for j in range(mv):  # for each recovered dual vectors, check Linear Independency with previously found dual vectors
            h = H_candidate[j]  # a dual vector
            if H_recovered is None:
                H_recovered = np.array([h])  # initial vector
                recovered_vector_count += 1

                newerly_recovered_vec = np.array([h])  # initial vector
                continue
            Htemp = np.append(H_recovered, [h], axis=0)
            if np.linalg.matrix_rank(Htemp) > recovered_vector_count:  # increase rank
                H_recovered = np.append(H_recovered, [h], axis=0)  # add h into H
                recovered_vector_count += 1

                if newerly_recovered_vec is None:
                    newerly_recovered_vec = np.array([h])  # initial vector
                else:
                    newerly_recovered_vec = np.append(newerly_recovered_vec, [h], axis=0)  # add h into H

        if (i + 1) % 10 == 0:  # show every 10th iteration count
            print("{}th iteration".format(i + 1))
            print("Current vectors: ", recovered_vector_count)

    H_final_changed = False
    # 7. recover blocks - if a parity check vector can already be made by block shifting another vector, it is removed
    if get_all_block_shifts:
        if not (newerly_recovered_vec is None):  # if H_recovered is not None, and there is newerly found vectors
            for dual_vector in newerly_recovered_vec:  # for dual_vector in H_candidate: - sample L.I ones
                shifts = None
                if H_final is None:
                    shifts = qc_global_cyclic_shifts_numba(dual_vector, Z)  # shift (when block size is given, Z)
                    H_final = np.array(shifts)
                else:
                    # if dual vector is already in a array -> skip this vector
                    if remove_duplicate_block_shift:
                        del_idx = np.where(np.all(dual_vector == H_final, axis=1))
                        print()
                        print("index of duplicate block shift vector located in H_final",del_idx[0])
                        # print("Does it really exist? ",del_idx[0].size>0)
                        if del_idx[0].size>0: # dual_vector.tolist() in H_final.tolist()
                            # remove dual_vector from H_recovered if a dual vector can be obtained from block shifting
                            del_in_H_recov = np.where(np.all(dual_vector == H_recovered, axis=1))
                            print("index of duplicate block shift vector located in H_recovered",del_in_H_recov[0])
                            H_recovered = np.delete(H_recovered, (del_in_H_recov[0][0]), axis=0) # remove the first occurence
                            recovered_vector_count -= 1
                            continue
                    shifts = qc_global_cyclic_shifts_numba(dual_vector, Z)
                    H_final = np.concatenate((H_final, shifts), axis=0)

                if recovered_data_mode == 'a' and (shifts is not None):
                    save_matrix(shifts, 'H_recovered', mode='a')
                    H_final_changed = True
        else:
            print("H_recovered is None")
    else:
        H_final = H_recovered
        H_final_changed = True

    if H_recovered is not None:
        print("Shape of H_recovered", H_recovered.shape)
    if H_final is not None:
        print("Shape of H_final", H_final.shape)

    # 8. decoding using hard decision bit flip
    if not error_free and not (newerly_recovered_vec is None) and H_final_changed:  # only when something new is discovered & H_final is changed
        # H_final.astype(np.uint8)
        # A.astype(np.uint8)
        print("Decoding...")
        decoded_codeword_matrix, ok, _, _, _ = ldpc_bitflip_seqdecode_numba(H_final, A, max_iter=50)
        print("Decoding complete", end=' - ')
        print(" %s seconds" % round(time.time() - start_time, 3))
        correct_codewords = compare_matrix(A_error_free, decoded_codeword_matrix)
        print("Correct / Total = {} / {}".format(correct_codewords, total_codewords))
        decoding_codeword_matrix = np.copy(decoded_codeword_matrix)
    else:
        print(" %s seconds" % round(time.time() - start_time, 3))
    mainIter += 1

this_time = time.time()
print("Main loop done - ", end='')
print("Elapsed time: %s seconds" % round(this_time - start_time, 3))

# reliability extraction
# H_final = reliability_extraction(H_final, A, round(parity_num/Z)+2) # we need block seeds -> get as much as we can? parity_num amount?
# this_time = time.time()
# print("reliability extraction done - ", end = '')
# print("Elapsed time: %s seconds" % round(this_time - start_time,3))
# print(H_final.shape)

print()
print("Success?: ", check_success(H, H_final))

try:
    if recovered_data_mode == 'w':
        save_matrix(H_final, 'H_recovered')
        save_image_data(H_final, "recovered_qc")
    elif recovered_data_mode == 'a':  # append mode
        # load the matrix and draw a new binary image - used in append mode
        H_recovered = read_matrix("H_recovered")
        save_image_data(H_recovered, "recovered_qc")
except:
    print("saving img of H failed. Here is what H looks like")
    print(H)

'''
Comment:
희안하게도 이 방법으로 구하면 progressive reconst로 구한 벡터의 여집합이 잘 구해진다... 하
서로 강점인 분야가 다른건가? ECO 기반 방식과 gauss elim기반 방식의 차이점인걸까? 
만약 그렇다면 QC 실험에 gauss elim이 아닌 ECO를 사용해서 한다면 이 실험처럼 마지막 두칸이 빈 블록 뺴고 다 나올지도

=> 그럼 앙상블처럼 gauss elim을 이용하는 방식과 ECO를 쓰는 방식을 모두 써야 한다
'''