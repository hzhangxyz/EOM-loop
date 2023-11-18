import copy
import math
import numpy as np
import TAT

Tensor = TAT.No.Z.Tensor


def ProbabilityMatrix_balance(M, pd, dc, trun):
    '''
    _M: 通道数
    _pd： 探测器的效率
    _dc: 暗计数，取为1/250000能让结果更平滑一些
    _trun: 截断
    '''

    _P_Nk = np.zeros([M + 1, trun])
    for k in range(trun):
        temp_sum = np.double(0)

        for N in range(M + 1):
            if k < N:
                continue
            temp = 0
            for l in range(N + 1):
                temp += (-1)**l * (1 - dc)**(M - N + l) * math.comb(N, l) * (M - (M - N + l) * pd)**k
            if temp < 0:
                temp = 0
            temp = copy.deepcopy(1 / M**k * math.comb(M, N) * temp)
            temp_sum += abs(temp)
            if temp_sum > 1:
                break
            _P_Nk[N, k] = copy.deepcopy(temp)

    return _P_Nk


def device_tensor(d_in, d_out, pd, dc):
    result = Tensor(["O", "I"], [d_out, d_in])
    result.blocks[["O", "I"]] = ProbabilityMatrix_balance(d_out - 1, pd, dc, d_in)
    return result


# print(device_tensor(8, 10, 0.1, 0.1))
