import TAT
from han2.tensor_U import tensor_U
from measure_device import device_tensor

Tensor = TAT.No.Z.Tensor


def contract_whole_mps(mps):
    result = None
    for tensor in mps:
        if result is None:
            result = tensor
        else:
            result = result.contract(tensor, {("R", "L")})
    return result


def cut_mps(mps, mps_cut):
    length = len(mps)
    for i in range(length - 1):
        q, r = mps[i].qr('r', {"R"}, "R", "L")
        mps[i] = q
        mps[i + 1] = mps[i + 1].contract(r, {("L", "R")})
    for i in reversed(range(1, length)):
        u, s, v = mps[i].svd({"L"}, "R", "L", "L", "R", mps_cut)
        mps[i] = v
        mps[i - 1] = mps[i - 1].contract(u.contract(s, {("R", "L")}), {("R", "L")})


def normalize_state(v):
    return v / v.norm_2()


def chain(length, physics_cut, mps_cut, u1, u2):
    middle = u1.shrink({"I2": 0}).edge_rename({"I1": "L", "O1": "R", "O2": "P"})
    left = u1.shrink({"I2": 0, "I1": 0}).edge_rename({"O1": "R", "O2": "P"})
    right = left.same_shape().edge_rename({"R": "L"}).identity({("L", "P")})
    # construct mps
    mps = [left] + [middle] * (length - 1) + [right]
    # mps = [
    #     tensor.edge_rename({
    #         "L": "L1",
    #         "R": "R1",
    #         "P": "P1"
    #     }).contract(tensor.edge_rename({
    #         "L": "L2",
    #         "R": "R2",
    #         "P": "P2"
    #     }), set()).merge_edge({
    #         "L": ["L1", "L2"],
    #         "R": ["R1", "R2"],
    #     }) for tensor in mps
    # ]
    # import numpy as np
    # np.set_printoptions(linewidth=1000, precision=2, suppress=True)
    # s = normalize_state(contract_whole_mps([tensor.edge_rename({"P1": f"P{i}", "P2": f"P{i}'"}) for i, tensor in enumerate(mps)]))
    # print(s[{"P0": 1, "P1": 1, "P0'": 0, "P1'": 0}])
    # print(s[{"P0": 0, "P1": 0, "P0'": 1, "P1'": 1}])
    # print(s[{"P0": 1, "P1": 0, "P0'": 0, "P1'": 1}])
    # print(s[{"P0": 0, "P1": 1, "P0'": 1, "P1'": 0}])
    # exit()
    # 2 mps with unitary
    mps = [
        tensor.edge_rename({
            "L": "L1",
            "R": "R1",
        }).contract(u2, {("P", "I1")}).contract(tensor.edge_rename({
            "L": "L2",
            "R": "R2",
        }), {("I2", "P")}).merge_edge({
            "L": ["L1", "L2"],
            "R": ["R1", "R2"],
        }) for tensor in mps
    ]
    # import numpy as np
    # np.set_printoptions(linewidth=1000, precision=2, suppress=True)
    # s = normalize_state(contract_whole_mps([tensor.edge_rename({"O1": f"P{i}", "O2": f"P{i}'"}) for i, tensor in enumerate(mps)]))
    # print(s[{"P0": 1, "P1": 1, "P0'": 0, "P1'": 0}])
    # print(s[{"P0": 0, "P1": 0, "P0'": 1, "P1'": 1}])
    # print(s[{"P0": 1, "P1": 0, "P0'": 0, "P1'": 1}])
    # print(s[{"P0": 0, "P1": 1, "P0'": 1, "P1'": 0}])
    # exit()
    # Cut it
    cut_mps(mps, mps_cut)
    # Trace density matrix
    mps = [
        tensor.edge_rename({
            "L": "L1",
            "R": "R1",
            "O1": "P1"
        }).contract(tensor.conjugate().edge_rename({
            "L": "L2",
            "R": "R2",
            "O1": "P2",
        }), {("O2", "O2")}).merge_edge({
            "L": ["L1", "L2"],
            "R": ["R1", "R2"],
        }) for tensor in mps
    ]
    # Cut it
    cut_mps(mps, mps_cut)
    return mps


def possibility_of_given_n(ns, L, D, Dc, r1, omega1, phi1, psi1, r2, omega2, phi2, psi2):
    mps = chain(L, D, Dc, tensor_U(D, D, r=r1, omega=omega1, phi=phi1, psi=psi1), tensor_U(D, D, r=r2, omega=omega2, phi=phi2, psi=psi2))
    mps2 = [tensor.trace({("P1", "P2")}) for i, tensor in enumerate(mps)]
    result = []
    for n in ns:
        mps1 = [tensor.shrink({"P1": n[i], "P2": n[i]}) for i, tensor in enumerate(mps)]
        result.append(complex(contract_whole_mps(mps1) / contract_whole_mps(mps2)).real)
    return result


def total_parity(d_out, pd, dc, L, D, Dc, r1, omega1, phi1, psi1, r2, omega2, phi2, psi2):
    device = device_tensor(D, d_out, pd, dc)
    parity = Tensor(["P"], [d_out])
    for i in range(d_out):
        parity.storage[i] = +1 if i % 2 == 0 else -1
    device_parity = device.contract(parity, {("O", "P")})
    # for i in range(D):
    #     device_parity.storage[i] = +1 if i % 2 == 0 else -1
    mps = chain(L, D, Dc, tensor_U(D, D, r=r1, omega=omega1, phi=phi1, psi=psi1), tensor_U(D, D, r=r2, omega=omega2, phi=phi2, psi=psi2))
    #state = contract_whole_mps([tensor.trace(set(), {f"P{i}": ("P1", "P2")}) for i, tensor in enumerate(mps)])
    #print(state)
    mps1 = [tensor.trace(set(), {"P": ("P1", "P2")}).contract(device_parity, {("P", "I")}) for i, tensor in enumerate(mps)]
    mps2 = [tensor.trace({("P1", "P2")}) for i, tensor in enumerate(mps)]
    return complex(contract_whole_mps(mps1) / contract_whole_mps(mps2)).real


import math

possibility = possibility_of_given_n(
    ns=[[i, j] for i in range(4) for j in range(4)],
    L=1,
    D=10,
    Dc=-1,
    r1=2,
    omega1=0,
    phi1=math.pi,
    psi1=0,
    r2=0,
    omega2=math.pi / 2,
    phi2=0,
    psi2=math.pi / 2,
)
print(f"{possibility=}")
parity = total_parity(
    d_out=11,
    pd=0.8,
    dc=0.0,
    L=1,
    D=10,
    Dc=-1,
    r1=2,
    omega1=0,
    phi1=math.pi,
    psi1=0,
    r2=0,
    omega2=math.pi / 2,
    phi2=0,
    psi2=math.pi / 2,
)
print(f"{parity=}")
