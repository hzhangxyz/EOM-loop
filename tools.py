#!/usr/bin/env python
# -*- coding: utf-8 -*-

import weakref

import numpy as np
import TAT
from matrix_A import matrix_A
from matrix_U import matrix_U

Tensor = TAT(complex)


class StorageFunction:
    def __init__(self, func):
        self.func = func
        self.storage = {}

    def __call__(self, *args, **kwargs):
        kwtuple = tuple(kwargs.items())
        if args not in self.storage:
            self.storage[(args, kwtuple)] = self.func(*args, **kwargs)
        return self.storage[args, kwtuple]


@StorageFunction
def tensor_U(n, c, r, omega, phi, psi):
    """
    根据矩阵截断, 物理截断, r, omega, phi, psi创建U矩阵
    """
    fake_cut = 2 * c - 1
    data = np.array(matrix_U(fake_cut, r, omega, phi, psi))
    result = Tensor(["I1", "I2", "O1", "O2"], [n, n, n, n]).zero()
    result.block[{}][:c, :c, :c, :c] = data[:c, :c, :c, :c]
    return result


@StorageFunction
def tensor_A(n, c, k, eta):
    """
    根据矩阵截断, 物理截断, k, eta创建A矩阵
    """
    result = Tensor(["O", "I"], [n, n])
    result.block[{}] = matrix_A(n, c, k, eta)
    return result


def two_to_one(down, up, cut, endtag_down="", endtag_up=""):
    A = down.copy()
    B = up.copy()

    length = len(A)
    A[0] = A[0].edge_rename({"L": "L" + endtag_down})
    A[length - 1] = A[length - 1].edge_rename({"R": "R" + endtag_down})
    B[0] = B[0].edge_rename({"L": "L" + endtag_up})
    B[length - 1] = B[length - 1].edge_rename({"R": "R" + endtag_up})

    C = []
    for a, b in zip(A, B):
        a_l = TAT.Name("L") in a.name
        a_r = TAT.Name("R") in a.name
        a = a.edge_rename({"L": "AL", "R": "AR"})

        b_l = TAT.Name("L") in b.name
        b_r = TAT.Name("R") in b.name
        b = b.edge_rename({"L": "BL", "R": "BR"})

        c = a.contract(b, {("U", "D")})
        merge = {}
        rename = {}
        if a_l and b_l:
            merge["L"] = ["AL", "BL"]
        elif a_l:
            rename["AL"] = "L"
        elif b_l:
            rename["BL"] = "L"
        if a_r and b_r:
            merge["R"] = ["AR", "BR"]
        elif a_r:
            rename["AR"] = "R"
        elif b_r:
            rename["BR"] = "R"
        if merge:
            c = c.merge_edge(merge)
        if rename:
            c = c.edge_rename(rename)
        C.append(c)

    for i in range(length - 2):
        Q, R = C[i].qr('r', {"R"}, "R", "L")
        C[i] = Q
        C[i + 1] = C[i + 1].contract(R, {("L", "R")})
        # last one is l-2-1+1=l-2, chain end is l-1, it is ok

    parameter = 1.0
    for i in reversed(range(length - 1)):
        tensor_l = C[i]
        tensor_r = C[i + 1]
        name_l = {j for j in tensor_l.name if j != "R"}
        name_r = {j for j in tensor_r.name if j != "L"}
        map_l_1 = {j: "L-" + str(j) for j in name_l}
        map_r_1 = {j: "R-" + str(j) for j in name_r}
        map_l_2 = {"L-" + str(j): j for j in name_l}
        map_r_2 = {"R-" + str(j): j for j in name_r}

        big = tensor_l.edge_rename(map_l_1).contract(
            tensor_r.edge_rename(map_r_1), {("R", "L")})
        u, s, v = big.svd({"L-" + str(j) for j in name_l}, "R", "L", cut)
        norm = s.norm_max()
        s /= norm
        parameter *= norm
        C[i + 1] = v.edge_rename(map_r_2)
        C[i] = u.edge_rename(map_l_2).multiple(s, "R", 'u')

    return np.array(C), parameter


def contract_single_line(A):
    result = None
    for a in A:
        if result is None:
            result = a
        else:
            result = result.contract(a, {("R", "L")})
    return result


def conjugate_line(A):
    result = []
    for a in A:
        result.append(a.edge_rename({"D": "U", "U": "D"}).conjugate())
    return np.array(result)


@StorageFunction
def tensor_AA(n, c, eta):
    As = [tensor_A(n, c, k, eta) for k in range(n)]
    return sum([
        A.edge_rename({
            "I": "UI",
            "O": "UO"
        }).conjugate().contract(A.edge_rename({
            "I": "DI",
            "O": "DO"
        }), set()) for A in As
    ])


def uniform(mean, delta):
    if delta is None:
        return [mean]
    else:
        sample = 5
        return [
            mean * (1 + delta * i / sample)
            for i in range(-sample, sample + 1)
        ]


@StorageFunction
def tensor_UU(n,
              c,
              r,
              omega,
              phi,
              psi,
              delta_r=None,
              delta_omega=None,
              delta_phi=None,
              delta_psi=None,
              shrink_I2=False):
    UUs = []
    for real_r in uniform(r, delta_r):
        for real_omega in uniform(omega, delta_omega):
            for real_phi in uniform(phi, delta_phi):
                for real_psi in uniform(psi, delta_psi):
                    U = tensor_U(n, c, real_r, real_omega, real_phi, real_psi)
                    if shrink_I2:
                        U = U.shrink({"I2": 0})
                    UU = U.edge_rename({
                        "I1": "UI1",
                        "O1": "UO1",
                        "I2": "UI2",
                        "O2": "UO2"
                    }).conjugate().contract(
                        U.edge_rename({
                            "I1": "DI1",
                            "O1": "DO1",
                            "I2": "DI2",
                            "O2": "DO2"
                        }), set())
                    UUs.append(UU)
    return sum(UUs) / len(UUs)


@StorageFunction
def tensor_UUAA(n,
                c,
                r,
                omega,
                phi,
                psi,
                delta_r=None,
                delta_omega=None,
                delta_phi=None,
                delta_psi=None,
                eta_1=None,
                eta_2=None,
                shrink_I2=False):
    UU = tensor_UU(n, c, r, omega, phi, psi, delta_r, delta_omega, delta_phi,
                   delta_psi, shrink_I2)
    if eta_1 is not None:
        if eta_1 != 1:
            UU = UU.contract(tensor_AA(n, c, eta_1),
                             {("UO1", "UI"), ("DO1", "DI")}).edge_rename({
                                 "UO":
                                 "UO1",
                                 "DO":
                                 "DO1"
                             })
    if eta_2 is not None:
        if eta_2 != 1:
            UU = UU.contract(tensor_AA(n, c, eta_2),
                             {("UO2", "UI"), ("DO2", "DI")}).edge_rename({
                                 "UO":
                                 "UO2",
                                 "DO":
                                 "DO2"
                             })
    return UU


class LazyHandle:
    def __init__(self, func, *args, **kwargs):
        self._value = None
        self._downstream = set()
        self._func = func
        self._args = args
        self._kwargs = kwargs

        for i in self._args:
            if isinstance(i, LazyHandle):
                i._downstream.add(weakref.ref(self))

    def reset(self, value=None):
        if self._value != value:
            self._value = value
            for i in self._downstream:
                # 也可以选择再__del__中去掉，但是不知道为什么，对于global var会有问题
                if i():
                    i().reset()

    @property
    def value(self):
        if self._value is None:
            self._value = self._func(*map(self._unwrap_handle, self._args),
                                     **self._kwargs)
        return self._value

    @staticmethod
    def _unwrap_handle(handle):
        if isinstance(handle, LazyHandle):
            return handle.value
        else:
            return handle


def LazyRoot(value=None):
    result = LazyHandle(lambda: None)
    result.reset(value)
    return result


def loss_sign(a):
    if a < 0:
        return 0
    else:
        return 0.1 * a * a * a


def save_to_file(lattice, file_name):
    with open(file_name, "w") as file:
        print(*lattice.get_shape(), file=file)
        print(*lattice.get_value(), file=file)
        print(lattice.energy(), file=file)


def read_from_file(cls, file_name):
    with open(file_name, "r") as file:
        config = [[j for j in i.split()] for i in file.read().split("\n")]
        imps = cls(*map(int, config[0]))

        if len(config) != 1 and len(config[1]) != 0:
            imps.set_value([float(i) for i in config[1]])
            print("READ")
        else:
            print("RANDOM")
        return imps


def gradient(lattice):
    delta = 0.0001
    E = lattice.energy()
    xs = lattice.get_value()
    gradient = []
    for i in range(lattice.get_value_size()):
        xss = xs[:]
        xss[i] += delta
        new_E = lattice.set_value(xss).energy()
        gradient.append((new_E - E) / delta)
    lattice.set_value(xs)
    lattice._energy = E
    return gradient
