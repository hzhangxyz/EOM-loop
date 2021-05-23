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
