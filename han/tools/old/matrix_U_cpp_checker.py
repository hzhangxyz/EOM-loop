#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import e, sqrt
import numpy as np
from tools import StorageFunction, tensor_U
import random
import TAT
import time

Tensor = TAT(complex)


@StorageFunction
def get_a(n):
    result = Tensor(["I", "O"], [n, n]).zero()
    for i in range(n - 1):
        result[{"I": i + 1, "O": i}] = sqrt(i + 1)
    return result


@StorageFunction
def get_a_dagger(n):
    result = get_a(n)
    return result.edge_rename({"I": "O", "O": "I"})


@StorageFunction
def get_U_eom(n, r, omega, phi, psi):
    a1 = get_a(n).edge_rename({"I": "I1", "O": "O1"})
    a2 = get_a(n).edge_rename({"I": "I2", "O": "O2"})
    a1_dagger = get_a_dagger(n).edge_rename({"I": "I1", "O": "O1"})
    a2_dagger = get_a_dagger(n).edge_rename({"I": "I2", "O": "O2"})
    term1 = (e**(1.j * psi)) * a1.contract(a2_dagger, set())
    term2 = (e**(-1.j * psi)) * a1_dagger.contract(a2, set())
    return ((term1 - term2) * (omega / 2.)).exponential(
        {("I1", "O1"), ("I2", "O2")}, step=10)


@StorageFunction
def get_U_spdc(n, r, omega, phi, psi):
    a1 = get_a(n).edge_rename({"I": "I1", "O": "O1"})
    a2 = get_a(n).edge_rename({"I": "I2", "O": "O2"})
    a1_dagger = get_a_dagger(n).edge_rename({"I": "I1", "O": "O1"})
    a2_dagger = get_a_dagger(n).edge_rename({"I": "I2", "O": "O2"})
    term1 = (e**(1.j * phi)) * a1.contract(a2, set())
    term2 = (e**(-1.j * phi)) * a1_dagger.contract(a2_dagger, set())
    return ((term1 - term2) * (r / 2.)).exponential({("I1", "O1"),
                                                     ("I2", "O2")},
                                                    step=2)


def get_U(n, r, omega, phi, psi):
    return get_U_eom(n, r, omega, phi,
                     psi).contract(get_U_spdc(n, r, omega, phi, psi),
                                   {("O1", "I1"), ("O2", "I2")})


amp = 3


def check(n, r, omega, phi, psi):
    t1 = time.time()
    a = get_U(n * amp, r, omega, phi, psi).block[["O1", "O2", "I1", "I2"]]
    t2 = time.time()
    b = tensor_U(n, n, r, omega, phi, psi).block[["O1", "O2", "I1", "I2"]]
    t3 = time.time()
    global exp_time
    global gen_time
    exp_time += t2 - t1
    gen_time += t3 - t2
    a = a[:n, :n, :n, :n].reshape([n * n, n * n])
    b = b[:n, :n, :n, :n].reshape([n * n, n * n])
    diff = a - b
    print("%e" % (np.linalg.norm(diff) / np.linalg.norm(b)))


def random_uniform(a, b):
    return random.random() * (b - a) + a


for n in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
    print("n=%d" % n)
    global exp_time
    global gen_time
    exp_time = 0.
    gen_time = 0.
    for _ in range(10):
        r = random_uniform(-2, 2)
        omega = random_uniform(-6, +6)
        phi = random_uniform(-6, +6)
        psi = random_uniform(-6, +6)
        check(n, r, omega, phi, psi)
    print("exp:%s\tgen:%s" % (exp_time, gen_time))
