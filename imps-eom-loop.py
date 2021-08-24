#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2021 Hao Zhang<zh970205@mail.ustc.edu.cn>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import numpy as np
import TAT
from tools import StorageFunction, tensor_U, loss_sign, read_from_file, save_to_file
import random
import sys
from hamiltonian import get_H
from get_energy import get_energy
import opt_tools

Tensor = TAT(float)

delta = 1e-5


@StorageFunction
def get_U(n, r, omega, shrink=tuple()):
    return tensor_U(n, n, r, omega, 0, 0).edge_rename({
        "I1": "L",
        "O1": "R",
        "O2": "U",
        "I2": "D"
    }).to(float).shrink({i: 0 for i in shrink})


def random_uniform(a, b):
    return random.random() * (b - a) + a


class IMPS:

    def set_shape(self, depth, length, cutoff, cut1, cut2):
        self.depth = depth
        self.length = length
        self.cutoff = cutoff
        self.cut1 = cut1
        self.cut2 = cut2

        self.projector_1 = [random_uniform(-1, 1) for i in range(8)]
        self.projector_2 = [random_uniform(-1, 1) for i in range(8)]
        self.parameter = [[[random_uniform(-2, 2),
                            random_uniform(-6, 6)]
                           for i in range(self.depth)]
                          for _ in range(2)]

    def get_shape(self):
        return self.depth, self.length, self.cutoff, self.cut1, self.cut2

    def get_shape_size(self):
        return 5

    def get_value(self):
        xs = [*self.projector_1, *self.projector_2]
        for j in range(2):
            for i in range(self.depth):
                xs.append(self.parameter[j][i][0])
                xs.append(self.parameter[j][i][1])
        return xs

    def get_value_size(self):
        return 2 * 8 + 4 * self.depth

    def set_value(self, xs):
        self._energy = None
        self.projector_1 = xs[0:8]
        self.projector_2 = xs[8:16]
        index = 2 * 8
        for j in range(2):
            for i in range(self.depth):
                self.parameter[j][i][0] = xs[index]
                index += 1
                self.parameter[j][i][1] = xs[index]
                index += 1
        return self

    def set_diff(self, down, up):
        self._diff_projector_1 = [[], []]
        self._diff_projector_2 = [[], []]
        self._diff_parameter = [[[[0., 0.]
                                  for i in range(self.depth)]
                                 for _ in range(2)]
                                for _ in range(2)]
        for it, index in enumerate([down, up]):
            xs = [0. for i in range(self.get_value_size())]
            if index != -1:
                xs[index] = delta
            self._diff_projector_1[it] = xs[0:8]
            self._diff_projector_2[it] = xs[8:16]
            index = 2 * 8
            for j in range(2):
                for i in range(self.depth):
                    self._diff_parameter[it][j][i][0] = xs[index]
                    index += 1
                    self._diff_parameter[it][j][i][1] = xs[index]
                    index += 1

    def __init__(self, depth, length, cutoff, cut1, cut2):
        self.H = get_H()
        self.HH = self.H.contract(self.H, {("I1", "O1"), ("I2", "O2")})
        print("H", self.H)
        print("HH", self.HH)
        self.set_shape(depth, length, cutoff, cut1, cut2)

        self._energy = None

        self.set_diff(-1, -1)

    def __call__(self, *, depth, length):
        """
        length in 0 ~ L-1
        depth  in 0 ~ D (Dth for projector)
        不改变上下，当depth > D时
        """
        down_not_up = True
        down_0_up_1 = 0
        if depth > self.depth:
            down_not_up = False
            down_0_up_1 = 1
            depth = 2 * self.depth - depth + 1
        if depth == self.depth:
            projector = Tensor(["D", "U"], [self.cutoff, 2]).zero()
            projector[{"D": 1, "U": 1}] = 1
            if length % 2 == 0:
                projector.block[["D", "U"]][:4, :2] = (
                    np.array(self.projector_1) +
                    np.array(self._diff_projector_1[down_0_up_1])).reshape(
                        [4, 2])
            else:
                projector.block[["D", "U"]][:4, :2] = (
                    np.array(self.projector_2) +
                    np.array(self._diff_projector_2[down_0_up_1])).reshape(
                        [4, 2])
            result = projector
        else:
            shrink = []
            if depth == 0:
                shrink.append("D")
            if length == 0:
                shrink.append("L")
            if length == self.length - 1:
                shrink.append("R")
            if length % 2 == 0:
                result = get_U(
                    self.cutoff, self.parameter[0][depth][0] +
                    self._diff_parameter[down_0_up_1][0][depth][0],
                    self.parameter[0][depth][1] +
                    self._diff_parameter[down_0_up_1][0][depth][1],
                    tuple(shrink))
            else:
                result = get_U(
                    self.cutoff, self.parameter[1][depth][0] +
                    self._diff_parameter[down_0_up_1][1][depth][0],
                    self.parameter[1][depth][1] +
                    self._diff_parameter[down_0_up_1][1][depth][1],
                    tuple(shrink))
        return result

    def get_energies(self, H=None):
        if H is None:
            H = self.H
        e1 = get_energy(self.length,
                        self.depth + 1,
                        self,
                        H,
                        self.length // 2 - 1,
                        self.cut1,
                        self.cut2,
                        double_layer=True)
        e2 = get_energy(self.length,
                        self.depth + 1,
                        self,
                        H,
                        self.length // 2,
                        self.cut1,
                        self.cut2,
                        double_layer=True)
        energy = (e1[0] + e2[0]) / 2

        res = [energy, (e1[1] + e2[1]) / 2, e1[2]]
        return res

    def energy(self):
        if self._energy is None:
            handle.set_diff(-1, -1)
            self._energy = self.get_energies()[0]
        return self._energy

    def norm_proj(self):
        n = np.max(np.abs([*self.projector_1, *self.projector_2]))
        self.projector_1 = [i / n for i in self.projector_1]
        self.projector_2 = [i / n for i in self.projector_2]


handle = read_from_file(IMPS, sys.argv[1])

if sys.argv[2] != "it":
    getattr(opt_tools, sys.argv[2])(handle, sys.argv[1], sys.argv[3:])
    exit()

delta_tau = float(sys.argv[3])
up_bond = 0.05

import opt_tools

t = 0
total_time = 0.
while True:
    t += 1
    print("step", t)
    print("norm proj")
    print("old xs", handle.get_value())
    handle.norm_proj()
    print("new xs", handle.get_value())
    print("measure...")
    size = handle.get_value_size()
    psiHpsi = [None for i in range(size + 1)]
    psipsi = [[None for _ in range(size + 1)] for _ in range(size + 1)]
    handle.set_diff(-1, -1)
    energy, psiHpsi[size], psipsi[size][size] = handle.get_energies()
    print("psipsi", psipsi[size][size])
    _, psiHHpsi, _ = handle.get_energies(handle.HH)

    print("energy =", energy)
    print("saving data")
    save_to_file(handle, sys.argv[1])
    with open(sys.argv[1] + ".log", "a") as file:
        print(total_time, energy, file=file)
    for i in range(size):
        print(i, end=" ", flush=True)
        handle.set_diff(i, -1)
        _, psiHpsi[i], psipsi[i][size] = handle.get_energies()
        handle.set_diff(-1, i)
        _, _, psipsi[size][i] = handle.get_energies()
    for i in range(size):
        for j in range(size):
            print((i, j), end=" ", flush=True)
            handle.set_diff(i, j)
            _, _, psipsi[i][j] = handle.get_energies()
    print("AC...")
    C = [(psiHpsi[size] - psiHpsi[i]) / delta for i in range(size)]
    A = [[
        (psipsi[i][j] + psipsi[size][size] - psipsi[i][size] - psipsi[size][j])
        / (delta**2) for j in range(size)
    ] for i in range(size)]
    C = np.array(C)
    A = np.array(A)
    print("solving...")
    xs = handle.get_value()
    xss, residuals, rank, s = np.linalg.lstsq(A, C, rcond=1e-3)
    print("xs", xs)
    print("C", C)
    print("singular of A", s)
    print("xss", xss)
    print("apply...")

    amp = min(delta_tau,
              up_bond * np.linalg.norm(np.array(xs)) / np.linalg.norm(xss))
    total_time += amp
    xss *= amp

    error = xss @ A @ xss + 2 * amp * xss @ C + amp * amp * psiHHpsi
    print("|psi' d theta + tau H psi|^2", error)
    handle.set_value([i + j for i, j in zip(xs, xss)])
