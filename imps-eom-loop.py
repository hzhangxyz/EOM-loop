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
from tools import StorageFunction, tensor_U, loss_sign, read_from_file
import random
import sys
from hamiltonian import get_H
from get_energy import get_energy

Tensor = TAT(float)


@StorageFunction
def get_U(n, r, omega, shrink=tuple()):
    return tensor_U(n, n, r, omega, 0, 0).edge_rename({
        "I1": "L",
        "O1": "R",
        "O2": "U",
        "I2": "D"
    }).to(float).shrink({i: 0
                         for i in shrink})


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
                            random_uniform(-6, 6)] for i in range(self.depth)]
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

    def __init__(self, depth, length, cutoff, cut1, cut2):
        self.set_shape(depth, length, cutoff, cut1, cut2)

        self._energy = None

    def __call__(self, *, depth, length):
        if depth == self.depth:
            projector = Tensor(["D", "U"], [self.cutoff, 2]).zero()
            projector[{"D": 1, "U": 1}] = 1
            if length % 2 == 0:
                projector.block[["D", "U"]][:4, :2] = np.array(
                    self.projector_1).reshape([4, 2])
            else:
                projector.block[["D", "U"]][:4, :2] = np.array(
                    self.projector_2).reshape([4, 2])
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
                result = get_U(self.cutoff, self.parameter[0][depth][0],
                               self.parameter[0][depth][1], tuple(shrink))
            else:
                result = get_U(self.cutoff, self.parameter[1][depth][0],
                               self.parameter[1][depth][1], tuple(shrink))
        return result

    def energy(self):
        if self._energy:
            return self._energy
        e1 = get_energy(self.length, self.depth + 1, self, get_H(),
                        self.length // 2 - 1, self.cut1, self.cut2)
        e2 = get_energy(self.length, self.depth + 1, self, get_H(),
                        self.length // 2, self.cut1, self.cut2)
        self._energy = (e1 + e2) / 2

        for l in self.parameter:
            for r, omega in l:
                self._energy += loss_sign(abs(r) - 2)

        return self._energy


handle = read_from_file(IMPS, sys.argv[1])

import opt_tools

getattr(opt_tools, sys.argv[2])(handle, sys.argv[1], sys.argv[3:])
