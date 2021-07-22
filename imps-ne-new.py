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

import TAT
import tools
import random
import sys
from hamiltonian import get_H
from get_energy import get_lattice_energy

Tensor = TAT(float)


@tools.StorageFunction
def get_U(n, r, omega, shrink=tuple()):
    return tools.tensor_U(n, n, r, omega, 0, 0).edge_rename({
        "I1": "L",
        "O1": "R",
        "O2": "U",
        "I2": "D"
    }).to(float).shrink({i: 0
                         for i in shrink})


def random_uniform(a, b):
    return random.random() * (b - a) + a


class IMPS:
    def __init__(self, depth, length, cutoff, cut1, cut2):
        self.depth = depth
        self.length = length
        self.cutoff = cutoff
        self.cut1 = cut1
        self.cut2 = cut2

        self.projector_1 = random_uniform(-1, 1)
        self.projector_2 = random_uniform(-1, 1)
        self.parameter = [[[random_uniform(-2, 2),
                            random_uniform(-6, 6)] for i in range(self.depth)]
                          for _ in range(2)]

    def __call__(self, *, depth, length):
        if depth == self.depth:
            projector = Tensor(["D", "U"], [self.cutoff, 2]).zero()
            projector[{"D": 1, "U": 1}] = 1
            if length % 2 == 0:
                projector[{"D": 0, "U": 0}] = self.projector_1
            else:
                projector[{"D": 0, "U": 0}] = self.projector_2
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

    @staticmethod
    def loss_sign(a):
        if a < 0:
            return 0
        else:
            return 0.1 * a * a * a

    def energy(self):
        if self._energy:
            return self._energy
        e1 = get_lattice_energy(self.length, self.depth + 1, self, get_H(),
                                self.length // 2 - 1, self.cut1, self.cut2)
        e2 = get_lattice_energy(self.length, self.depth + 1, self, get_H(),
                                self.length // 2, self.cut1, self.cut2)
        self._energy = (e1 + e2) / 2

        self._energy += self.loss_sign(abs(self.projector_1) - 1)
        self._energy += self.loss_sign(abs(self.projector_2) - 1)
        for l in self.parameter:
            for r, omega in l:
                self._energy += self.loss_sign(abs(r) - 2)

        return self._energy

    def get_value(self):
        xs = [self.projector_1, self.projector_2]
        for j in range(2):
            for i in range(self.depth):
                xs.append(self.parameter[j][i][0])
                xs.append(self.parameter[j][i][1])
        return xs

    def set_value(self, xs):
        self._energy = None
        self.projector_1 = xs[0]
        self.projector_2 = xs[1]
        index = 2
        for j in range(2):
            for i in range(self.depth):
                self.parameter[j][i][0] = xs[index]
                index += 1
                self.parameter[j][i][1] = xs[index]
                index += 1
        return self

    def gradient(self):
        delta = 0.0001
        E = self.energy()
        xs = self.get_value()
        gradient = []
        for i in range(self.depth * 4 + 2):
            xss = xs[:]
            xss[i] += delta
            new_E = self.set_value(xss).energy()
            gradient.append((new_E - E) / delta)
        self.set_value(xs)
        self._energy = E
        return gradient

    def save_to_file(self, file_name):
        with open(file_name, "w") as file:
            print(self.depth,
                  self.length,
                  self.cutoff,
                  self.cut1,
                  self.cut2,
                  file=file)
            print(*self.get_value(), file=file)
            print(self.energy(), file=file)

    @staticmethod
    def read_from_file(file_name):
        with open(file_name, "r") as file:
            config = [i for i in file.read().split()]
            imps = IMPS(depth=int(config[0]),
                        length=int(config[1]),
                        cutoff=int(config[2]),
                        cut1=int(config[3]),
                        cut2=int(config[4]))

            if len(config) > 5:
                imps.set_value([float(i) for i in config[5:]])
                print("READ")
            else:
                print("RANDOM")
            return imps


handle = IMPS.read_from_file(sys.argv[1])

import opt_tools

getattr(opt_tools, sys.argv[2])(handle, sys.argv[1], sys.argv[3:])
