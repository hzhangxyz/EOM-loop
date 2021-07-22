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
from numpy import sin, cos

Tensor = TAT(float)


@tools.StorageFunction
def get_U(n, r, omega):
    return tools.tensor_U(n, n, r, omega, 0, 0).edge_rename({
        "I1": "D",
        "O1": "U"
    }).to(float).merge_edge({"R": ["I2", "O2"]})


def get_fake_id(n):
    res = Tensor(["D", "U", "A", "B"], [n, n, n, n]).identity({("D", "A"),
                                                               ("U", "B")})
    return res.merge_edge({"L": ["A", "B"]})


def get_id(n):
    res = Tensor(["D", "U"], [n, n]).identity({("D", "U")})
    return res


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
        self.rotator_1 = random_uniform(0, 6)
        self.projector_2 = random_uniform(-1, 1)
        self.rotator_2 = random_uniform(0, 6)
        self.parameter = [[random_uniform(-2, 2),
                           random_uniform(-6, 6)] for i in range(self.depth)]

        self.fake_identity = get_fake_id(cutoff)
        self.identity = get_id(cutoff)

        self._energy = None

    def __call__(self, *, depth, length):
        # l -> 2*l + 1
        # d -> 2*d
        if depth == self.depth * 2 + 1 - 1:
            projector = Tensor(["D", "U"], [self.cutoff, 2]).zero()
            projector[{"D": 1, "U": 1}] = 1
            if length % 2 == 0:
                p = self.projector_1
                r = self.rotator_1
            else:
                p = self.projector_2
                r = self.rotator_2
            projector[{"D": 0, "U": 0}] = p * cos(r)
            projector[{"D": 1, "U": 0}] = p * sin(r)
            projector[{"D": 0, "U": 1}] = -sin(r)
            projector[{"D": 1, "U": 1}] = cos(r)
            result = projector
        elif depth % 2 == 0:
            "PDC"
            if length == 2 * self.length + 1 - 1:
                "identity"
                result = self.identity
            elif length % 2 == 0:
                "real"
                result = get_U(self.cutoff, 0, self.parameter[depth // 2][1])
            else:
                "fake"
                result = self.fake_identity
        else:
            "BS"
            if length == 0:
                "identity"
                result = self.identity
            elif length % 2 == 1:
                "real"
                result = get_U(self.cutoff, self.parameter[depth // 2][0], 0)
            else:
                "fake"
                result = self.fake_identity
        if depth == 0:
            result = result.shrink({"D": 0})
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
        e1 = get_lattice_energy(self.length * 2 + 1, self.depth * 2 + 1, self,
                                get_H(), self.length, self.cut1, self.cut2)
        e2 = get_lattice_energy(self.length * 2 + 1, self.depth * 2 + 1, self,
                                get_H(), self.length + 1, self.cut1, self.cut2)
        self._energy = (e1 + e2) / 2

        self._energy += self.loss_sign(abs(self.projector_1) - 1)
        self._energy += self.loss_sign(abs(self.projector_2) - 1)
        for r, omega in self.parameter:
            self._energy += self.loss_sign(abs(r) - 2)

        return self._energy

    def get_value(self):
        xs = [
            self.projector_1, self.rotator_1, self.projector_2, self.rotator_2
        ]
        for i in range(self.depth):
            xs.append(self.parameter[i][0])
            xs.append(self.parameter[i][1])
        return xs

    def set_value(self, xs):
        self._energy = None
        self.projector_1 = xs[0]
        self.rotator_1 = xs[1]
        self.projector_2 = xs[2]
        self.rotator_2 = xs[3]
        index = 4
        for i in range(self.depth):
            self.parameter[i][0] = xs[index]
            index += 1
            self.parameter[i][1] = xs[index]
            index += 1
        return self

    def gradient(self):
        delta = 0.0001
        E = self.energy()
        xs = self.get_value()
        gradient = []
        for i in range(self.depth * 2 + 4):
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
