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

import os
import sys
import random
import TAT
import tools

Tensor = TAT(float)


def get_U(n, r, omega):
    return tools.tensor_U(n, n, r, omega, 0, 0).edge_rename({
        "I1": "L",
        "O1": "R",
        "O2": "U",
        "I2": "D"
    }).to(float)


# Hamiltonian is selected by environment variable
def get_H():
    n = 2
    result = Tensor(["I1", "I2", "O1", "O2"], [n, n, n, n]).zero()
    block = result.block[{}]
    name = os.environ["Hamiltonian"]
    if name == "Ising":
        "g Sz + SxSx"
        g = float(os.environ["IsingG"])
        block[0, 0, 0, 0] = g / 2.
        block[0, 1, 0, 1] = g / 2.
        block[1, 0, 1, 0] = -g / 2.
        block[1, 1, 1, 1] = -g / 2.
        block[1, 1, 0, 0] = 1 / 4.
        block[1, 0, 0, 1] = 1 / 4.
        block[0, 1, 1, 0] = 1 / 4.
        block[0, 0, 1, 1] = 1 / 4.
    elif name == "Heisenberg":
        block[0, 0, 0, 0] = 1 / 4.
        block[0, 1, 0, 1] = -1 / 4.
        block[1, 0, 1, 0] = -1 / 4.
        block[1, 1, 1, 1] = 1 / 4.
        block[1, 0, 0, 1] = 2 / 4.
        block[0, 1, 1, 0] = 2 / 4.
    elif name == "XY":
        block[1, 0, 0, 1] = 2 / 4.
        block[0, 1, 1, 0] = 2 / 4.
    else:
        raise RuntimeError("Unknown Hamiltonian")
    J = 1
    if "J" in os.environ:
        J = float(os.environ["J"])
    return result * J


class Site:
    delta = 1e-6

    def __init__(self, n, r, omega):
        self.n = n

        self.r = tools.LazyRoot(r)
        self.omega = tools.LazyRoot(omega)

        self.U = tools.LazyHandle(get_U, n, self.r, self.omega)
        self.d_r = tools.LazyHandle(self.calc_d_r, self.U, self.n, self.r,
                                    self.omega)
        self.d_omega = tools.LazyHandle(self.calc_d_omega, self.U, self.n,
                                        self.r, self.omega)

    @staticmethod
    def calc_d_r(U, n, r, omega):
        return (get_U(n, r + Site.delta, omega) - U) / Site.delta

    @staticmethod
    def calc_d_omega(U, n, r, omega):
        return (get_U(n, r, Site.delta + omega) - U) / Site.delta


class MPS:
    def __init__(self, depth, length, cutoff, dcutoff):
        self.depth = depth
        self.length = length
        self.cutoff = cutoff
        self.dcutoff = dcutoff

        self.hamiltonian = get_H()

        self.sites = [[
            Site(self.cutoff,
                 random.random() * 4 - 2,
                 random.random() * 12 - 6) for j in range(self.length)
        ] for i in range(self.depth)]

        self.construct_column()
        self.construct_env()
        self.construct_energy_and_gradient()

    def construct_column(self):
        self.two_layer_column = []
        for column in range(self.length):
            projector = Tensor(["U", "D"], [2, self.cutoff]).zero()
            projector[{"U": 0, "D": 0}] = 1
            projector[{"U": 1, "D": 1}] = 1

            def get_column(op, *tensors):
                result = [i for i in tensors]
                depth = len(result)
                result[0] = result[0].shrink({"D": 0})
                result[depth - 1] = result[depth - 1].contract(
                    projector, {("U", "D")})
                for i in reversed(range(depth)):
                    result.append(result[i].edge_rename({"U": "D", "D": "U"}))
                if op == "L":
                    for i in range(depth * 2):
                        result[i] = result[i].shrink({"L": 0})
                elif op == "R":
                    for i in range(depth * 2):
                        result[i] = result[i].shrink({"R": 0})
                return result

            op = ""
            if column == 0:
                op = "L"
            elif column == self.length - 1:
                op = "R"
            self.two_layer_column.append(
                tools.LazyHandle(
                    get_column, op,
                    *[self.sites[i][column].U for i in range(self.depth)]))

    def construct_env(self):
        self.env = {}
        self.env["L->R"] = {}
        self.env["R->L"] = {}

        self.env["L->R"][-1] = tools.LazyRoot(
            [Tensor(1) for _ in range(self.depth * 2)])
        self.env["R->L"][self.length] = tools.LazyRoot(
            [Tensor(1) for _ in range(self.depth * 2)])

        for i in range(self.length):
            self.env["L->R"][i] = tools.LazyHandle(MPS.two_to_one,
                                                   self.env["L->R"][i - 1],
                                                   self.two_layer_column[i],
                                                   self.dcutoff, "R", "L",
                                                   i == 0)
        for i in reversed(range(self.length)):
            self.env["R->L"][i] = tools.LazyHandle(MPS.two_to_one,
                                                   self.env["R->L"][i + 1],
                                                   self.two_layer_column[i],
                                                   self.dcutoff, "L", "R",
                                                   i == self.length - 1)

    @staticmethod
    def _cut_line(chain, l, r, cut):
        chain = chain[:]
        size = len(chain)

        for i in range(size - 2):
            Q, R = chain[i].qr('r', {r}, r, l)
            chain[i] = Q
            chain[i + 1] = chain[i + 1].contract(R, {(l, r)})

        for i in reversed(range(size - 1)):
            name_l = {j for j in chain[i].name if j != r}
            name_r = {j for j in chain[i + 1].name if j != l}
            map_l_1 = {j: "L--" + str(j) for j in name_l}
            map_r_1 = {j: "R--" + str(j) for j in name_r}
            map_l_2 = {"L--" + str(j): j for j in name_l}
            map_r_2 = {"R--" + str(j): j for j in name_r}

            tensor_l = chain[i]
            tensor_r = chain[i + 1]

            big = tensor_l.edge_rename(map_l_1).contract(
                tensor_r.edge_rename(map_r_1), {(r, l)})
            u, s, v = big.svd({"L--" + str(j) for j in name_l}, r, l, cut)
            s /= s.norm_max()
            chain[i + 1] = v.edge_rename(map_r_2)
            chain[i] = u.edge_rename(map_l_2).multiple(s, r, 'u')

        return chain

    @staticmethod
    def two_to_one(As, Bs, cut, An, Bn, no_contract):
        length = len(As)

        def merge_map(i):
            result = {"U": ["AU", "BU"], "D": ["AD", "BD"]}
            if i == 0:
                del result["D"]
            if i == length - 1:
                del result["U"]
            return result

        Cs = [
            As[i].edge_rename({
                "U": "AU",
                "D": "AD"
            }).contract(Bs[i].edge_rename({
                "U": "BU",
                "D": "BD"
            }),
                        set() if no_contract else {(An, Bn)}).merge_edge(
                            merge_map(i)) for i in range(length)
        ]
        return MPS._cut_line(Cs, "D", "U", cut)

    def construct_energy(self):
        pass


mps = MPS(5, 4, 3, 6)
[print(i) for i in mps.env["R->L"][0].value]


class IMPS_Handle:
    def __init__(self, imps):
        self.imps = imps
        self._energy = None

    def set_value(self, xs):
        self._energy = None
        imps = self.imps
        index = 0
        for i in range(imps.depth):
            imps.parameter[i][0] = xs[index]
            index += 1
            imps.parameter[i][1] = xs[index]
            index += 1
        return self

    def get_value(self):
        imps = self.imps
        xs = []
        for i in range(imps.depth):
            xs.append(imps.parameter[i][0])
            xs.append(imps.parameter[i][1])
        return xs

    def energy(self, suprress_print=False):
        if self._energy:
            return self._energy
        imps = self.imps
        result = imps.energy(cutoff_amp=self.cutoff_amp)
        for r, omega in imps.parameter:
            loss = (abs(r) - 2) * 0.1
            result += max(0, loss)

        self._energy = result
        if not suprress_print:
            print(result)
        return result

    def gradient(self):
        delta = 0.0001
        E = self.energy(True)
        xs = self.get_value()
        gradient = []
        for i in range(self.imps.depth * 2):
            xss = xs[:]
            xss[i] += delta
            new_E = self.set_value(xss).energy(True)
            gradient.append((new_E - E) / delta)
        self.set_value(xs)
        return gradient

    @staticmethod
    def read_from_file(file_name):
        with open(file_name, "r") as file:
            config = [i for i in file.read().split()]
            imps = IMPS(depth=int(config[0]), cutoff=int(config[1]))

            result = IMPS_Handle(imps)
            result.cutoff_amp = int(config[2])
            result.env_iter = int(config[3])
            if len(config) > 4:
                result.set_value([float(i) for i in config[4:]])
                print("READ")
            else:
                print("RANDOM")
            return result

    def save_to_file(self, file_name):
        with open(file_name, "w") as file:
            print(self.imps.depth,
                  self.imps.cutoff,
                  self.cutoff_amp,
                  self.imps.length,
                  file=file)
            print(*self.get_value(), file=file)
            print(self.energy(True), file=file)


#handle = IMPS_Handle.read_from_file(sys.argv[1])

#import opt_tools

#getattr(opt_tools, sys.argv[2])(handle, sys.argv[1], sys.argv[3:])
