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


from hamiltonian import get_H


class IMPS:
    def __init__(self, depth, cutoff):
        self.depth = depth
        self.cutoff = cutoff

        self.hamiltonian = get_H()

        self.projector_parameter_1 = random.random() * 2 - 1
        self.projector_parameter_2 = random.random() * 2 - 1

        self.parameter = [[random.random() * 4 - 2,
                           random.random() * 12 - 6]
                          for _ in range(self.depth)]

    def energy(self, cutoff_amp=2, env_iter=100):
        chain = self._get_chain()
        if cutoff_amp > 0:
            environment_cutoff = int(cutoff_amp * self.cutoff)
        else:
            environment_cutoff = abs(cutoff_amp)
        left = IMPS._construct_environment(chain, "L", "R", environment_cutoff,
                                           env_iter)
        right = IMPS._construct_environment(chain, "R", "L",
                                            environment_cutoff, env_iter)

        down_list = [[] for _ in range(self.depth)]
        for i in range(self.depth):
            j = i
            down_list[i].append(left[j])
            down_list[i].append(chain[j])
            down_list[i].append(chain[j])
            down_list[i].append(right[j])
        down = IMPS._contract_lattice_4(down_list, "U", "D")

        up_list = [[] for _ in range(self.depth)]
        for i in range(self.depth):
            j = 2 * self.depth - i - 1
            up_list[i].append(left[j])
            up_list[i].append(chain[j])
            up_list[i].append(chain[j])
            up_list[i].append(right[j])
        up = IMPS._contract_lattice_4(up_list, "D", "U")

        psipsi = up.contract_all_edge(down)
        Hpsi = up.contract(self.hamiltonian, {("P--1", "I1"),
                                              ("P--2", "I2")}).edge_rename({
                                                  "O1":
                                                  "P--1",
                                                  "O2":
                                                  "P--2"
                                              })
        psiHpsi = Hpsi.contract_all_edge(down)

        result = float(psiHpsi) / float(psipsi)
        return result

    @staticmethod
    def _contract_lattice_4(lattice, u, d):
        depth = len(lattice)
        for i in range(depth):
            for j in range(4):
                u_j = "P" + "--" + str(j)
                if i == 0 and j == 0:
                    result = lattice[0][0].edge_rename({u: u_j})
                else:
                    site = lattice[i][j].edge_rename({u: u_j})
                    set_c = set()
                    if j != 0:
                        set_c.add(("R", "L"))
                    if i != 0:
                        set_c.add((u_j, d))
                    result = result.contract(site, set_c)
        return result

    @staticmethod
    def _construct_environment(chain, l, r, cut, env_iter):
        size = len(chain)
        result = chain[:]
        for i in range(size):
            result[i] = result[i].shrink({l: 0})
        for t in range(env_iter):
            for i in range(size):
                if i == 0:
                    map_r = {"U": "R--U"}
                    map_c = {"U": "C--U"}
                    map_m = {"U": ["R--U", "C--U"]}
                elif i == size - 1:
                    map_r = {"D": "R--D"}
                    map_c = {"D": "C--D"}
                    map_m = {"D": ["R--D", "C--D"]}
                else:
                    map_r = {"U": "R--U", "D": "R--D"}
                    map_c = {"U": "C--U", "D": "C--D"}
                    map_m = {"U": ["R--U", "C--U"], "D": ["R--D", "C--D"]}
                R = result[i].edge_rename(map_r)
                C = chain[i].edge_rename(map_c)
                result[i] = R.contract(C, {(r, l)}).merge_edge(map_m)

            result = IMPS._cut_line(result, "D", "U", cut)

        return result

    def _get_chain(self):
        projector = Tensor(["I", "O", "PL", "PR"],
                           [self.cutoff, 2, 2, 2]).zero()
        projector[{
            "I": 0,
            "O": 0,
            "PL": 0,
            "PR": 1
        }] = self.projector_parameter_1
        projector[{"I": 1, "O": 1, "PL": 0, "PR": 1}] = 1
        projector[{
            "I": 0,
            "O": 0,
            "PL": 1,
            "PR": 0
        }] = self.projector_parameter_2
        projector[{"I": 1, "O": 1, "PL": 1, "PR": 0}] = 1
        site = [
            get_U(self.cutoff, self.parameter[i][0], self.parameter[i][1])
            for i in range(self.depth)
        ]
        chain = []
        for i in range(self.depth):
            this_site = site[i]
            if i == self.depth - 1:
                this_site = this_site.contract(
                    projector, {("U", "I")}).edge_rename({"O": "U"})
                this_site = this_site.merge_edge({"L": ["PL", "L"], "R": ["PR", "R"]})
            if i == 0:
                this_site = this_site.shrink({"D": 0})
            chain.append(this_site)
        for i in reversed(range(self.depth)):
            chain.append(chain[i].edge_rename({"D": "U", "U": "D"}))
        return chain

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


class IMPS_Handle:
    def __init__(self, imps):
        self.imps = imps
        self._energy = None

    def set_value(self, xs):
        self._energy = None
        imps = self.imps
        imps.projector_parameter_1 = xs[0]
        imps.projector_parameter_2 = xs[1]
        index = 2
        for i in range(imps.depth):
            imps.parameter[i][0] = xs[index]
            index += 1
            imps.parameter[i][1] = xs[index]
            index += 1
        return self

    def get_value(self):
        imps = self.imps
        xs = [imps.projector_parameter_1, imps.projector_parameter_2]
        for i in range(imps.depth):
            xs.append(imps.parameter[i][0])
            xs.append(imps.parameter[i][1])
        return xs

    def energy(self):
        if self._energy:
            return self._energy
        imps = self.imps
        result = imps.energy(cutoff_amp=self.cutoff_amp,
                             env_iter=self.env_iter)
        for r, omega in imps.parameter:
            loss = (abs(r) - 2) * 0.1
            result += max(0, loss)

        loss = (abs(imps.projector_parameter_1) - 1) * 0.1
        result += max(0, loss)
        loss = (abs(imps.projector_parameter_2) - 1) * 0.1
        result += max(0, loss)

        self._energy = result
        return result

    def gradient(self):
        delta = 0.0001
        E = self.energy()
        xs = self.get_value()
        gradient = []
        for i in range(self.imps.depth * 2 + 2):
            xss = xs[:]
            xss[i] += delta
            new_E = self.set_value(xss).energy()
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
                  self.env_iter,
                  file=file)
            print(*self.get_value(), file=file)
            print(self.energy(), file=file)


#imps = IMPS(depth=2, cutoff=3)
#imps.energy(cutoff_amp=2, env_iter=10)

handle = IMPS_Handle.read_from_file(sys.argv[1])

import opt_tools

getattr(opt_tools, sys.argv[2])(handle, sys.argv[1], sys.argv[3:])
