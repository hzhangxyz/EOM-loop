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

import random
import TAT
import tools

Tensor = TAT(complex)

def get_U(n, r, omega):
    return tools.tensor_U(n, n, r, omega, 0, 0).edge_rename({
        "I1": "L",
        "O1": "R",
        "O2": "U",
        "I2": "D"
    })

def get_H(n):
    result = Tensor(["I1", "I2", "O1", "O2"], [n, n, n, n]).zero()
    block = result.block[{}]
    block[0, 0, 0, 0] = 2
    block[0, 1, 0, 1] = 2
    block[1, 0, 1, 0] = -2
    block[1, 1, 1, 1] = -2
    block[1, 1, 0, 0] = 1
    block[1, 0, 0, 1] = 1
    block[0, 1, 1, 0] = 1
    block[0, 0, 1, 1] = 1
    return result

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
        return (get_U(n, r, omega + Site.delta) - U) / Site.delta

class IMPS():

    def __init__(self, depth, cutoff):
        self.depth = depth
        self.cutoff = cutoff
        self.site = [Site(cutoff, random.random(), random.random()) for j in range(depth)]

        self.env_half = tools.LazyHandle(lambda U: U.shrink({"D": 0}).edge_rename(IMPS._rename_map(-depth)), self.site[0].U)
        for i in range(1, depth):
            self.env_half = tools.LazyHandle(lambda E, U, index: E.contract(U.edge_rename(IMPS._rename_map(index)), {("U", "D")}), self.env_half, self.site[i].U, i-depth)

        half_env_map = {"L"+IMPS._number_to_string(-i):"L"+IMPS._number_to_string(i) for i in range(1, 1+depth)} | {"R"+IMPS._number_to_string(-i):"R"+IMPS._number_to_string(i) for i in range(1, 1+depth)}
        another_half_env = tools.LazyHandle(lambda h: h.edge_rename(half_env_map), self.env_half)
        self.env = tools.LazyHandle(lambda a, b: a.contract(b, {("U", "U")}), self.env_half, another_half_env)

        env_pair = {("L"+IMPS._number_to_string(i), "R"+IMPS._number_to_string(i)) for i in range(-depth, 1+depth) if i != 0}
        big_env_list = [self.env]

        for i in range(10):
            last = big_env_list[-1]
            next = tools.LazyHandle(lambda a, b: a.contract(b, env_pair), last, last)
            next = tools.LazyHandle(lambda a: a/a.norm_max(), next)
            big_env_list.append(next)

        self.real_env = big_env_list[-1]

        self.hamiltonian = get_H(cutoff)

        mid_I = tools.LazyHandle(lambda a, b: a.contract(b, env_pair), self.env, self.env)

        half_env_pair = {("L"+IMPS._number_to_string(i), "R"+IMPS._number_to_string(i)) for i in range(-depth, 0) if i != 0}
        two_half_env = tools.LazyHandle(lambda a: a.edge_rename({"U":"P1"}).contract(a.edge_rename({"U":"P2"}), half_env_pair), self.env_half)
        another_two_half_env = tools.LazyHandle(lambda h: h.edge_rename(half_env_map), two_half_env)
        mid_H = tools.LazyHandle(lambda a, H, b: a.contract(H,{("P1", "I1"),("P2","I2")}).contract(b, {("O1","P1"),("O2","P2")}), two_half_env, self.hamiltonian, another_two_half_env)

        shrink_left_map = {"L"+IMPS._number_to_string(i): 0 for i in range(-depth, 1+depth) if i != 0}
        shrink_right_map = {"R"+IMPS._number_to_string(i): 0 for i in range(-depth, 1+depth) if i != 0}

        def contract_and_shrink(r, m ,l):
            return l.shrink(shrink_right_map).contract(m, env_pair).contract(r.shrink(shrink_left_map), env_pair)

        self.total_I = tools.LazyHandle(contract_and_shrink, self.real_env, mid_I, self.real_env)
        self.total_H = tools.LazyHandle(contract_and_shrink, self.real_env, mid_H, self.real_env)

        self.energy_handle = tools.LazyHandle(lambda h, i: (complex(h)/complex(i)).real, self.total_H, self.total_I)

    @staticmethod
    def _number_to_string(number):
        if number > 0:
            string = "+" + str(number)
        else:
            string = str(number)
        return string

    @staticmethod
    def _rename_map(number):
        string = IMPS._number_to_string(number)
        result = {}
        result["L"] = "L" + string
        result["R"] = "R" + string
        return result

    def get_value(self):
        xs = []
        for i in self.site:
            xs.append(i.r.value)
            xs.append(i.omega.value)
        return xs

    def set_value(self, xs):
        index = 0
        for i in self.site:
            i.r.reset(xs[index])
            index += 1
            i.omega.reset(xs[index])
            index += 1
        return self

    def energy(self):
        xs = self.get_value()
        loss = 0
        for i in range(self.depth):
            r = abs(xs[i*2])
            if r > 2:
                loss += (r-2)*0.1
        return self.energy_handle.value + loss

    def gradient(self):
        delta = 0.001
        E = self.energy()
        xs = self.get_value()
        gradient = []
        for i in range(2*self.depth):
            xss = xs[:]
            xss[i] += delta
            new_E = self.set_value(xss).energy()
            gradient.append((new_E - E)/delta)
        self.set_value(xs)
        return gradient

import sys

with open(sys.argv[1], "r") as file:
    config = [i for i in file.read().split()]

    imps = IMPS(depth=int(config[0]), cutoff=2)

    if int(config[1]) != 0:
        imps.set_value([float(i) for i in config[2:]])
        print("READ:", imps.energy())
    else:
        print("NOT READ")
    import opt_tools
    getattr(opt_tools, sys.argv[2])(imps, sys.argv[1], sys.argv[3:])
