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
from matrix_U import matrix_U

Tensor = TAT(complex)


def get_U(n, r, omega, phi, psi):
    result = Tensor(["I1", "I2", "O1", "O2"], [n, n, n, n])
    result.block[{}] = matrix_U(n, n, r, omega, phi, psi)
    return result.edge_rename({"I1": "L", "O1": "R", "O2": "U", "I2": "D"})


def get_H(n):
    result = Tensor(["I1", "I2", "O1", "O2"], [n, n, n, n]).zero()
    block = result.block[{}]
    block[0, 0, 0, 0] = 1
    block[0, 1, 0, 1] = -1
    block[1, 0, 1, 0] = -1
    block[1, 1, 1, 1] = 1
    # block[1, 0, 0, 1] = 2
    # block[0, 1, 1, 0] = 2
    return result


class Site:
    def __init__(self, n, r, omega, phi, psi):
        self.n = n
        self._value = None
        self._r = r
        self._omega = omega
        self._phi = phi
        self._psi = psi

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, value):
        self._r = value
        self._value = None

    @property
    def omega(self):
        return self._omega

    @omega.setter
    def omega(self, value):
        self._omega = value
        self._value = None

    @property
    def phi(self):
        return self._phi

    @phi.setter
    def phi(self, value):
        self._phi = value
        self._value = None

    @property
    def psi(self):
        return self._psi

    @psi.setter
    def psi(self, value):
        self._psi = value
        self._value = None

    @property
    def value(self):
        if self._value is None:
            self._value = get_U(self.n, self._r, self._omega, self._phi,
                                self._psi)
        return self._value


class Chain:
    def __init__(self, n, l, d=1):
        self.n = n
        self.l = l
        self.d = d
        self.sites = [[Site(n, 0, 0, 0, 0) for j in range(l)]
                      for i in range(d)]
        self.hamiltonian = get_H(n)

    def get_mps(self, depth=None):
        if depth is None:
            depth = self.d - 1
        if depth == 0:
            result = []
            for i in range(self.l):
                this = self.sites[0][i].value
                if i == 0:
                    this = this.shrink({"L": 0, "D": 0})
                else:
                    this = this.shrink({"D": 0})
                result.append(this)
            result.append(
                Tensor(["L", "U"], [self.n, self.n]).identity({("L", "U")}))
            return result
        else:
            last_layer = self.get_mps(depth - 1)
            result = []
            for i in range(self.l):
                if i == 0:
                    this = last_layer[0].edge_rename({
                        "U": "U0"
                    }).contract(
                        last_layer[1].edge_rename({
                            "U": "U1",
                            "R": "R0"
                        }), {("R", "L")}).contract(
                            self.sites[depth][0].value.edge_rename({"R":
                                                                    "R1"}),
                            {("U0", "L"),
                             ("U1", "D")}).merge_edge({"R": ["R0", "R1"]})
                elif i == self.l - 1:
                    this = self.sites[depth][i].value.merge_edge(
                        {"L": ["D", "L"]})
                else:
                    this = last_layer[i + 1].edge_rename({
                        "L": "L0",
                        "R": "R0"
                    }).contract(
                        self.sites[depth][i].value.edge_rename({
                            "L": "L1",
                            "R": "R1"
                        }), {("U", "D")}).merge_edge({
                            "L": ["L0", "L1"],
                            "R": ["R0", "R1"]
                        })
                result.append(this)
            result.append(
                Tensor(["L", "U"], [self.n, self.n]).identity({("L", "U")}))
            return result

    def energy(self):
        mps = self.get_mps()
        result = 0

        # 辅助矩阵
        left = {}
        right = {}
        left[0] = mps[0].edge_rename({
            "R": "R1"
        }).contract(mps[0].edge_rename({"R": "R2"}), {("U", "U")})
        for i in range(1, self.l):
            left[i] = left[i - 1].contract(mps[i].edge_rename({"R": "R1"}),
                                           {("R1", "L")}).contract(
                                               mps[i].edge_rename({"R": "R2"}),
                                               {("R2", "L"), ("U", "U")})
        # print(left)
        right[self.l] = mps[self.l].edge_rename({
            "L": "L1"
        }).contract(mps[self.l].edge_rename({"L": "L2"}), {("U", "U")})
        for i in range(self.l - 1, 1, -1):
            right[i] = right[i + 1].contract(
                mps[i].edge_rename({"L": "L1"}),
                {("L1", "R")}).contract(mps[i].edge_rename({"L": "L2"}),
                                        {("L2", "R"), ("U", "U")})
        # print(right)

        # 计算能量
        for i in range(self.l):
            if i == 0:
                result = result + mps[i].edge_rename({
                    "U": "U1",
                    "R": "R1"
                }).contract(mps[i + 1].edge_rename({
                    "U": "U2",
                    "R": "R1"
                }), {("R1", "L")}).contract(
                    self.hamiltonian.edge_rename({
                        "O1": "U1",
                        "O2": "U2"
                    }), {("U1", "I1"), ("U2", "I2")}).contract(
                        mps[i].edge_rename({"R": "R2"}), {
                            ("U1", "U")
                        }).contract(mps[i + 1].edge_rename({"R": "R2"}), {
                            ("U2", "U"), ("R2", "L")
                        }).contract(right[i + 2], {("R1", "L1"), ("R2", "L2")})
            elif i == self.l - 1:
                result = result + left[i - 1].contract(
                    mps[i].edge_rename({
                        "U": "U1",
                        "R": "R1"
                    }), {("R1", "L")}).contract(
                        mps[i + 1].edge_rename({"U": "U2"}),
                        {("R1", "L")}).contract(
                            self.hamiltonian.edge_rename({
                                "O1": "U1",
                                "O2": "U2"
                            }), {("U1", "I1"), ("U2", "I2")}).contract(
                                mps[i].edge_rename({"R": "R2"}), {
                                    ("U1", "U"), ("R2", "L")
                                }).contract(mps[i + 1], {("U2", "U"),
                                                         ("R2", "L")})
            else:
                result = result + left[i - 1].contract(
                    mps[i].edge_rename({
                        "U": "U1",
                        "R": "R1"
                    }), {
                        ("R1", "L")
                    }).contract(mps[i + 1].edge_rename({
                        "U": "U2",
                        "R": "R1"
                    }), {("R1", "L")}).contract(
                        self.hamiltonian.edge_rename({
                            "O1": "U1",
                            "O2": "U2"
                        }), {("U1", "I1"), ("U2", "I2")}).contract(
                            mps[i].edge_rename({"R": "R2"}), {
                                ("U1", "U"), ("R2", "L")
                            }).contract(mps[i + 1].edge_rename({"R": "R2"}), {
                                ("U2", "U"), ("R2", "L")
                            }).contract(right[i + 2], {("R1", "L1"),
                                                       ("R2", "L2")})
        return complex(
            result /
            left[self.l - 1].contract(right[self.l], {("R1", "L1"),
                                                      ("R2", "L2")})).real

    def opt(self, delta):
        energy = self.energy()
        # r
        for i in range(self.d):
            for j in range(self.l):
                while True:
                    self.sites[i][j].r = self.sites[i][j].r + delta
                    energy_h = self.energy()
                    self.sites[i][j].r = self.sites[i][j].r - 2 * delta
                    energy_l = self.energy()
                    self.sites[i][j].r = self.sites[i][j].r + delta
                    if energy_h < energy:
                        self.sites[i][j].r = self.sites[i][j].r + delta
                        energy = energy_h
                    elif energy_l < energy:
                        self.sites[i][j].r = self.sites[i][j].r - delta
                        energy = energy_l
                    else:
                        break
        # omega
        for i in range(self.d):
            for j in range(self.l):
                while True:
                    self.sites[i][j].omega = self.sites[i][j].omega + delta
                    energy_h = self.energy()
                    self.sites[i][j].omega = self.sites[i][j].omega - 2 * delta
                    energy_l = self.energy()
                    self.sites[i][j].omega = self.sites[i][j].omega + delta
                    if energy_h < energy:
                        self.sites[i][j].omega = self.sites[i][j].omega + delta
                        energy = energy_h
                    elif energy_l < energy:
                        self.sites[i][j].omega = self.sites[i][j].omega - delta
                        energy = energy_l
                    else:
                        break
        return energy


import sys
d = int(sys.argv[1])
l = int(sys.argv[2]) - 1
c = Chain(n=2, l=l, d=d)
print(c.energy())
for t in range(10):
    print(c.opt(0.1) / (l + 1))
    """for i in c.sites:
        for j in i:
            print(j.r, j.omega)"""
for t in range(10):
    print(c.opt(0.01) / (l + 1))
    """for i in c.sites:
        for j in i:
            print(j.r, j.omega)"""
for t in range(10):
    print(c.opt(0.001) / (l + 1))
    """for i in c.sites:
        for j in i:
            print(j.r, j.omega)"""
