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
import numpy as np
import TAT

import L_ops
import R_ops
import tools

Tensor = TAT(complex)


def get_U(n, r, omega, phi, psi):
    return tools.tensor_U(n, n, r, omega, phi, psi).edge_rename({
        "I1": "L",
        "O1": "R",
        "O2": "U",
        "I2": "D"
    })


def get_H(n):
    """
    Ising model的哈密顿量
    """
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
    delta = 0.0001

    def __init__(self, n, r, omega, phi, psi):
        self.n = n

        self.r = tools.LazyRoot(r)
        self.omega = tools.LazyRoot(omega)
        self.phi = tools.LazyRoot(phi)
        self.psi = tools.LazyRoot(psi)

        self.U = tools.LazyHandle(get_U, n, self.r, self.omega, self.phi,
                                  self.psi)
        self.d_r = tools.LazyHandle(self.calc_d_r, self.U, self.n, self.r,
                                    self.omega, self.phi, self.psi)
        self.d_omega = tools.LazyHandle(self.calc_d_omega, self.U, self.n,
                                        self.r, self.omega, self.phi, self.psi)
        self.d_phi = tools.LazyHandle(self.calc_d_phi, self.U, self.n, self.r,
                                      self.omega, self.phi, self.psi)
        self.d_psi = tools.LazyHandle(self.calc_d_psi, self.U, self.n, self.r,
                                      self.omega, self.phi, self.psi)

    @staticmethod
    def calc_d_r(U, n, r, omega, phi, psi):
        return (get_U(n, r + Site.delta, omega, phi, psi) - U) / Site.delta

    @staticmethod
    def calc_d_omega(U, n, r, omega, phi, psi):
        return (get_U(n, r, omega + Site.delta, phi, psi) - U) / Site.delta

    @staticmethod
    def calc_d_phi(U, n, r, omega, phi, psi):
        return (get_U(n, r, omega, phi + Site.delta, psi) - U) / Site.delta

    @staticmethod
    def calc_d_psi(U, n, r, omega, phi, psi):
        return (get_U(n, r, omega, phi, psi + Site.delta) - U) / Site.delta


class Chain:
    def get_site(self, *, length, depth):
        return self.sites[length][depth]

    def __init__(self, cutoff, length, depth):
        self.cutoff = cutoff
        self.length = length
        self.depth = depth
        self.sites = [[
            Site(cutoff, random.random(), random.random(), 0, 0)
            for j in range(depth)
        ] for i in range(length)]
        self.hamiltonian = get_H(cutoff)

        self.aux = {}

        for index_hamiltonian in range(length):
            for index_length in range(length):
                for index_depth in range(-depth, 1 + depth):
                    self.create_aux("L", index_hamiltonian, index_length,
                                    index_depth)
            for index_length in reversed(range(length)):
                for index_depth in reversed(range(-depth, 1 + depth)):
                    self.create_aux("R", index_hamiltonian, index_length,
                                    index_depth)

    @staticmethod
    def get_name(direction, number):
        if number > 0:
            return direction + "+" + str(number)
        else:
            return direction + str(number)

    @staticmethod
    def get_pair_names(direction, number):
        if direction == "L":
            if number > 0:
                return Chain.get_name("L",
                                      number), Chain.get_name("R", number + 1)
            else:
                return Chain.get_name("L",
                                      number), Chain.get_name("R", number - 1)
        else:
            if number > 0:
                return Chain.get_name("L",
                                      number - 1), Chain.get_name("R", number)
            else:
                return Chain.get_name("L",
                                      number + 1), Chain.get_name("R", number)

    def hole_at(self, index_hamiltonian, index_length, index_depth):
        if index_depth == -self.depth:
            if index_length == 0:
                left = None
            else:
                left = self.aux["L", index_hamiltonian, index_length - 1,
                                +self.depth].value
        else:
            left = self.aux["L", index_hamiltonian, index_length,
                            index_depth - 1].value
        if index_depth == +self.depth:
            if index_length == self.length - 1:
                right = None
            else:
                right = self.aux["R", index_hamiltonian, index_length + 1,
                                 -self.depth].value
        else:
            right = self.aux["R", index_hamiltonian, index_length,
                             index_depth + 1].value
        out_names = {
            self.get_pair_names("L", i)
            for i in range(1 - self.depth, self.depth) if i != 0
        }
        in_names = {(self.get_name("R", i), self.get_name("L", i))
                    for i in range(-self.depth, self.depth + 1) if i != 0}
        if left == None:
            result = right.trace(out_names)
        elif right == None:
            result = left.trace(
                out_names - {(self.get_name("L", self.depth - 1),
                              self.get_name("R", self.depth))})
        else:
            in_names.remove(
                (self.get_name("R",
                               index_depth), self.get_name("L", index_depth)))
            trace_names = set()
            if index_length == 0:
                for i in range(index_depth, self.depth + 1):
                    if i != 0:
                        if i != self.depth:
                            # 最后一个地方没有out
                            out_names.remove(self.get_pair_names("L", i))
                        if i != self.depth and i != index_depth:
                            trace_names.add(self.get_pair_names("L", i))
                        if i != index_depth:
                            in_names.remove(
                                (self.get_name("R", i), self.get_name("L", i)))
            if index_length == self.length - 1:
                for i in range(-self.depth, index_depth + 1):
                    if i != 0:
                        if abs(i) != 1:
                            # 第一个地方没有out
                            out_names.remove(self.get_pair_names("R", i))
                        if abs(i) != 1 and i != index_depth:
                            trace_names.add(self.get_pair_names("R", i))
                        if i != index_depth:
                            in_names.remove(
                                (self.get_name("R", i), self.get_name("L", i)))
            if index_hamiltonian != 0 and (index_hamiltonian - 1, 0) < (
                    index_length, index_depth) < (index_hamiltonian, 0):
                hamiltonian_names = {("I2", "I2"), ("O2", "O2")}
            else:
                hamiltonian_names = set()
            result = left.contract(right,
                                   in_names | out_names | hamiltonian_names)
            if trace_names:
                result = result.trace(trace_names)
        rename_map = {"U": "D", "D": "U"}
        if index_length == 0:
            rename_map[self.get_pair_names("L", index_depth)[1]] = "L"
        else:
            rename_map[self.get_name("R", index_depth)] = "L"
        if index_length == self.length - 1:
            rename_map[self.get_pair_names("R", index_depth)[0]] = "R"
        else:
            rename_map[self.get_name("L", index_depth)] = "R"
        return result.edge_rename(rename_map)

    def create_aux(self, direction, index_hamiltonian, index_length,
                   index_depth):
        if direction == "L":
            self.aux["L", index_hamiltonian, index_length,
                     index_depth] = self.create_aux_L(index_hamiltonian,
                                                      index_length,
                                                      index_depth)
        else:
            self.aux["R", index_hamiltonian, index_length,
                     index_depth] = self.create_aux_R(index_hamiltonian,
                                                      index_length,
                                                      index_depth)

    def get_half_parity(self):
        return self.length & 3 == 2

    def create_aux_L(self, index_hamiltonian, index_length, index_depth):
        if index_hamiltonian != 0 and (index_length, index_depth) < (
                index_hamiltonian - 1, 0):
            # 加上hamiltonian的量，如果还没算到hamiltonian算符处，和没有hamiltonian一样
            return self.aux["L", 0, index_length, index_depth]
        if index_depth == 0:
            former = self.aux["L", index_hamiltonian, index_length, -1]
            if index_hamiltonian == 0:
                return former
            if index_length == index_hamiltonian - 1:
                return tools.LazyHandle(L_ops.contract_hamiltonian, former,
                                        self.hamiltonian)
            elif index_length == index_hamiltonian:
                return tools.LazyHandle(L_ops.trace_hamiltonian, former)
            else:
                return former
        else:
            if index_depth == -self.depth:
                if index_length == 0:
                    former = None
                else:
                    former = self.aux["L", index_hamiltonian, index_length - 1,
                                      +self.depth]
            else:
                former = self.aux["L", index_hamiltonian, index_length,
                                  index_depth - 1]
            site = self.get_site(length=index_length,
                                 depth=self.depth - abs(index_depth)).U
            r_name = self.get_name("R", index_depth)
            l_name = self.get_name("L", index_depth)
            if index_length == 0:
                if index_depth < 0:
                    if index_depth == -self.depth:
                        return tools.LazyHandle(L_ops.left_down_corner,
                                                site,
                                                r_name=r_name)
                    else:
                        return tools.LazyHandle(L_ops.left_edge_down_part,
                                                former,
                                                site,
                                                r_name=r_name,
                                                l_name=l_name)
                else:
                    if index_depth == +self.depth:
                        return tools.LazyHandle(L_ops.left_up_corner,
                                                former,
                                                site,
                                                r_name=r_name)
                    else:
                        return tools.LazyHandle(L_ops.left_edge_up_part,
                                                former,
                                                site,
                                                r_name=r_name,
                                                l_name=l_name)
            else:
                if index_depth < 0:
                    if index_depth == -self.depth:
                        return tools.LazyHandle(L_ops.down_edge,
                                                former,
                                                site,
                                                r_name=r_name)
                    else:
                        if index_depth == -1 and index_length == self.length - 1:
                            return tools.LazyHandle(
                                L_ops.down_part_tail,
                                former,
                                site,
                                r_name=r_name,
                                parity=self.get_half_parity())
                        else:
                            return tools.LazyHandle(L_ops.down_part,
                                                    former,
                                                    site,
                                                    r_name=r_name)
                else:
                    if index_depth == +self.depth:
                        return tools.LazyHandle(L_ops.up_edge,
                                                former,
                                                site,
                                                r_name=r_name)
                    else:
                        if index_depth == +1 and index_length == self.length - 1:
                            return tools.LazyHandle(
                                L_ops.up_part_tail,
                                former,
                                site,
                                r_name=r_name,
                                parity=self.get_half_parity())
                        else:
                            return tools.LazyHandle(L_ops.up_part,
                                                    former,
                                                    site,
                                                    r_name=r_name)

    def create_aux_R(self, index_hamiltonian, index_length, index_depth):
        if index_hamiltonian != 0 and (index_length,
                                       index_depth) > (index_hamiltonian, 0):
            # 加上hamiltonian的量，如果还没算到hamiltonian算符处，和没有hamiltonian一样
            return self.aux["R", 0, index_length, index_depth]
        if index_depth == 0:
            former = self.aux["R", index_hamiltonian, index_length, +1]
            if index_hamiltonian == 0:
                return former
            if index_length == index_hamiltonian:
                return tools.LazyHandle(R_ops.hole_hamiltonian, former,
                                        self.cutoff)
            elif index_length == index_hamiltonian - 1:
                return tools.LazyHandle(R_ops.contract_hamiltonian, former,
                                        self.hamiltonian)
            else:
                return former
        else:
            if index_depth == self.depth:
                if index_length == self.length - 1:
                    former = None
                else:
                    former = self.aux["R", index_hamiltonian, index_length + 1,
                                      -self.depth]
            else:
                former = self.aux["R", index_hamiltonian, index_length,
                                  index_depth + 1]
            site = self.get_site(length=index_length,
                                 depth=self.depth - abs(index_depth)).U
            r_name = self.get_name("R", index_depth)
            l_name = self.get_name("L", index_depth)
            if index_length == self.length - 1:
                if index_depth > 0:
                    if index_depth == +self.depth:
                        return tools.LazyHandle(R_ops.right_up_corner,
                                                site,
                                                l_name=l_name,
                                                r_name=r_name)
                    elif index_depth == +1:
                        return tools.LazyHandle(R_ops.right_edge_up_part_tail,
                                                former,
                                                site,
                                                l_name=l_name,
                                                r_name=r_name,
                                                parity=self.get_half_parity())
                    else:
                        return tools.LazyHandle(R_ops.right_edge_up_part,
                                                former,
                                                site,
                                                l_name=l_name,
                                                r_name=r_name)
                else:
                    if index_depth == -self.depth:
                        return tools.LazyHandle(R_ops.right_down_corner,
                                                former,
                                                site,
                                                l_name=l_name,
                                                r_name=r_name)
                    if index_depth == -1:
                        return tools.LazyHandle(
                            R_ops.right_edge_down_part_tail,
                            former,
                            site,
                            l_name=l_name,
                            r_name=r_name,
                            parity=self.get_half_parity())
                    else:
                        return tools.LazyHandle(R_ops.right_edge_down_part,
                                                former,
                                                site,
                                                l_name=l_name,
                                                r_name=r_name)
            else:
                if index_depth > 0:
                    if index_depth == +self.depth:
                        if index_length == 0:
                            return tools.LazyHandle(R_ops.left_up_corner,
                                                    former,
                                                    site,
                                                    l_name=l_name)
                        else:
                            return tools.LazyHandle(R_ops.up_edge,
                                                    former,
                                                    site,
                                                    l_name=l_name)
                    else:
                        return tools.LazyHandle(R_ops.up_part,
                                                former,
                                                site,
                                                l_name=l_name)
                else:
                    if index_depth == -self.depth:
                        if index_length == 0:
                            return tools.LazyHandle(R_ops.left_down_corner,
                                                    former,
                                                    site,
                                                    l_name=l_name)
                        else:
                            return tools.LazyHandle(R_ops.down_edge,
                                                    former,
                                                    site,
                                                    l_name=l_name)
                    else:
                        return tools.LazyHandle(R_ops.down_part,
                                                former,
                                                site,
                                                l_name=l_name)

    def get_psiHpsi(self, index_hamiltonian):
        return self.aux["L", index_hamiltonian, self.length - 1,
                        self.depth].value.trace({
                            self.get_pair_names("L", i)
                            for i in range(1 - self.depth, self.depth)
                            if i != 0
                        })

    def get_gradient(self, index_length, index_depth):
        partialpsiHpsi = sum(
            self.hole_at(i, index_length, index_depth - self.depth)
            for i in range(1, self.length))
        partialpsipsi = self.hole_at(0, index_length, index_depth - self.depth)
        psiHpsi = sum(self.get_psiHpsi(i) for i in range(1, self.length))
        psipsi = self.get_psiHpsi(0)
        term_1 = partialpsiHpsi * (2. / complex(psipsi))
        term_2 = partialpsipsi * (2 * complex(psiHpsi) / (complex(psipsi)**2))
        return term_1 - term_2

    def get_energy(self):
        psiHpsi = sum(self.get_psiHpsi(i) for i in range(1, self.length))
        psipsi = self.get_psiHpsi(0)
        return (complex(psiHpsi) / complex(psipsi)).real / self.length

    def gradient_descent_once(self, delta):
        gradients = [[self.get_gradient(i, j) for j in range(self.depth)]
                     for i in range(self.length)]
        for i in range(self.length):
            for j in range(self.depth):
                site = self.get_site(length=i, depth=j)
                gradient = gradients[i][j]
                expand_map = {}
                if j == 0:
                    expand_map["D"] = (0, 2)
                if i == 0 and j == 0:
                    expand_map["L"] = (0, 2)
                if i == self.length - 1 and j == self.depth - 1:
                    expand_map["R"] = (self.get_half_parity(), 2)
                if expand_map:
                    gradient = gradient.expand(expand_map)
                self.update_tensor(site, gradient, delta)

    @staticmethod
    def update_tensor(site, gradient, delta):
        if False:
            site.U.reset(site.U.value - gradient * delta)
        else:
            names = {(i, i) for i in "UDLR"}
            d_r = complex(site.d_r.value.contract(gradient, names)).real
            d_omega = complex(site.d_omega.value.contract(gradient,
                                                          names)).real
            site.r.reset(site.r.value - d_r * delta)
            site.omega.reset(site.omega.value - d_omega * delta)


def main():
    for l in range(3, 13):
        c = Chain(cutoff=2, length=l, depth=2)

        energy = c.get_energy()
        while True:
            c.gradient_descent_once(1)
            energy_new = c.get_energy()
            if abs(energy_new - energy) < 0.001:
                break
            energy = energy_new
        print(energy)


main()
