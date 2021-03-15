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
    block[1, 0, 0, 1] = 2
    block[0, 1, 1, 0] = 2
    return result


class Site:
    delta = 1e-6

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
        self.projector = Tensor(["I", "O"], [cutoff, cutoff]).zero()
        self.projector[{"I": 0, "O": 0}] = 1
        self.projector[{"I": 1, "O": 1}] = 1

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
                return tools.LazyHandle(L_ops.project, former, self.projector)
            if index_length == index_hamiltonian - 1:
                return tools.LazyHandle(L_ops.contract_hamiltonian, former,
                                        self.hamiltonian)
            elif index_length == index_hamiltonian:
                return tools.LazyHandle(L_ops.trace_hamiltonian, former)
            else:
                return tools.LazyHandle(L_ops.project, former, self.projector)
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
                        if index_depth == -1 and index_length == self.length - 1:
                            return tools.LazyHandle(
                                L_ops.down_part_tail_depth_1,
                                former,
                                site,
                                r_name=r_name,
                                parity=self.get_half_parity())
                        else:
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
                        if index_depth == +1 and index_length == self.length - 1:
                            return tools.LazyHandle(
                                L_ops.up_part_tail_depth_1,
                                former,
                                site,
                                r_name=r_name,
                                parity=self.get_half_parity())
                        else:
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
                return tools.LazyHandle(R_ops.project, former, self.projector)
            if index_length == index_hamiltonian:
                return tools.LazyHandle(R_ops.hole_hamiltonian, former,
                                        self.cutoff)
            elif index_length == index_hamiltonian - 1:
                return tools.LazyHandle(R_ops.contract_hamiltonian, former,
                                        self.hamiltonian)
            else:
                return tools.LazyHandle(R_ops.project, former, self.projector)
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
                    if index_depth == +self.depth == +1:
                        return tools.LazyHandle(R_ops.right_up_corner_depth_1,
                                                site,
                                                l_name=l_name,
                                                r_name=r_name,
                                                parity=self.get_half_parity())
                    elif index_depth == +self.depth:
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
                    if index_depth == -self.depth == -1:
                        return tools.LazyHandle(
                            R_ops.right_down_corner_depth_1,
                            former,
                            site,
                            l_name=l_name,
                            r_name=r_name,
                            parity=self.get_half_parity())
                    elif index_depth == -self.depth:
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

    def energy(self):
        psiHpsi = sum(self.get_psiHpsi(i) for i in range(1, self.length))
        psipsi = self.get_psiHpsi(0)
        result = (complex(psiHpsi) / complex(psipsi)).real / self.length
        """
        print("E",
              complex(psiHpsi).real / self.length,
              complex(psipsi).real, result)
        """
        return result

    def parameter_gradient(self, i, j, gradient):
        site = self.get_site(length=i, depth=j)
        expand_map = {}
        if j == 0:
            expand_map["D"] = (0, self.cutoff)
        if i == 0 and j == 0:
            expand_map["L"] = (0, self.cutoff)
        if i == self.length - 1 and j == self.depth - 1:
            expand_map["R"] = (self.get_half_parity(), self.cutoff)
        if expand_map:
            gradient = gradient.expand(expand_map)
        names = {(i, i) for i in "UDLR"}
        return (complex(site.d_r.value.contract(gradient, names)).real,
                complex(site.d_omega.value.contract(gradient, names)).real)

    def psipsi_gradient(self):
        gs = []
        for i in range(self.length):
            for j in range(self.depth):
                p_r, p_omega = self.parameter_gradient(
                    i, j, self.hole_at(0, i, j - self.depth))
                gs.append(p_r)
                gs.append(p_omega)
        return gs

    def get_psi(self):
        result = Tensor(1)
        for i in range(self.length):
            for j in range(self.depth):
                site = self.get_site(length=i, depth=j).U.value
                if j == 0:
                    site = site.shrink({"D": 0})
                if i == 0 and j == 0:
                    site = site.shrink({"L": 0})
                if i == self.length - 1 and j == self.depth - 1:
                    site = site.shrink({"R": self.get_half_parity()})
                result = result.contract(site.edge_rename({"R": "R" + str(j)}),
                                         {("U", "D"), ("R" + str(j), "L")})
                result = result.edge_rename({
                    "L": "L" + str(j),
                    "R": "R" + str(j)
                })
            result = result.edge_rename({"U": "P" + str(i)})
        result = result.trace({("L" + str(j + 1), "R" + str(j))
                               for j in range(self.depth)})
        return result

    def gradient(self):
        gs = []
        for i in range(self.length):
            for j in range(self.depth):
                p_r, p_omega = self.parameter_gradient(i, j,
                                                       self.get_gradient(i, j))
                gs.append(p_r)
                gs.append(p_omega)
        return gs

    def get_value(self):
        xs = []
        for i in range(self.length):
            for j in range(self.depth):
                site = self.get_site(length=i, depth=j)
                xs.append(site.r.value)
                xs.append(site.omega.value)
        return xs

    @staticmethod
    def check_r(r):
        if r > +2:
            return +2
        elif r < -2:
            return -2
        else:
            return r

    def set_value(self, xs):
        index = 0
        for i in range(self.length):
            for j in range(self.depth):
                site = self.get_site(length=i, depth=j)
                site.r.reset(self.check_r(xs[index]))
                index += 1
                site.omega.reset(xs[index])
                index += 1
        return self


# main
def main():
    import sys

    with open(sys.argv[1], "r") as file:
        config = [i for i in file.read().split()]

    chain = Chain(cutoff=2, length=int(config[0]), depth=int(config[1]))

    if int(config[2]) != 0:
        chain.set_value([float(i) for i in config[3:]])
        print("READ:", chain.energy())
    else:
        print("NOT READ")

    import opt_tools

    getattr(opt_tools, sys.argv[2])(chain, sys.argv[1], sys.argv[3:])


if __name__ == "__main__":
    main()
