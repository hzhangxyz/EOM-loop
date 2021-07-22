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
import sys

Tensor = TAT(float)


def get_lattice_energy(length, depth, get_site, hamiltonian, h_index, cut1,
                       cut2):
    length_m = len(hamiltonian.name) // 2

    def get_left(*, length, depth):
        return get_site(depth=depth, length=h_index - length - 1)

    def get_middle(*, length, depth):
        return get_site(depth=depth, length=length + h_index)

    def get_right(*, length, depth):
        return get_site(depth=depth, length=length + h_index + length_m)

    return get_energy(h_index, length - h_index - length_m, depth, get_left,
                      get_right, get_middle, hamiltonian, cut1, cut2)


def get_energy(length_l, length_r, depth, get_left, get_right, get_middle,
               hamiltonian, cut1, cut2):
    left = contract_lattice_environment(length_l, depth, get_left, cut1, True)
    right = contract_lattice_environment(length_r, depth, get_right, cut1,
                                         False)

    length_m = len(hamiltonian.name) // 2

    for i in range(depth):
        this = []
        this.append(left[i])
        for j in range(length_m):
            this.append(get_middle(length=j, depth=i))
        this.append(right[i])
        if i == 0:
            down = this
        else:
            for j in range(length_m + 2):
                down[j] = merge_tensor(down[j], this[j], {"L", "R"},
                                       {("U", "D")})
            down = cut_line(down, "L", "R", cut2)

    for k in range(depth):
        i = 2 * depth - k - 1
        this = []
        this.append(left[i])
        for j in range(length_m):
            this.append(
                get_middle(length=j, depth=k).edge_rename({
                    "D": "U",
                    "U": "D"
                }))
        this.append(right[i])
        if k == 0:
            up = this
        else:
            for j in range(length_m + 2):
                up[j] = merge_tensor(up[j], this[j], {"L", "R"}, {("D", "U")})
            up = cut_line(up, "L", "R", cut2)

    upv = up[0].edge_rename({"D": "P0"})
    for i in range(1, length_m + 2):
        upv = upv.contract(up[i].edge_rename({"D": "P" + str(i)}),
                           {("R", "L")})
    downv = down[0].edge_rename({"U": "P0"})
    for i in range(1, length_m + 2):
        downv = downv.contract(down[i].edge_rename({"U": "P" + str(i)}),
                               {("R", "L")})

    all_pair = {("P" + str(i), "P" + str(i)) for i in range(length_m + 2)}
    psipsi = upv.contract(downv, all_pair)
    Hpsi = upv.contract(hamiltonian,
                        {("P" + str(i), "I" + str(i))
                         for i in range(1, length_m + 1)}).edge_rename({
                             "O" + str(i): "P" + str(i)
                             for i in range(1, length_m + 1)
                         })
    psiHpsi = Hpsi.contract(downv, all_pair)

    result = float(psiHpsi) / float(psipsi)
    return result


def get_chain(get_function, length_index, depth):
    chain = []
    for j in range(depth):
        chain.append(get_function(length=length_index, depth=j))
    for j in reversed(range(depth)):
        chain.append(chain[j].edge_rename({"D": "U", "U": "D"}))
    return chain


def merge_tensor(A, B, maybe_merge, maybe_contract):
    map_A = {}
    map_B = {}
    map_m = {}
    for name in maybe_merge:
        if name in A.name and name in B.name:
            map_A[name] = "A" + str(name)
            map_B[name] = "B" + str(name)
            map_m[name] = ["A" + str(name), "B" + str(name)]
    contract_pair = set()
    for a, b in maybe_contract:
        if a in A.name:
            contract_pair.add((a, b))
    return A.edge_rename(map_A).contract(B.edge_rename(map_B),
                                         contract_pair).merge_edge(map_m)


def contract_lattice_environment(length, depth, get_function, cut,
                                 left_not_right):
    if left_not_right:
        contract_pair = {("R", "L")}
    else:
        contract_pair = {("L", "R")}
    for i in reversed(range(length)):
        this = get_chain(get_function, i, depth)
        if i == length - 1:
            chain = this
        else:
            for j in range(depth * 2):
                chain[j] = merge_tensor(chain[j], this[j], {"U", "D"},
                                        contract_pair)
            chain = cut_line(chain, "D", "U", cut)
    return chain


def cut_line(chain, l, r, cut):
    """
    |   |   |   |
    X - X - X - X
    |   |   |   |
    """
    chain = chain[:]
    size = len(chain)
    for i in range(size - 1):
        Q, R = chain[i].qr('r', {r}, r, l)
        chain[i] = Q
        chain[i + 1] = chain[i + 1].contract(R, {(l, r)})

    for i in reversed(range(1, size)):
        U, S, V = chain[i].svd({l}, r, l, cut)
        chain[i] = V
        S /= S.norm_max()
        chain[i - 1] = chain[i - 1].contract(U.multiple(S, r, 'u'), {(r, l)})

    return chain
