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
from matrix_A import matrix_A
from matrix_U import matrix_U

Tensor = TAT(complex)


def get_U(n, r, omega, phi, psi):
    result = Tensor(["I1", "I2", "O1", "O2"], [n, n, n, n])
    result.block[{}] = matrix_U(n, n, r, omega, phi, psi)
    return result


def get_site(ps, Us):
    return sum([
        p * U.edge_rename({
            "I1": "1.I1",
            "I2": "1.I2",
            "O1": "1.O1",
            "O2": "1.O2"
        }).shrink({
            "1.I2": 0
        }).contract(
            U.edge_rename({
                "I1": "2.I1",
                "I2": "2.I2",
                "O1": "2.O1",
                "O2": "2.O2"
            }).shrink({
                "2.I2": 0
            }).conjugate(), set()) for p, U in zip(ps, Us)
    ])

def get_A(n, k, eta):
    result = Tensor(["O", "I"], [n, n])
    result.block[{}] = matrix_A(n, n, k, eta)
    return result

def trace_mps(l, a, g = None, bs = None):
    ah = a.shrink({"1.I1": 0, "2.I1": 0}).edge_rename({"1.O2": "A.1.O2", "2.O2": "A.2.O2"})
    if bs:
        ah = ah.contract(bs, {("1.O1", "1")}).edge_rename({"2": "1.O1"})
    result = ah.trace({("1.O1", "2.O1")})
    for t in range(1, l):
        if g:
            result = result.contract(g.edge_rename({"1.O": "A.1.O2", "2.O": "A.2.O2"}), {("A.1.O2", "1.I"), ("A.2.O2", "2.I")})
        result = result.contract(a.edge_rename({"1.O2": "A.1.O2", "2.O2": "A.2.O2"}), {("A.1.O2", "1.I1"), ("A.2.O2", "2.I1")})
        if bs:
            result = result.contract(bs, {("1.O1", "1")}).edge_rename({"2": "1.O1"})
        result = result.trace({("1.O1", "2.O1")})
    return result.trace({("A.1.O2", "A.2.O2")})

def contract_mps(l, a, b, ga = None, gb = None, bs = None):
    ah = a.shrink({"1.I1": 0, "2.I1": 0}).edge_rename({"1.O2": "A.1.O2", "2.O2": "A.2.O2"})
    bh = b.shrink({"1.I1": 0, "2.I1": 0}).edge_rename({"1.O2": "B.1.O2", "2.O2": "B.2.O2"})
    if bs:
        ah = ah.contract(bs, {("1.O1", "1")}).edge_rename({"2": "1.O1"}).contract(bs, {("2.O1", "1")}).edge_rename({"2": "2.O1"})
    result = ah.contract(bh, {("1.O1", "1.O1"), ("2.O1", "2.O1")});
    for t in range(1, l):
        if ga:
            result = result.contract(ga.edge_rename({"1.O": "A.1.O2", "2.O": "A.2.O2"}), {("A.1.O2", "1.I"), ("A.2.O2", "2.I")})
        if gb:
            result = result.contract(gb.edge_rename({"1.O": "B.1.O2", "2.O": "B.2.O2"}), {("B.1.O2", "1.I"), ("B.2.O2", "2.I")})
        result = result.contract(a.edge_rename({"1.O2": "A.1.O2", "2.O2": "A.2.O2"}), {("A.1.O2", "1.I1"), ("A.2.O2", "2.I1")})
        if bs:
            result = result.contract(bs, {("1.O1", "1")}).edge_rename({"2": "1.O1"}).contract(bs, {("2.O1", "1")}).edge_rename({"2": "2.O1"})
        result = result.contract(
            b.edge_rename({"1.O2": "B.1.O2", "2.O2": "B.2.O2"}),
            {("B.1.O2", "1.I1"), ("B.2.O2", "2.I1"), ("1.O1", "1.O1"), ("2.O1", "2.O1")})
    return result.trace({("A.1.O2", "B.1.O2"), ("A.2.O2", "B.2.O2")})

def main(l, n, r, omega, phi, psi, eta, delta, backward = 3):
    U = get_U(n, r, omega, phi, psi)
    S = get_site([1], [U])
    sample = 5
    Ss = get_site([1./(2*sample+1) for i in range(-sample, sample+1)],
                  [get_U(n, r * (1 + i * delta / sample), omega, phi, psi) for i in range(-sample, sample+1)])
    S /= S.norm_max()
    Ss /= Ss.norm_max()
    Aks = [get_A(n, k, eta) for k in range(n)]
    gsite = sum([Ak.edge_rename({"I": "1.I", "O": "1.O"}).contract(Ak.conjugate().edge_rename({"I": "2.I", "O": "2.O"}), set()) for Ak in Aks])
    bsite = Tensor(["1", "2"], [n, n]).zero()
    for i in range(backward):
        bsite[{"1":i, "2":i}] = 1
    tracerhorhos = contract_mps(l, S, Ss, gb = gsite, bs = bsite)
    tracerho = trace_mps(l, S, bs = bsite)
    tracerhos = trace_mps(l, Ss, gsite, bs = bsite)
    f = tracerhorhos / (tracerho * tracerhos)
    print(f"Fidelity is {complex(f).real}")

if __name__ == "__main__":
    import fire
    def Display(lines, out):
        text = "\n".join(lines) + "\n"
        out.write(text)
    fire.core.Display = Display
    fire.Fire(main)
