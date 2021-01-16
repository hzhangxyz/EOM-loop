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

def trace_mps(l, a):
    ah = a.shrink({"1.I1": 0, "2.I1": 0}).edge_rename({"1.O2": "A.1.O2", "2.O2": "A.2.O2"})
    result = ah.trace({("1.O1", "2.O1")})
    for t in range(1, l):
        result = result.contract(a.edge_rename({"1.O2": "A.1.O2", "2.O2": "A.2.O2"}), {("A.1.O2", "1.I1"), ("A.2.O2", "2.I1")})\
        .trace({("1.O1", "2.O1")})
    return result.trace({("A.1.O2", "A.2.O2")})

def contract_mps(l, a, b):
    ah = a.shrink({"1.I1": 0, "2.I1": 0}).edge_rename({"1.O2": "A.1.O2", "2.O2": "A.2.O2"})
    bh = b.shrink({"1.I1": 0, "2.I1": 0}).edge_rename({"1.O2": "B.1.O2", "2.O2": "B.2.O2"})
    result = ah.contract(bh, {("1.O1", "1.O1"), ("2.O1", "2.O1")});
    for t in range(1, l):
        result = result.contract(a.edge_rename({"1.O2": "A.1.O2", "2.O2": "A.2.O2"}), {("A.1.O2", "1.I1"), ("A.2.O2", "2.I1")})\
        .contract(
            b.edge_rename({"1.O2": "B.1.O2", "2.O2": "B.2.O2"}),
            {("B.1.O2", "1.I1"), ("B.2.O2", "2.I1"), ("1.O1", "1.O1"), ("2.O1", "2.O1")})
    return result.trace({("A.1.O2", "B.1.O2"), ("A.2.O2", "B.2.O2")})

def main(l, n, r, omega, phi, psi, delta):
    U = get_U(n, r, omega, phi, psi)
    S = get_site([1], [U])
    sample = 5
    Ss = get_site([1./(2*sample+1) for i in range(-sample, sample+1)],
                  [get_U(n, r * (1 + i * delta / sample), omega, phi, psi) for i in range(-sample, sample+1)])
    S /= S.norm_max()
    Ss /= Ss.norm_max()
    tracerhorhos = contract_mps(l, S, Ss)
    tracerho = trace_mps(l, S)
    tracerhos = trace_mps(l, Ss)
    f = tracerhorhos / (tracerho * tracerhos)
    print(f"Fidelity is {complex(f).real}")

if __name__ == "__main__":
    import fire
    def Display(lines, out):
        text = "\n".join(lines) + "\n"
        out.write(text)
    fire.core.Display = Display
    fire.Fire(main)
