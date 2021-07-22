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
import TAT
import tools

Tensor = TAT(float)


# Hamiltonian is selected by environment variable
def get_H(n=2):
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
    elif name == "ZIsing":
        "g Sx + SzSz"
        g = float(os.environ["IsingG"])
        block[0, 0, 0, 0] = 1 / 4.
        block[0, 1, 0, 1] = -1 / 4.
        block[1, 0, 1, 0] = -1 / 4.
        block[1, 1, 1, 1] = 1 / 4.
        block[0, 0, 0, 1] = g / 2.
        block[0, 1, 0, 0] = g / 2.
        block[1, 0, 1, 1] = g / 2.
        block[1, 1, 1, 0] = g / 2.
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
