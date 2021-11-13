#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2020-2021 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

from tetragono.auxiliaries import Auxiliaries
from .abstract_system import AbstractSystem


def inner_product(psi1, psi2):
    # this two should be AbstractSystem with Lattice implement
    # their L2, Dc, Tensor should be the same
    assert (psi1.L2 == psi2.L2)
    assert (psi1.Dc == psi2.Dc)
    assert (psi1.Tensor == psi2.Tensor)
    L2 = psi1.L2
    Dc = psi1.Dc
    Tensor = psi1.Tensor

    L1s = psi1.L1 + psi2.L1  # L1 sum

    auxiliaries = Auxiliaries(L1s, L2, Dc, False, Tensor)
    for l1 in range(psi1.L1):
        for l2 in range(psi1.L2):
            auxiliaries[l1, l2] = psi1[l1, l2]
    for l1 in range(psi2.L1):
        for l2 in range(psi2.L2):
            auxiliaries[L1s - l1 - 1, l2] = psi2[l1, l2].edge_rename({
                "U": "D",
                "D": "U"
            }).conjugate()
    return float(auxiliaries(()))
