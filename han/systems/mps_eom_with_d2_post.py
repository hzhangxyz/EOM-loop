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

import TAT
from ..utility.storage_function import StorageFunction
from .abstract_system import AbstractSystem
from ..utility.tensor_U import tensor_U


@StorageFunction
def get_U(Tensor, n, r, omega, shrink=tuple()):
    result = tensor_U(n, n, r, omega, 0, 0).edge_rename({
        "I1": "L",
        "O1": "R",
        "O2": "D",
        "I2": "U"
    }).shrink({i: 0 for i in shrink})
    if Tensor == TAT.No.D.Tensor:
        return result.to(float)
    elif Tensor == TAT.No.S.Tensor:
        return result.to("float32")
    elif Tensor == TAT.No.C.Tensor:
        return result.to("complex64")
    elif Tensor == TAT.No.Z.Tensor:
        return result
    else:
        raise ValueError("invalid tensor type")


class MPS_EOM_with_x2_post(AbstractSystem):

    def __init__(self, depth, length, D, Dc, Tensor):
        super(MPS_EOM_with_x2_post, self).__init__(depth, length, Dc, Tensor)
        self.D = D

    def _get_tensor(self, l1l2, param=None):
        if param is None:
            param = self._parameter
        l1, l2 = l1l2
        r = param[(l1, l2, "r")]
        omega = param[(l1, l2, "omega")]
        shrink = set()
        if l1 == 0:
            shrink.add("U")
        if l2 == 0:
            shrink.add("L")
        if l2 == self.L2 - 1:
            shrink.add("R")
        result = get_U(self.Tensor, self.D, r, omega, tuple(shrink))
        if l1 == self.L1 - 1:
            projector = self.Tensor(["D", "U"], [self.d, self.D]).zero()
            projector[{"D": 0, "U": 0}] = param[("P", l2, 0, "p")]
            projector[{"D": 1, "U": 1}] = param[("P", l2, 1, "p")]
            result = result.contract(projector, {("D", "U")})
        return result

    def _clear_tensor(self, key):
        if len(key) == 3:
            l1, l2, param = key
            self._real_clear_tensor(l1, l2)
        if len(key) == 4:
            _, l2, d, _ = key
            self._real_clear_tensor(self.L1 - 1, l2)
