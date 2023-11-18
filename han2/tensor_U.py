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

import numpy as np
import TAT
from .matrix_U import matrix_U
from .storage_function import StorageFunction

Tensor = TAT.No.Z.Tensor


@StorageFunction
def tensor_U(n, c, r, omega, phi, psi):
    """
    根据矩阵截断, 物理截断, r, omega, phi, psi创建U矩阵
    """
    fake_cut = 2 * c - 1
    data = np.array(matrix_U(fake_cut, r, omega, phi, psi))
    result = Tensor(["I1", "I2", "O1", "O2"], [n, n, n, n]).zero()
    result.blocks[result.names][:c, :c, :c, :c] = data[:c, :c, :c, :c]
    return result
