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
import lazy
from ..utility.storage_function import StorageFunction
from .abstract_system import AbstractSystem, r_bound
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


class MPS_EOM_with_x4_post(AbstractSystem):

    def __init__(self, depth, length, D, Dc, Tensor):
        super(MPS_EOM_with_x4_post, self).__init__(depth + 1, length, Dc,
                                                   Tensor)
        self.D = D

        self._construct_params()
        self._construct_tensors()

    def _construct_params(self):
        for l1 in range(self.L1 - 1):
            for l2 in range(self.L2):
                self.parameter.add(("r1", l1, l2))
                self.parameter.add(("omega1", l1, l2))
                self.parameter.add(("r2", l1, l2))
                self.parameter.add(("omega2", l1, l2))
        for l2 in range(self.L2):
            for ed in range(self.d**2):
                for e4 in range(self.d**2):
                    self.parameter.add(("P", l2, ed, e4))

    def _construct_tensors(self):
        for l1 in range(self.L1):
            for l2 in range(self.L2):
                self._construct_tensor(l1, l2)

    def _construct_projector_tensor(self, l2, *args):
        p1 = self.Tensor(["D", "U"], [self.d**2, self.D**2]).zero()
        for i in range(self.d):
            for j in range(self.d):
                index = i * self.d + j
                p1[{"D": index, "U": index}] = 1

        npa = np.array(args).reshape(self.d**2, self.d**2)
        q, r = np.linalg.qr(npa)
        d = np.diag(np.sign(r.diagonal()))
        projector = self.Tensor(["D", "U"], [self.d**2, self.d**2]).zero()
        projector.blocks[projector.names] = q @ d

        p2 = self.Tensor(["D", "U"], [self.d, self.d**2]).zero()
        for i in range(self.d):
            p2[{"D": i, "U": i}] = 1

        return p1.contract(projector, {("D", "U")}).contract(p2, {("D", "U")})

    def _construct_normal_tensor(self, l1, l2, r1, omega1, r2, omega2):
        shrink = set()
        if l1 == 0:
            shrink.add("U")
        if l2 == 0:
            shrink.add("L")
        if l2 == self.L2 - 1:
            shrink.add("R")
        result1 = get_U(self.Tensor, self.D, r1, omega1)
        result2 = get_U(self.Tensor, self.D, r2, omega2)
        result = result1.edge_rename({
            "D": "D1",
            "U": "U1"
        }).contract(result2.edge_rename({
            "D": "D2",
            "U": "U2"
        }), {("R", "L")}).merge_edge({
            "U": ["U1", "U2"],
            "D": ["D1", "D2"]
        })
        return result.shrink({i: 0 for i in shrink})

    def _construct_tensor(self, l1, l2):
        if l1 == self.L1 - 1:
            args = [
                self.parameter.param["P", l2, ed, e4]
                for ed in range(self.d**2)
                for e4 in range(self.d**2)
            ]
            self.tensor[l1][l2].replace(
                lazy.Node(self._construct_projector_tensor, l2, *args))
        else:
            self.tensor[l1][l2].replace(
                lazy.Node(self._construct_normal_tensor, l1, l2,
                          self.parameter.param["r1", l1, l2],
                          self.parameter.param["omega1", l1, l2],
                          self.parameter.param["r2", l1, l2],
                          self.parameter.param["omega2", l1, l2]))

    def _modified_tensor(self, key):
        if len(key) == 3:
            romega, l1, l2 = key
            return [(l1, l2)]
        if len(key) == 4:
            _, l2, ed, e4 = key
            return [(self.L1 - 1, l2)]

    def generate_initial_state(self, seed):
        TAT.random.seed(seed)
        uni1 = TAT.random.uniform_real(-1, +1)
        unir = TAT.random.uniform_real(-r_bound, +r_bound)
        unipi = TAT.random.uniform_real(-3.14, +3.14)

        for l1 in range(self.L1 - 1):
            for l2 in range(self.L2):
                self.parameter["r1", l1, l2] = unir()
                self.parameter["omega1", l1, l2] = unipi()
                self.parameter["r2", l1, l2] = unir()
                self.parameter["omega2", l1, l2] = unipi()
        for l2 in range(self.L2):
            for ed in range(self.d**2):
                for e4 in range(self.d**2):
                    self.parameter["P", l2, ed, e4] = uni1()

    def refine_parameters(self):
        for l1 in range(self.L1 - 1):
            for l2 in range(self.L2):
                if self.parameter["r1", l1, l2] > +r_bound:
                    self.parameter["r1", l1, l2] = +r_bound
                if self.parameter["r1", l1, l2] < -r_bound:
                    self.parameter["r1", l1, l2] = -r_bound
                if self.parameter["r2", l1, l2] > +r_bound:
                    self.parameter["r2", l1, l2] = +r_bound
                if self.parameter["r2", l1, l2] < -r_bound:
                    self.parameter["r2", l1, l2] = -r_bound
        max_P = 0
        for l2 in range(self.L2):
            for ed in range(self.d**2):
                for e4 in range(self.d**2):
                    value = abs(self.parameter["P", l2, ed, e4])
                    if value > max_P:
                        max_P = value
        for l2 in range(self.L2):
            for ed in range(self.d**2):
                for e4 in range(self.d**2):
                    self.parameter["P", l2, ed,
                                   e4] = self.parameter["P", l2, ed, e4] / max_P
