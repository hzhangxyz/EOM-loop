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
                self.parameter.add(("r", l1, l2))
                self.parameter.add(("omega", l1, l2))
        for l2 in range(0, self.L2, 2):
            for ed in range(self.d**2):
                for e4 in range(self.d**2):
                    self.parameter.add(("P", l2, ed, e4))

    def _construct_tensors(self):
        for l1 in range(self.L1):
            for l2 in range(self.L2):
                self._construct_tensor(l1, l2)

    def _construct_projector_tensor(self, l2, *args):
        p = self.Tensor(["D", "U"], [self.d, self.D]).zero()
        for i in range(self.d):
            p[{"D": i, "U": i}] = 1

        npa = np.array(args).reshape(self.d**2, self.d**2)
        q, r = np.linalg.qr(npa)
        d = np.diag(np.sign(r.diagonal()))
        projector = self.Tensor(["D", "U"], [self.d**2, self.d**2]).zero()
        projector.blocks[projector.names] = q @ d
        projector = projector.split_edge({
            "D": [("D", self.d), ("D'", self.d)],
            "U": [("U", self.d), ("U'", self.d)]
        }).merge_edge({"R": ["D'", "U'"]})
        return projector.contract(p, {("U", "D")})

    def _construct_id(self):
        p = self.Tensor(["D", "U"], [self.d, self.D]).zero()
        for i in range(self.d):
            p[{"D": i, "U": i}] = 1

        projector = self.Tensor(["D", "U"],
                                [self.d**2, self.d**2]).identity({("D", "U")})
        projector = projector.split_edge({
            "D": [("D", self.d), ("D'", self.d)],
            "U": [("U", self.d), ("U'", self.d)]
        }).merge_edge({"L": ["D'", "U'"]})
        return projector.contract(p, {("U", "D")})

    def _construct_normal_tensor(self, l1, l2, r, omega):
        shrink = set()
        if l1 == 0:
            shrink.add("U")
        if l2 == 0:
            shrink.add("L")
        if l2 == self.L2 - 1:
            shrink.add("R")
        result = get_U(self.Tensor, self.D, r, omega, tuple(shrink))
        return result

    def _construct_tensor(self, l1, l2):
        if l1 == self.L1 - 1:
            if l2 % 2 == 0:
                args = [
                    self.parameter.param["P", l2, ed, e4]
                    for ed in range(self.d**2)
                    for e4 in range(self.d**2)
                ]
                self.tensor[l1][l2].replace(
                    lazy.Node(self._construct_projector_tensor, l2, *args))
            else:
                self.tensor[l1][l2].replace(lazy.Node(self._construct_id))
        else:
            self.tensor[l1][l2].replace(
                lazy.Node(self._construct_normal_tensor, l1, l2,
                          self.parameter.param["r", l1, l2],
                          self.parameter.param["omega", l1, l2]))

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
                self.parameter["r", l1, l2] = unir()
                self.parameter["omega", l1, l2] = unipi()
        for l2 in range(0, self.L2, 2):
            for ed in range(self.d**2):
                for e4 in range(self.d**2):
                    self.parameter["P", l2, ed, e4] = uni1()

    def refine_parameters(self):
        for l1 in range(self.L1 - 1):
            for l2 in range(self.L2):
                if self.parameter["r", l1, l2] > +r_bound:
                    self.parameter["r", l1, l2] = +r_bound
                if self.parameter["r", l1, l2] < -r_bound:
                    self.parameter["r", l1, l2] = -r_bound
        max_P = 0
        for l2 in range(0, self.L2, 2):
            for ed in range(self.d**2):
                for e4 in range(self.d**2):
                    value = abs(self.parameter["P", l2, ed, e4])
                    if value > max_P:
                        max_P = value
        for l2 in range(0, self.L2, 2):
            for ed in range(self.d**2):
                for e4 in range(self.d**2):
                    self.parameter["P", l2, ed,
                                   e4] = self.parameter["P", l2, ed, e4] / max_P
