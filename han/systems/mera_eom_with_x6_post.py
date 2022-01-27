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
import scipy.linalg
import TAT
import lazy
from ..utility.storage_function import StorageFunction
from .abstract_system import AbstractSystem, r_bound
from ..utility.tensor_U import tensor_U


@StorageFunction
def get_U(Tensor, n, r, omega, config):
    result = tensor_U(n, n, r, omega, 0, 0).edge_rename({
        "I1": "I0",
        "O1": "O0",
        "O2": "O1",
        "I2": "I1"
    })

    if config == "I0S":
        result = result.shrink({
            "I0": 0
        }).edge_rename({
            "I1": "U",
            "O0": "D",
            "O1": "R"
        })
    elif config == "I1S":
        result = result.shrink({
            "I1": 0
        }).edge_rename({
            "I0": "U",
            "O0": "D",
            "O1": "R"
        })
    elif config == "I0SCLD":
        id_ld = Tensor(["L", "D"], [n, n]).identity({("L", "D")})
        result = result.shrink({
            "I0": 0
        }).contract(id_ld, set()).edge_rename({
            "I1": "U",
            "O1": "R"
        }).merge_edge({"D": ["D", "O0"]})
    elif config == "I1SCLD":
        id_ld = Tensor(["L", "D"], [n, n]).identity({("L", "D")})
        result = result.shrink({
            "I1": 0
        }).contract(id_ld, set()).edge_rename({
            "I0": "U",
            "O1": "R"
        }).merge_edge({"D": ["D", "O0"]})
    elif config == "I2S":
        result = result.shrink({
            "I0": 0,
            "I1": 0
        }).edge_rename({
            "O0": "D",
            "O1": "R"
        })
    elif config == "I2M":
        result = result.merge_edge({
            "U": ["I0", "I1"]
        }).edge_rename({
            "O0": "D",
            "O1": "R"
        })
    else:
        raise ValueError("invalid config")

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


class Mera_EOM_with_x6_post(AbstractSystem):

    def __init__(self, layer, D, Dc, Tensor):
        depth = layer * 2
        length = 1
        for i in range(layer):
            length += 1
            length *= 2
        super(Mera_EOM_with_x6_post, self).__init__(depth + 1, length, Dc,
                                                    Tensor)
        print(f"mera network depth {depth} length {length}")
        self.D = D

        self.layer = layer
        self._generate_structure()
        self._construct_params()
        self._construct_tensors()

    def _construct_params(self):
        LP = 1
        for l1 in range(self.L1 - 1):
            if l1 % 2 == 0:
                if l1 != 0:
                    LP *= 2
            else:
                LP += 1
            for lp in range(LP):
                self.parameter.add(("r", l1, lp))
                self.parameter.add(("omega", l1, lp))
        for l2 in range(self.L2):
            for ed in range(self.d**2):
                for e6 in range(self.d**2):
                    self.parameter.add(("P", l2, ed, e6))

    def _construct_tensors(self):
        for l1 in range(self.L1):
            for l2 in range(self.L2):
                self.tensor[l1][l2].replace(self._construct_tensor(l1, l2))

    def _generate_structure(self):
        last_length = 1
        param_length = [1]
        for i in range(self.layer):
            last_length += 1
            param_length.append(last_length)
            if i != self.layer - 1:
                last_length *= 2
                param_length.append(last_length)

        self._mera_params = {}
        self._mera_tensors = {}

        interval = 1
        for l1 in reversed(range(self.L1 - 1)):
            LP = param_length[l1]
            for lp in range(LP):
                if l1 % 2 == 1:
                    if l1 == self.L1 - 1 - 1:
                        l2 = lp * 2
                    else:
                        l2 = self._mera_params[(l1 + 1, lp * 2)][1]
                    self._mera_params[(l1, lp)] = (l1, l2)
                    if lp == 0:
                        config = "I0S"
                    elif lp == LP - 1:
                        config = "I1S"
                    else:
                        config = "I2M"
                    self._mera_tensors[(l1, l2)] = (l1, lp, config)
                    for delta in range(1, interval):
                        self._mera_tensors[(l1, l2 + delta)] = (-1, -1, "IDLR")
                    self._mera_tensors[(l1, l2 + interval)] = (-1, -1, "IDLD")
                else:
                    l2 = self._mera_params[(l1 + 1, lp)][1]
                    self._mera_params[(l1, lp)] = (l1, l2)
                    if l1 == 0:
                        config = "I2S"
                    elif lp % 2 == 0:
                        if lp == 0:
                            config = "I0S"
                        else:
                            config = "I0SCLD"
                    else:
                        config = "I1SCLD"
                    self._mera_tensors[(l1, l2)] = (l1, lp, config)
                    for delta in range(1, interval):
                        self._mera_tensors[(l1, l2 + delta)] = (-1, -1, "IDLR")
                    if lp == LP - 1:
                        self._mera_tensors[(l1, l2 + interval)] = (-1, -1,
                                                                   "IDLD")
            if l1 % 2 == 1:
                interval *= 2

    def _construct_projector_tensor(self, l2, *args):
        p1 = self.Tensor(["D", "U"], [self.d, self.D]).zero()
        for i in range(self.d):
            p1[{"D": i, "U": i}] = 1

        p2 = self.Tensor(["D", "U"], [self.d, self.d**2]).zero()
        for i in range(self.d):
            p2[{"D": i, "U": i}] = 1

        if l2 % 2 == 0:
            assert self.d == 2
            npa = np.array(args).reshape(self.d**2, self.d**2)
            to_exp = npa - npa.T
            u = scipy.linalg.expm(to_exp)
            projector = self.Tensor(["D", "U"], [self.d**2, self.d**2]).zero()
            projector.blocks[projector.names] = u
            projector = projector.split_edge(
                {"U": [("U", self.d), ("R", self.d)]})

            result = p1.contract(projector,
                                 {("D", "U")}).contract(p2, {("D", "U")})
            return result
        else:
            identity = self.Tensor(["L", "U"],
                                   [self.d, self.d]).identity({("L", "U")})
            fake = self.Tensor(["D"], [self.d]).zero()
            fake.storage[0] = 1
            result = p1.contract(identity,
                                 {("D", "U")}).contract(fake, {("D", "U")})
            return result

    def _construct_tensor(self, l1, l2):
        if l1 == self.L1 - 1:
            args = [
                self.parameter.param["P", l2, ed, e6]
                for ed in range(self.d**2)
                for e6 in range(self.d**2)
            ]
            return lazy.Node(self._construct_projector_tensor, l2, *args)
        l1l2 = l1, l2
        if l1l2 not in self._mera_tensors:
            return lazy.Root(self.Tensor(1))
        _, lp, config = self._mera_tensors[l1l2]
        if config == "IDLD":
            return lazy.Root(
                self.Tensor(["L", "D"],
                            [self.D, self.D]).identity({("L", "D")}))
        elif config == "IDLR":
            return lazy.Root(
                self.Tensor(["L", "R"],
                            [self.D, self.D]).identity({("L", "R")}))
        else:
            return lazy.Node(get_U, self.Tensor, self.D,
                             self.parameter.param["r", l1, lp],
                             self.parameter.param["omega", l1, lp], config)

    def _modified_tensor(self, key):
        if len(key) == 3:
            romega, l1, lp = key
            l1, l2 = self._mera_params[l1, lp]
            return [(l1, l2)]
        if len(key) == 4:
            _, l2, ed, e6 = key
            return [(self.L1 - 1, l2)]

    def generate_initial_state(self, seed):
        TAT.random.seed(seed)
        uni1 = TAT.random.uniform_real(-1, +1)
        unir = TAT.random.uniform_real(-r_bound, +r_bound)
        unipi = TAT.random.uniform_real(-3.14, +3.14)

        LP = 1
        for l1 in range(self.L1 - 1):
            if l1 % 2 == 0:
                if l1 != 0:
                    LP *= 2
            else:
                LP += 1
            for lp in range(LP):
                self.parameter["r", l1, lp] = unir()
                self.parameter["omega", l1, lp] = unipi()
        for l2 in range(self.L2):
            for ed in range(self.d**2):
                for e6 in range(self.d**2):
                    self.parameter["P", l2, ed, e6] = unipi()

    def refine_parameters(self):
        LP = 1
        for l1 in range(self.L1 - 1):
            if l1 % 2 == 0:
                if l1 != 0:
                    LP *= 2
            else:
                LP += 1
            for lp in range(LP):
                if self.parameter["r", l1, lp] > +r_bound:
                    self.parameter["r", l1, lp] = +r_bound
                if self.parameter["r", l1, lp] < -r_bound:
                    self.parameter["r", l1, lp] = -r_bound
