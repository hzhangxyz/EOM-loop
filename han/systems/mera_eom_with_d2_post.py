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


class Mera_EOM_with_d2_post(AbstractSystem):

    def __init__(self, layer, D, Dc, Tensor):
        depth = layer * 2
        length = 1
        for i in range(layer):
            length += 1
            length *= 2
        super(Mera_EOM_with_d2_post, self).__init__(depth, length, Dc, Tensor)
        self.D = D

        self.layer = layer
        self._generate_structure()

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
        for l1 in reversed(range(self.L1)):
            LP = param_length[l1]
            for lp in range(LP):
                if l1 % 2 == 1:
                    if l1 == self.L1 - 1:
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

    def _get_tensor(self, l1l2, param=None):
        if param is None:
            param = self._parameter
        l1, l2 = l1l2
        if l1l2 not in self._mera_tensors:
            return self.Tensor(1)
        _, lp, config = self._mera_tensors[l1l2]
        if config == "IDLD":
            result = self.Tensor(["L", "D"],
                                 [self.D, self.D]).identity({("L", "D")})
        elif config == "IDLR":
            result = self.Tensor(["L", "R"],
                                 [self.D, self.D]).identity({("L", "R")})
        else:
            r = param[(l1, lp, "r")]
            omega = param[(l1, lp, "omega")]
            result = get_U(self.Tensor, self.D, r, omega, config)
        if l1 == self.L1 - 1:
            projector = self.Tensor(["D", "U"], [self.d, self.D]).zero()
            projector[{"D": 0, "U": 0}] = param[("P", l2, 0, "p")]
            projector[{"D": 1, "U": 1}] = param[("P", l2, 1, "p")]
            result = result.contract(projector, {("D", "U")})
        return result

    def _clear_tensor(self, key):
        if len(key) == 3:
            l1, lp, param = key
            l1, l2 = self._mera_params[l1, lp]
            self._real_clear_tensor(l1, l2)
        if len(key) == 4:
            _, l2, d, _ = key
            self._real_clear_tensor(self.L1 - 1, l2)
