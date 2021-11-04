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


class Param:

    def __init__(self, owner):
        self.owner = owner

    def __setitem__(self, key, value):
        if key not in self.owner._parameter or self.owner._parameter[
                key] != value:
            self.owner._parameter[key] = value
            self.owner._clear_tensor(key)

    def __getitem__(self, key):
        return self.owner._parameter[key]


class AbstractSystem:

    def __init__(self, L1, L2, Dc, Tensor):
        self.L1 = L1
        self.L2 = L2
        self.Dc = Dc
        self.Tensor = Tensor
        self.parameter = Param(self)
        self._parameter = {}  # dict[any, float]
        self._tensors = {}  # dict[tuple[int, int], tensor|None]
        self.hamiltonians = {}  # dict[tuple[int, int], tensor]
        self.auxiliaries = None

    def _clear_tensor(self, key):
        # set tensors related to key to none
        raise NotImplementedError("not implemented in abstract system")

    def _real_clear_tensor(self, l1, l2):
        self._tensors[(l1, l2)] = None
        if self.auxiliaries is not None:
            self.auxiliaries[(l1, l2)] = None

    def _set_auxiliaries(self, l1, l2):
        l1l2 = (l1, l2)
        tensor = self._tensors[l1l2]
        """
        auxiliaries:
        0   X X X
        ... .....
        L-1 X X X
        L   HHHHH
        L+1 X X X
        ... .....
        L+L X X X
        """
        self.auxiliaries[l1, l2] = tensor
        self.auxiliaries[2 * self.L1 - l1, l2] = tensor.edge_rename({
            "U": "D",
            "D": "U"
        }).conjugate()

    def __getitem__(self, l1l2):
        if l1l2 not in self._tensors or self._tensors[l1l2] is None:
            self._tensors[l1l2] = self._get_tensor(l1l2)
            l1, l2 = l1l2
            self._set_auxiliaries(l1, l2)
        return self._tensors[l1l2]

    def _get_tensor(self, l1l2, param=None):
        # get tensors at l1 l2
        raise NotImplementedError("not implemented in abstract system")

    def refresh_auxiliaries(self):
        if self.auxiliaries is None:
            self.auxiliaries = Auxiliaries(self.L1 * 2 + 1, self.L2, self.Dc,
                                           False, self.Tensor)
            for l1 in range(self.L1):
                for l2 in range(self.L2):
                    _ = self[l1, l2]
                    self._set_auxiliaries(l1, l2)
        else:
            for l1 in range(self.L1):
                for l2 in range(self.L2):
                    _ = self[l1, l2]
        for l2 in range(self.L2):
            d = self[self.L1 - 1, l2].edges("D")
            i = self.Tensor(["U", "D"], [d, d]).identity({("U", "D")})
            self.auxiliaries[self.L1, l2] = i

    def _energies(self):
        psiHpsi = 0
        psipsi = 0
        psipsi = float(self.auxiliaries(()))
        for position, hamiltonian in self.hamiltonians.items():
            hole = self.auxiliaries(tuple((self.L1, l2) for l2 in position))
            body = len(hamiltonian.names) // 2
            this = hole.contract(
                hamiltonian, {
                    *((f"D{i}", f"I{i}") for i in range(body)),
                    *((f"U{i}", f"O{i}") for i in range(body))
                })
            psiHpsi += float(this)
        return psiHpsi, psipsi

    def _collect_hole(self):
        result = {}
        for l1 in range(self.L1):
            for l2 in range(self.L2):
                this_hole = self.auxiliaries(((l1, l2),))
                this_hole = this_hole.edge_rename({
                    "U0": "D",
                    "D0": "U",
                    "L0": "R",
                    "R0": "L"
                }).conjugate()
                result[(l1, l2)] = this_hole
        return result

    def _hole_of_tensor(self):
        hole_of_psipsi = {}
        hole_of_psiHpsi = {}
        # dict[tuple[int, int], tensor]
        hole_of_psipsi = self._collect_hole()
        for position, hamiltonian in self.hamiltonians.items():
            if len(position) >= 3:
                raise NotImplementedError("3 body hamitlonian not implement")
            if len(position) == 2:
                i, j = position
                if i + 1 != j:
                    raise NotImplementedError(
                        "non nearest hamiltonian not implement")

                self.auxiliaries[self.L1, i] = hamiltonian.edge_rename({
                    "I0": "U",
                    "O0": "D"
                }).merge_edge({"R": ["I1", "O1"]})
                d = self[self.L1 - 1, j].edges("D")
                self.auxiliaries[self.L1, i + 1] = self.Tensor(
                    ["U", "LU", "D", "LD"], [d, d, d, d]).identity({
                        ("U", "LU"), ("D", "LD")
                    }).merge_edge({"L": ["LU", "LD"]})

                this = self._collect_hole()

                d = self[self.L1 - 1, j].edges("D")
                iden = self.Tensor(["U", "D"], [d, d]).identity({("U", "D")})
                self.auxiliaries[self.L1, j] = iden
                d = self[self.L1 - 1, i].edges("D")
                iden = self.Tensor(["U", "D"], [d, d]).identity({("U", "D")})
                self.auxiliaries[self.L1, i] = iden
            if len(position) == 1:
                i = position
                self.auxiliaries[self.L1, i] = hamiltonian.edge_rename({
                    "I0": "U",
                    "O0": "D"
                })

                this = self._collect_hole()

                d = self[self.L1 - 1, i].edges("D")
                iden = self.Tensor(["U", "D"], [d, d]).identity({("U", "D")})
                self.auxiliaries[self.L1, i] = iden
            for i, j in this.items():
                if i not in hole_of_psiHpsi:
                    hole_of_psiHpsi[i] = j
                else:
                    hole_of_psiHpsi[i] += j
        return hole_of_psiHpsi, hole_of_psipsi

    def _grad_of_tensor(self):
        psiHpsi, psipsi = self._energies()
        hole_of_psiHpsi, hole_of_psipsi = self._hole_of_tensor()
        result = {}
        for l1 in range(self.L1):
            for l2 in range(self.L2):
                l1l2 = (l1, l2)
                result[l1l2] = 2 * hole_of_psiHpsi[l1l2] / psipsi - psiHpsi / (
                    psipsi * psipsi) * 2 * hole_of_psipsi[l1l2]
        return result

    def _grad_of_param(self):
        grad_of_tensor = self._grad_of_tensor()
        delta = 1e-3
        result = {}
        for k in self._parameter:
            param = self._parameter.copy()
            param[k] += delta
            param_grad = 0
            for l1 in range(self.L1):
                for l2 in range(self.L2):
                    tensor_diff = (self._get_tensor(
                        (l1, l2), param) - self[l1, l2]) / delta
                    param_grad += float(
                        tensor_diff.contract(
                            grad_of_tensor[(l1, l2)],
                            {(n, n) for n in tensor_diff.names}))
            result[k] = param_grad
        return result
