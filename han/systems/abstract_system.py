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

    def _get_tensor(self, l1l2, param=None):
        # get tensors at l1 l2
        raise NotImplementedError("not implemented in abstract system")

    def _clear_tensor(self, key):
        # set tensors related to key to none
        raise NotImplementedError("not implemented in abstract system")

    def __init__(self, L1, L2, Dc, Tensor):
        self.L1 = L1  # depth
        self.L2 = L2  # length
        self.Dc = Dc
        self.Tensor = Tensor

        self.parameter = Param(self)
        self._parameter = {}  # dict[any, float]
        self._tensors = {}  # dict[tuple[int, int], tensor|None]
        self.hamiltonians = {}  # dict[tuple[int, int], tensor]


class AbstractSamplingSystem(AbstractSystem):

    def __init__(self, L1, L2, Dc, Tensor):
        super(AbstractSamplingSystem, self).__init__(L1, L2, Dc, Tensor)
        self.aux = None
        self.gen01 = None
        # 约定：有且仅有最外面一层经典Tensor层, and it has only two edge: U and D
        # Design:
        # s = get_new_configuration() # without classical tensor, may return None
        # ss.append(s)
        #
        # for s1 in ss:
        #   for s2 in ss:
        #     es.append(energy(s1, s2))
        # e = mean(es)
        # not support grad inside

    def _real_clear_tensor(self, l1, l2):
        self._tensors[(l1, l2)] = None

    def __getitem__(self, l1l2):
        if l1l2 not in self._tensors or self._tensors[l1l2] is None:
            self._tensors[l1l2] = self._get_tensor(l1l2)
        return self._tensors[l1l2]

    def energy_ss(self, s1, s2):
        L1 = self.L1
        L2 = self.L2
        p1 = [self[L1 - 1, l2].shrink({"U": s1[l2]}) for l2 in range(L2)]
        p2 = [self[L1 - 1, l2].shrink({"U": s2[l2]}) for l2 in range(L2)]
        n12 = [
            p1[l2].contract(p2[l2].conjugate(), {("D", "D")})
            for l2 in range(L2)
        ]
        total_e = 0.
        for positions, hamiltonian in self.hamiltonians.items():
            this_e = hamiltonian
            for index, position in enumerate(positions):
                this_e = this_e.contract(p1[position], {(f"I{index}", "D")})
                this_e = this_e.contract(p2[position], {(f"O{index}", "D")})
            for l2 in range(L2):
                if l2 not in positions:
                    this_e = this_e.contract(n12[l2], set())
            total_e += float(this_e)
        den = self.Tensor(1)
        for l2 in range(L2):
            den = den.contract(n12[l2], set())
        return total_e, float(den)

    def new_aux(self):
        self.aux = Auxiliaries(self.L1 * 2 - 2, self.L2, self.Dc, False,
                               self.Tensor)
        # 上下对称这一点没有用到，在测量过程中存在浪费
        for l1 in range(self.L1 - 1):
            for l2 in range(self.L2):
                tensor = self[l1, l2]
                self.aux[l1, l2] = tensor
                self.aux[2 * self.L1 - l1 - 3, l2] = tensor.edge_rename({
                    "U": "D",
                    "D": "U"
                }).conjugate()

    def get_configuration(self):
        # s ~ w(s)^2
        p = 1.
        aux = self.aux.copy()
        ss = []
        for l2 in range(self.L2):
            up_to_down = aux._inline_up_to_down[self.L1 - 2, l2]()
            down_to_up = aux._inline_down_to_up[self.L1 - 1, l2]()
            rho = down_to_up.contract(up_to_down,
                                      {("U1", "D1"),
                                       ("U3", "D3")}).blocks[["U2", "D2"]]
            rho = np.diagonal(rho).copy()
            rho[rho < 0] = 0  # 数值误差
            rho = rho / np.sum(rho)
            this_s = self.choice(rho)
            p *= rho[this_s]
            aux[self.L1 - 2, l2] = aux[self.L1 - 2, l2].shrink({"D": this_s})
            aux[self.L1 - 1, l2] = aux[self.L1 - 1, l2].shrink({"U": this_s})
            ss.append(this_s)
        return ss, aux, p

    def get_configurations(self, length):
        self.gen01 = TAT.random.uniform_real(0, 1)
        self.new_aux()

        ssb = [self.get_configuration() for _ in range(length)]
        # [([s], aux, p)]
        sss = [ss for ss, aux, p in ssb]
        auxs = [aux for ss, aux, p in ssb]
        old_p = [p for ss, aux, p in ssb]

        sss_uniq, sss_index, sss_count = np.unique(sss,
                                                   return_index=True,
                                                   return_counts=True,
                                                   axis=0)
        sss_aux = [auxs[i] for i in sss_index]
        sss_oldp = [old_p[i] for i in sss_index]
        sss_oldp = np.array(sss_oldp)
        sss_oldp /= np.sum(sss_oldp)

        self.gen01 = None
        self.aux = None
        return list(zip(sss_uniq, sss_aux, sss_oldp,
                        sss_count))  # [([s], aux, old_p, count)]

    def energy(self, ssb, hint=1):
        num = 0.
        den = 0.

        # the possibility it should be
        sss_newp = np.array([
            float(aux.hole((), hint=("V", hint)))
            for ss, aux, oldp, count in ssb
        ])
        sss_uniq = [ss for ss, aux, oldp, count in ssb]
        sss_count = [count for ss, aux, oldp, count in ssb]

        # P -> sqrt(P)
        # P N -> sqrt(P) N
        sss_newp = np.sqrt(sss_newp)
        sss_newp /= np.sum(sss_newp)

        for [s1, _, p1, n1], q1 in zip(ssb, sss_newp):
            for [s2, _, p2, n2], q2 in zip(ssb, sss_newp):
                n, d = self.energy_ss(s1, s2)
                # print(p1, p2, q1, q2)
                p = (n1 * n2) * (q1 * q2) / (p1 * p2)
                num += n * p
                den += d * p
        return num / den

    def _grad_of_param(self, ssb, energy):
        delta = 1e-3
        result = {}
        for k in self._parameter:
            param_rec = self._parameter.copy()

            self.parameter[k] += delta  # some tensor become None
            for l1 in range(self.L1 - 1):
                for l2 in range(self.L2):
                    if self._tensors[l1, l2] == None:
                        up_to_down = self[l1, l2]
                        down_to_up = up_to_down.edge_rename({
                            "U": "D",
                            "D": "U"
                        }).conjugate()

                        for ss, aux, oldp, count in ssb:
                            if l1 == self.L1 - 2:
                                this_s = ss[l2]
                                aux[l1, l2] = up_to_down.shrink({"D": this_s})
                                aux[2 * self.L1 - l1 - 3,
                                    l2] = down_to_up.shrink({"U": this_s})
                            else:
                                aux[l1, l2] = up_to_down
                                aux[2 * self.L1 - l1 - 3, l2] = down_to_up

            new_energy = self.energy(ssb)

            self.parameter[k] -= delta
            for l1 in range(self.L1 - 1):
                for l2 in range(self.L2):
                    if self._tensors[l1, l2] == None:
                        up_to_down = self[l1, l2]
                        down_to_up = up_to_down.edge_rename({
                            "U": "D",
                            "D": "U"
                        }).conjugate()

                        for ss, aux, oldp, count in ssb:
                            if l1 == self.L1 - 2:
                                this_s = ss[l2]
                                aux[l1, l2] = up_to_down.shrink({"D": this_s})
                                aux[2 * self.L1 - l1 - 3,
                                    l2] = down_to_up.shrink({"U": this_s})
                            else:
                                aux[l1, l2] = up_to_down
                                aux[2 * self.L1 - l1 - 3, l2] = down_to_up

            result[k] = (new_energy - energy) / delta
        return result

    def choice(self, rho):
        p = self.gen01()
        for i, r in enumerate(rho):
            p -= r
            if p < 0:
                return i
        return i


class AbstractHoleSystem(AbstractSystem):

    def __init__(self, L1, L2, Dc, Tensor):
        super(AbstractHoleSystem, self).__init__(L1, L2, Dc, Tensor)

        self.auxiliaries = None

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
            if self.auxiliaries is not None:
                self._set_auxiliaries(l1, l2)
        return self._tensors[l1l2]

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
        psipsi = float(self.auxiliaries.hole(()))
        for position, hamiltonian in self.hamiltonians.items():
            hole = self.auxiliaries.hole(tuple(
                (self.L1, l2) for l2 in position))
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
                this_hole = self.auxiliaries.hole(((l1, l2),))
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
                i = position[0]
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
