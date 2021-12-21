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
from tetragono.auxiliaries import Auxiliaries


class Param:

    def __getstate__(self):
        return {key: node() for key, node in self.param.items()}

    def __setstate__(self, state):
        self.param = {}
        for key, value in state.items():
            self.add(key)
            self[key] = value

    def __init__(self):
        self.param = {}

    def add(self, key):
        self.param[key] = lazy.Root()

    def __setitem__(self, key, value):
        self.param[key].reset(value)

    def __getitem__(self, key):
        return self.param[key]()


class AbstractSystem:

    def __init__(self, L1, L2, Dc, Tensor):
        self.L1 = L1  # depth
        self.L2 = L2  # length
        self.Dc = Dc
        self.Tensor = Tensor

        self.parameter = Param()
        self.tensor = [[lazy.Root() for l2 in range(L2)] for l1 in range(L1)]
        self.hamiltonians = {}  # dict[tuple[int, int], tensor]

    def __setstate__(self, state):
        for key, value in state.items():
            setattr(self, key, value)
        self.tensor = [
            [lazy.Root() for l2 in range(self.L2)] for l1 in range(self.L1)
        ]
        self._construct_tensors()
        self._construct_auxiliaries()

    def __getstate__(self):
        except_list = ["auxiliaries", "tensor", "configuration"]
        state = {
            key: getattr(self, key)
            for key in self.__dict__
            if key not in except_list
        }
        return state


class AbstractSamplingSystem(AbstractSystem):

    def __init__(self, L1, L2, Dc, Tensor):
        # 约定：有且仅有最外面一层经典Tensor层, and it has only two edge: U and D
        super(AbstractSamplingSystem, self).__init__(L1, L2, Dc, Tensor)
        self._construct_auxiliaries()

    def _construct_auxiliaries(self):
        # when copy aux, need to copy conf togethor
        self.configuration = [lazy.Root() for l2 in range(self.L2)]

        self.auxiliaries = {}

        self.auxiliaries[1] = Auxiliaries(self.L1 - 1, self.L2, self.Dc, False,
                                          self.Tensor)
        self.auxiliaries[2] = Auxiliaries(self.L1 * 2 - 2, self.L2, self.Dc,
                                          False, self.Tensor)
        for l1 in range(self.L1 - 1):
            for l2 in range(self.L2):
                tensor_node = lazy.Node(
                    lambda tensor, conf: tensor.shrink({"D": conf})
                    if l1 == self.L1 - 2 and conf is not None else tensor,
                    self.tensor[l1][l2], self.configuration[l2])
                self.auxiliaries[1]._lattice[l1][l2].replace(tensor_node)
                self.auxiliaries[2]._lattice[l1][l2].replace(tensor_node)
                transposed_node = lazy.Node(
                    lambda tensor: tensor.edge_rename({
                        "U": "D",
                        "D": "U"
                    }).conjugate(), self.auxiliaries[2]._lattice[l1][l2])
                self.auxiliaries[2]._lattice[2 * self.L1 - l1 -
                                             3][l2].replace(transposed_node)

    def _construct_branch(self, s1, s2):
        L1 = self.L1
        L2 = self.L2
        p1 = [
            lazy.Node(lambda tensor, conf: tensor.shrink({"U": conf}),
                      self.tensor[L1 - 1][l2], s1[l2]) for l2 in range(L2)
        ]
        p2 = [
            lazy.Node(lambda tensor, conf: tensor.shrink({"U": conf}),
                      self.tensor[L1 - 1][l2], s2[l2]) for l2 in range(L2)
        ]
        n12 = [
            lazy.Node(lambda a, b: float(a.contract(b, {("D", "D")})), pp1, pp2)
            for pp1, pp2 in zip(p1, p2)
        ]

        es = {}
        for positions, hamiltonian in self.hamiltonians.items():
            es[positions] = lazy.Node(
                self._get_branch_core,
                hamiltonian,
                *(p1[p] for p in positions),
                *(p2[p] for p in positions),
                *(n12[p] for p in positions),
            )

        e = lazy.Node(lambda *v: np.sum(v), *es.values())
        den = lazy.Node(lambda *v: np.prod(v), *n12)

        return (p1, p2, n12, es, e, den)

    def _get_branch_core(self, H, *args):
        result = H
        rank = H.rank // 2
        for index in range(rank):
            p1 = args[index]
            p2 = args[index + rank]
            n12 = args[index + rank * 2]
            result = result.contract(p1, {(f"I{index}", "D")})
            result = result.contract(p2, {(f"O{index}", "D")})
            result /= n12
        return float(result)

    def energy_ss(self, branch, changed=None):
        p1, p2, n12, es, e, den = branch
        if changed is not None:
            cp, l2 = changed
            cp(p1[l2])
            cp(p2[l2])
            cp(n12[l2])
            for k in es:
                cp(es[k])
            e = cp(e)
            den = cp(den)
        return e(), den()

    def clear_configuration(self):
        for l2 in range(self.L2):
            self.configuration[l2].reset()

    def get_configuration(self, gen01):
        cp = lazy.Copy()
        conf = [cp(s) for s in self.configuration]
        aux1 = self.auxiliaries[1].copy(cp)
        aux2 = self.auxiliaries[2].copy(cp)
        # s ~ w(s)^2
        possibility = 1.
        for l2 in range(self.L2):
            up_to_down = aux2._inline_up_to_down[self.L1 - 2, l2]()
            down_to_up = aux2._inline_down_to_up[self.L1 - 1, l2]()
            rho = down_to_up.contract(up_to_down,
                                      {("U1", "D1"),
                                       ("U3", "D3")}).blocks[["U2", "D2"]]
            rho = np.diagonal(rho).copy()
            rho[rho < 0] = 0  # 数值误差
            rho = rho / np.sum(rho)
            this_s = self.choice(gen01(), rho)
            possibility *= rho[this_s]
            conf[l2].reset(this_s)
        # This lazy node also depend on tensor and param, it only copy conf and aux
        return possibility, conf, aux1, aux2

    def get_configurations(self, gen01, length):
        data = [self.get_configuration(gen01) for _ in range(length)]
        confs = [[s() for s in c] for _, c, _, _ in data]

        conf_uniq, conf_index, conf_count = np.unique(confs,
                                                      return_index=True,
                                                      return_counts=True,
                                                      axis=0)
        # poss, conf, aux1, aux2, count
        return [(data[i][0], data[i][1], data[i][2], data[i][3], c)
                for i, c in zip(conf_index, conf_count)]

    def energy(self, data, branchs=None, changed=None):
        change_classical = change_quantum = False
        if changed is not None:
            cp, l1, l2 = changed
            if l1 == self.L1 - 1:
                change_classical = True
            else:
                change_quantum = True

        if branchs is None:
            branchs = [[
                self._construct_branch(s1, s2) for _, s2, _, _, _ in data
            ] for _, s1, _, _, _ in data]
        num = 0.
        den = 0.

        total_count = sum(count for p, conf, aux1, aux2, count in data)

        # the possibility it should be
        wss = []
        for p, conf, aux1, aux2, count in data:
            replacement = {}
            if change_quantum:
                replacement[l1, l2] = cp(aux1._lattice[l1][l2])()
            result = aux1.replace(replacement)
            wss.append(abs(float(result)))
        # This abs is the reason to use sampling

        cls_change = None
        if change_classical:
            cls_change = cp, l2
        for i1, [[p1, s1, _, _, n1], q1] in enumerate(zip(data, wss)):
            for i2, [[p2, s2, _, _, n2], q2] in enumerate(zip(data, wss)):
                branch = branchs[i1][i2]
                e, d = self.energy_ss(branch, cls_change)
                p = (n1 * n2) * (q1 * q2) / (p1 * p2)
                dp = d * p
                num += e * dp
                den += dp
        return num / den, branchs

    # It is complex to compute grad of tensor here, so calculate grad of param directly

    def grad_of_param(self, data, energy, branchs):
        delta = 1e-3
        result = {}
        for k in self.parameter.param:
            modified = self._modified_tensor(k)
            if len(modified) != 1:
                raise NotImplementedError(
                    "Not work if multiple modified tensor")
            l1, l2 = modified[0]

            cp = lazy.Copy()
            new_param = cp(self.parameter.param[k])
            new_param.reset(new_param() + delta)
            cp(self.tensor[l1][l2])
            new_energy, _ = self.energy(
                data,
                branchs=branchs,
                changed=(cp, l1, l2),
            )
            result[k] = (new_energy - energy) / delta
        return result

    def choice(self, p, rho):
        for i, r in enumerate(rho):
            p -= r
            if p < 0:
                return i
        return i


class AbstractHoleSystem(AbstractSystem):

    def __init__(self, L1, L2, Dc, Tensor):
        super(AbstractHoleSystem, self).__init__(L1, L2, Dc, Tensor)
        self._construct_auxiliaries()

    def __setstate__(self, state):
        super().__setstate__(state)

    def _construct_auxiliaries(self):
        self.auxiliaries = Auxiliaries(self.L1 * 2 + 1, self.L2, self.Dc, False,
                                       self.Tensor)
        for l1 in range(self.L1):
            for l2 in range(self.L2):
                tensor_node = lazy.Node(lambda x: x, self.tensor[l1][l2])
                # tensor will be replaced later
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
                self.auxiliaries._lattice[l1][l2].replace(tensor_node)

                transposed_node = lazy.Node(
                    lambda tensor: tensor.edge_rename({
                        "U": "D",
                        "D": "U"
                    }).conjugate(), self.auxiliaries._lattice[l1][l2])

                self.auxiliaries._lattice[2 * self.L1 -
                                          l1][l2].replace(transposed_node)
        for l2 in range(self.L2):
            d = self.d
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
                d = self.d
                self.auxiliaries[self.L1, i + 1] = self.Tensor(
                    ["U", "LU", "D", "LD"], [d, d, d, d]).identity({
                        ("U", "LU"), ("D", "LD")
                    }).merge_edge({"L": ["LU", "LD"]})

                this = self._collect_hole()

                d = self.d
                iden = self.Tensor(["U", "D"], [d, d]).identity({("U", "D")})
                self.auxiliaries[self.L1, j] = iden
                d = self.d
                iden = self.Tensor(["U", "D"], [d, d]).identity({("U", "D")})
                self.auxiliaries[self.L1, i] = iden
            if len(position) == 1:
                i = position[0]
                self.auxiliaries[self.L1, i] = hamiltonian.edge_rename({
                    "I0": "U",
                    "O0": "D"
                })

                this = self._collect_hole()

                d = self.d
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

    def grad_of_param(self):
        grad_of_tensor = self._grad_of_tensor()
        delta = 1e-5
        result = {}
        for k in self.parameter.param:
            modified = self._modified_tensor(k)
            original = {(l1, l2): self.tensor[l1][l2]() for l1, l2 in modified}
            self.parameter[k] += delta
            now = {(l1, l2): self.tensor[l1][l2]() for l1, l2 in modified}
            self.parameter[k] -= delta
            param_grad = 0
            for l1, l2 in modified:
                tensor_diff = (now[l1, l2] - original[l1, l2]) / delta
                param_grad += float(
                    tensor_diff.contract(grad_of_tensor[(l1, l2)],
                                         {(n, n) for n in tensor_diff.names}))
            result[k] = param_grad
        return result
