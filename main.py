#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2021 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

import tools

Tensor = TAT(complex)


def trace_one(length, As):
    """
    收缩一个长度为length的mpo表示的密度矩阵的迹
    """
    # mpo最左边的格点上下求迹
    result = As[0].trace({("U", "D")})
    for t in range(1, length):
        # 第t个格点上下求迹
        this = As[t].trace({("U", "D")})
        # 第t个格点收缩到左边去
        result = result.contract(this, {("R", "L")})
    return result


def trace_two(length, As, Bs):
    """
    收缩两个长度为length的mpo表示的密度矩阵的乘积的迹
    """
    # mpo最左端的右侧重命名为A.R, 另一个重命名为B.R, 他们的U和D相互缩并
    result = As[0].edge_rename({
        "R": "A.R"
    }).contract(Bs[0].edge_rename({"R": "B.R"}), {("U", "D"), ("D", "U")})
    for t in range(1, length):
        # 先缩并掉相同位置的AB两个chain的site的方式比较慢
        result = result.contract(As[t].edge_rename({
            "L": "A.L",
            "R": "A.R"
        }), {("A.R", "A.L")})
        result = result.contract(Bs[t].edge_rename({
            "L": "B.L",
            "R": "B.R"
        }), {("B.R", "B.L"), ("D", "U"), ("U", "D")})
    return result


def fidelity(length, As, Bs):
    # 求迹前先归一化mpo
    for i in range(length):
        As[i] /= As[i].norm_max()
        Bs[i] /= Bs[i].norm_max()
    # 两个密度矩阵的fidelity = trace(a@b)
    # 由于没有归一化, 所以一fidelity = trace(a@b) / (trace(a)trace(b))
    trace_a = trace_one(length, As)
    trace_b = trace_one(length, Bs)
    trace_ab = trace_two(length, As, Bs)
    # print(trace_a, trace_b, trace_ab)
    f = trace_ab / (trace_a * trace_b)
    return complex(f).real


def build_chain(l, UU, n):
    if "UI2" in UU.name:
        UU_with_I2_0 = UU.shrink({"UI2": 0, "DI2": 0})
    else:
        UU_with_I2_0 = UU
    result = []
    result.append(
        UU_with_I2_0.shrink({
            "UI1": 0,
            "DI1": 0
        }).edge_rename({
            "UO2": "U",
            "DO2": "D"
        }).merge_edge({"R": ["UO1", "DO1"]}))
    middle_site = UU_with_I2_0.edge_rename({
        "UO2": "U",
        "DO2": "D"
    }).merge_edge({
        "L": ["UI1", "DI1"],
        "R": ["UO1", "DO1"]
    })
    for t in range(1, l):
        result.append(middle_site)
    result.append(
        Tensor(["UL", "DL", "U", "D"], [n, n, n, n]).identity({
            ("UL", "U"), ("DL", "D")
        }).merge_edge({"L": ["UL", "DL"]}))
    return result


def create_post(n, c=None, selected=None):
    result = Tensor(["U", "D"], [n, n]).zero()
    if c is not None:
        if selected is not None:
            pass
        else:
            for i in range(c):
                result[{"U": i, "D": i}] = 1
    else:
        if selected is not None:
            for i in selected:
                result[{"U": i, "D": i}] = 1
        else:
            pass
    return result


def add_post(As, sites, post):
    last_As = None
    for i in sites:
        if last_As == As[i]:
            As[i] = As[j]
        else:
            last_As = As[i]
            j = i
            As[i] = As[i].contract(post,
                                   {("U", "D")}).contract(post, {("D", "U")})


def add_parity(As, n, length, parity, only_up=False):
    P = Tensor(["P", "L", "R"], [n, 2, 2])
    data_P = P.block[["P", "L", "R"]]
    for p in range(n):
        if p % 2 == 0:
            data_P[p] = [[1, 0], [0, 1]]
        else:
            data_P[p] = [[0, 1], [1, 0]]
    last_As = None
    for t in range(length + 1):
        if t != 0 and t != length and last_As == As[t]:
            As[t] = As[t - 1]
        else:
            Pt = P
            merge_map = {"L": ["AL", "PL"], "R": ["AR", "PR"]}
            if t == 0:
                del merge_map["L"]
                Pt = Pt.shrink({"L": 0})
            if t == length:
                del merge_map["R"]
                if parity == 1:
                    Pt = Pt.shrink({"R": 1})
                elif parity == 0:
                    Pt = Pt.shrink({"R": 0})
                else:
                    raise RuntimeError("Invalid parity")
            last_As = As[t]
            As[t] = As[t].edge_rename({
                "L": "AL",
                "R": "AR"
            }).contract(Pt.edge_rename({
                "L": "PL",
                "R": "PR",
                "P": "U"
            }), set()).merge_edge(merge_map)
            if not only_up:
                As[t] = As[t].edge_rename({
                    "L": "AL",
                    "R": "AR"
                }).contract(Pt.edge_rename({
                    "L": "PL",
                    "R": "PR",
                    "P": "D"
                }), set()).merge_edge(merge_map)


def cutoff_convergence(l, n, c, r, omega, phi, psi):
    As = build_chain(l,
                     tools.tensor_UU(n, c, r, omega, phi, psi, shrink_I2=True),
                     n)
    Bs = build_chain(l,
                     tools.tensor_UU(n, n, r, omega, phi, psi, shrink_I2=True),
                     n)
    print(fidelity(l + 1, As, Bs))


def exact_simulation(l, n, r, omega, phi, psi):
    As = build_chain(l,
                     tools.tensor_UU(n, n, r, omega, phi, psi, shrink_I2=True),
                     n)
    result = As[0].edge_rename({"U": "U0", "D": "D0"})
    for t in range(1, l + 1):
        result = result.contract(
            As[t].edge_rename({
                "U": "U" + str(t),
                "D": "D" + str(t)
            }), {("R", "L")})
    U, S, V = result.svd({"U" + str(i) for i in range(l + 1)}, "P", "P", cut=1)
    print(U.shrink({"P": 0}))


def error_convergence(l,
                      n,
                      r,
                      omega,
                      phi,
                      psi,
                      delta_r=None,
                      delta_omega=None,
                      delta_phi=None,
                      delta_psi=None,
                      eta=1,
                      post=None,
                      post_last=None,
                      filter_parity=None):
    As = build_chain(l,
                     tools.tensor_UU(n, n, r, omega, phi, psi, shrink_I2=True),
                     n)
    Bs = build_chain(
        l,
        tools.tensor_UUAA(n,
                          n,
                          r,
                          omega,
                          phi,
                          psi,
                          delta_r,
                          delta_omega,
                          delta_phi,
                          delta_psi,
                          eta_1=eta,
                          eta_2=eta,
                          shrink_I2=True), n)
    Cs = Bs[:]
    if post is not None:
        projection = create_post(n, post)
        add_post(As, range(l), projection)
        add_post(Bs, range(l), projection)
        if post_last is None:
            add_post(As, [l], projection)
            add_post(Bs, [l], projection)
    if post_last is not None:
        projection = create_post(n, selected=post_last)
        add_post(As, [l], projection)
        add_post(Bs, [l], projection)
    if filter_parity is not None:
        add_parity(As, n, l, filter_parity, only_up=True)
        add_parity(Bs, n, l, filter_parity, only_up=True)
    print("Fidelity is", fidelity(l + 1, As, Bs))

    for i in range(l + 1):
        Bs[i] /= Bs[i].norm_max()
        Cs[i] /= Cs[i].norm_max()
    # 两个密度矩阵的fidelity = trace(a@b)
    # 由于没有归一化, 所以一fidelity = trace(a@b) / (trace(a)trace(b))
    trace_b = trace_one(l + 1, Bs)
    trace_c = trace_one(l + 1, Cs)
    f = trace_b / trace_c
    print("Post Selection Rate is", complex(f).real)


if __name__ == "__main__":
    import fire

    def Display(lines, out):
        text = "\n".join(lines) + "\n"
        out.write(text)

    fire.core.Display = Display
    fire.Fire({
        "convergence": cutoff_convergence,
        "error": error_convergence,
        "exact": exact_simulation
    })
