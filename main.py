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
from matrix_A import matrix_A
from matrix_U import matrix_U

Tensor = TAT(complex)


def get_U(n, c, r, omega, phi, psi):
    """
    根据矩阵截断, 物理截断, r, omega, phi, psi创建U矩阵
    """
    result = Tensor(["I1", "I2", "O1", "O2"], [n, n, n, n])
    result.block[{}] = matrix_U(n, c, r, omega, phi, psi)
    return result


def get_A(n, c, k, eta):
    """
    根据矩阵截断, 物理截断, k, eta创建A矩阵
    """
    result = Tensor(["O", "I"], [n, n])
    result.block[{}] = matrix_A(n, c, k, eta)
    return result


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
        # mpo的第t个格点按照最左端的规则重命名并收缩两个mpo的第t个格点
        if As[t] != As[t - 1] or Bs[t] != Bs[t - 1]:
            this = As[t].edge_rename({
                "L": "A.L",
                "R": "A.R"
            }).contract(Bs[t].edge_rename({
                "L": "B.L",
                "R": "B.R"
            }), {("U", "D"), ("D", "U")})
        # 收缩到左边
        result = result.contract(this, {("A.R", "A.L"), ("B.R", "B.L")})
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
    f = trace_ab / (trace_a * trace_b)
    return complex(f).real


def build_from_UU(l, UU, n):
    # 应该在这里shrink, 为了加速在create_UU中shrink
    # UU_with_I2_0 = UU.shrink({"UI2": 0, "DI2": 0})
    UU_with_I2_0 = UU
    result = []
    result.append(
        UU_with_I2_0.shrink({
            "UI1": 0,
            "DI1": 0
        }).edge_rename({
            "UO1": "U",
            "DO1": "D"
        }).merge_edge({"R": ["UO2", "DO2"]}))
    middle_site = UU_with_I2_0.edge_rename({
        "UO1": "U",
        "DO1": "D"
    }).merge_edge({
        "L": ["UI1", "DI1"],
        "R": ["UO2", "DO2"]
    })
    for t in range(1, l):
        result.append(middle_site)
    result.append(
        Tensor(["UL", "DL", "U", "D"], [n, n, n, n]).identity({
            ("UL", "U"), ("DL", "D")
        }).merge_edge({"L": ["UL", "DL"]}))
    return result


def create_UU(n, c, r, omega, phi, psi):
    U = get_U(n, c, r, omega, phi, psi)
    # 应该在create_from_UU中shrink为了加速在这里提前shrink
    U = U.shrink({"I2": 0})
    UU = U.edge_rename({
        "I1": "UI1",
        "O1": "UO1",
        "I2": "UI2",
        "O2": "UO2"
    }).conjugate().contract(
        U.edge_rename({
            "I1": "DI1",
            "O1": "DO1",
            "I2": "DI2",
            "O2": "DO2"
        }), set())
    return UU


def uniform(mean, delta):
    if delta is None:
        return [mean]
    else:
        sample = 5
        return [
            mean * (1 + delta * i / sample)
            for i in range(-sample, sample + 1)
        ]


def build_chain(l,
                n,
                c,
                r,
                omega,
                phi,
                psi,
                delta_r=None,
                delta_omega=None,
                delta_phi=None,
                delta_psi=None):
    UUs = []
    for real_r in uniform(r, delta_r):
        for real_omega in uniform(omega, delta_omega):
            for real_phi in uniform(phi, delta_phi):
                for real_psi in uniform(psi, delta_psi):
                    UUs.append(
                        create_UU(n, c, real_r, real_omega, real_phi,
                                  real_psi))
    UU = sum(UUs) / len(UUs)
    return build_from_UU(l, UU, n)


def create_kraus(n, c, eta):
    Aks = [get_A(n, c, k, eta) for k in range(n)]
    kraus = sum([
        Ak.edge_rename({
            "I": "UI",
            "O": "UO"
        }).conjugate().contract(Ak.edge_rename({
            "I": "DI",
            "O": "DO"
        }), set()) for Ak in Aks
    ])
    return kraus.merge_edge({"L": ["UI", "DI"], "R": ["UO", "DO"]})


def add_kraus(As, sites, kraus):
    last_As = None
    for i in sites:
        if last_As == As[i]:
            As[i] = As[j]
        else:
            last_As = As[i]
            j = i
            As[i] = As[i].contract(kraus, {("R", "L")})


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


def 截断收敛性(l, n, c, r, omega, phi, psi):
    As = build_chain(l, n, c, r, omega, phi, psi)
    Bs = build_chain(l, n, n, r, omega, phi, psi)
    print(fidelity(l + 1, As, Bs))


def 误差大小(l,
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
         post_last=None):
    As = build_chain(l, n, n, r, omega, phi, psi)
    Bs = build_chain(l,
                     n,
                     n,
                     r,
                     omega,
                     phi,
                     psi,
                     delta_r=delta_r,
                     delta_omega=delta_omega,
                     delta_phi=delta_phi,
                     delta_psi=delta_psi)
    if eta != 1:
        add_kraus(Bs, range(l - 1), create_kraus(n, n, eta))
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
    print(fidelity(l + 1, As, Bs))


def 后选择成功率(l, n, r, omega, phi, psi, post, post_last=None):
    As = build_chain(l, n, n, r, omega, phi, psi)
    Bs = build_chain(l, n, n, r, omega, phi, psi)
    add_post(Bs, range(l), create_post(n, post))
    if post_last is None:
        add_post(Bs, [l], create_post(n, post))
    else:
        add_post(Bs, [l], create_post(n, selected=post_last))
    for i in range(l + 1):
        As[i] /= As[i].norm_max()
        Bs[i] /= Bs[i].norm_max()
    # 两个密度矩阵的fidelity = trace(a@b)
    # 由于没有归一化, 所以一fidelity = trace(a@b) / (trace(a)trace(b))
    trace_a = trace_one(l + 1, As)
    trace_b = trace_one(l + 1, Bs)
    f = trace_b / trace_a
    print(complex(f).real)


if __name__ == "__main__":
    import fire

    def Display(lines, out):
        text = "\n".join(lines) + "\n"
        out.write(text)

    fire.core.Display = Display
    fire.Fire()
