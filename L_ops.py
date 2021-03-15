#!/usr/bin/env python
# -*- coding: utf-8 -*-


def project(former, projector):
    return former.contract(projector, {("U", "I")}).edge_rename({"O": "U"})


def contract_hamiltonian(former, hamiltonian):
    """
    处理哈密顿量的前半部分
    """
    return former.contract(hamiltonian, {("U", "I1")}).edge_rename({"O1": "U"})


def trace_hamiltonian(former):
    """
    处理哈密顿量的后半部分
    """
    return former.trace({("U", "I2")}).edge_rename({"O2": "U"})


def left_down_corner(site, *, r_name):
    return site.shrink({"D": 0, "L": 0}).edge_rename({"R": r_name})


def left_up_corner(former, site, *, r_name):
    return former.contract(site.shrink({
        "D": 0,
        "L": 0
    }), {("U", "U")}).edge_rename({"R": r_name})


def left_edge_down_part(former, site, *, r_name, l_name):
    return former.contract(site.edge_rename({"L": l_name}),
                           {("U", "D")}).edge_rename({
                               "U": "U",
                               "R": r_name
                           })


def left_edge_up_part(former, site, *, r_name, l_name):
    return former.contract(site.edge_rename({"L": l_name}),
                           {("U", "U")}).edge_rename({
                               "D": "U",
                               "R": r_name
                           })


def down_edge(former, site, *, r_name):
    return former.contract(site.shrink({"D": 0}),
                           {(r_name, "L")}).edge_rename({
                               "U": "U",
                               "R": r_name
                           })


def down_part(former, site, *, r_name):
    return former.contract(site, {("U", "D"), (r_name, "L")}).edge_rename({
        "U":
        "U",
        "R":
        r_name
    })


def down_part_tail(former, site, *, r_name, parity):
    return former.contract(site.shrink({"R": parity}),
                           {("U", "D"), (r_name, "L")}).edge_rename({
                               "U": "U",
                               "R": r_name
                           })


def down_part_tail_depth_1(former, site, *, r_name, parity):
    return former.contract(site.shrink({"R": parity, "D": 0}), {(r_name, "L")})


def up_edge(former, site, *, r_name):
    return former.contract(site.shrink({"D": 0}),
                           {("U", "U"),
                            (r_name, "L")}).edge_rename({"R": r_name})


def up_part(former, site, *, r_name):
    return former.contract(site, {("U", "U"), (r_name, "L")}).edge_rename({
        "D":
        "U",
        "R":
        r_name
    })


def up_part_tail(former, site, *, r_name, parity):
    return former.contract(site.shrink({"R": parity}),
                           {("U", "U"), (r_name, "L")}).edge_rename({
                               "D": "U",
                               "R": r_name
                           })


def up_part_tail_depth_1(former, site, *, r_name, parity):
    return former.contract(site.shrink({
        "R": parity,
        "D": 0
    }), {("U", "U"), (r_name, "L")})
