#!/usr/bin/env python
# -*- coding: utf-8 -*-


def hole_hamiltonian(former, n):
    # return former.contract(hamiltonian, {("D", "O2")}).edge_rename({"I2": "D"})
    return former.edge_rename({
        "D": "O2"
    }).contract(
        type(former)(["I2", "D"], [n, n]).identity({("I2", "D")}), set())


def contract_hamiltonian(former, hamiltonian):
    # return former.trace({("D", "O1")}).edge_rename({"I1": "D"})
    return former.contract(hamiltonian,
                           {("D", "O1"), ("O2", "O2"),
                            ("I2", "I2")}).edge_rename({"I1": "D"})


def right_up_corner(site, *, l_name, r_name):
    return site.shrink({
        "D": 0
    }).edge_rename({
        "L": l_name,
        "R": r_name,
        "U": "D"
    })


def right_down_corner(former, site, *, l_name, r_name):
    return former.contract(site.shrink({"D": 0}), {("D", "U")}).edge_rename({
        "L":
        l_name,
        "R":
        r_name
    })


def right_edge_up_part(former, site, *, l_name, r_name):
    return former.contract(site.edge_rename({"R": r_name}),
                           {("D", "D")}).edge_rename({
                               "U": "D",
                               "L": l_name
                           })


def right_edge_down_part(former, site, *, l_name, r_name):
    return former.contract(site.edge_rename({"R": r_name}),
                           {("D", "U")}).edge_rename({
                               "D": "D",
                               "L": l_name
                           })


def right_edge_up_part_tail(former, site, *, l_name, r_name, parity):
    return former.contract(site.shrink({"R": parity}),
                           {("D", "D")}).edge_rename({
                               "U": "D",
                               "L": l_name
                           })


def right_edge_down_part_tail(former, site, *, l_name, r_name, parity):
    return former.contract(site.shrink({"R": parity}),
                           {("D", "U")}).edge_rename({
                               "D": "D",
                               "L": l_name
                           })


def up_edge(former, site, *, l_name):
    return former.contract(site.shrink({"D": 0}),
                           {(l_name, "R")}).edge_rename({
                               "U": "D",
                               "L": l_name
                           })


def left_up_corner(former, site, *, l_name):
    return former.contract(site.shrink({
        "D": 0,
        "L": 0
    }), {(l_name, "R")}).edge_rename({"U": "D"})


def up_part(former, site, *, l_name):
    return former.contract(site, {("D", "D"), (l_name, "R")}).edge_rename({
        "U":
        "D",
        "L":
        l_name
    })


def down_edge(former, site, *, l_name):
    return former.contract(site.shrink({"D": 0}),
                           {("D", "U"),
                            (l_name, "R")}).edge_rename({"L": l_name})


def left_down_corner(former, site, *, l_name):
    return former.contract(site.shrink({
        "D": 0,
        "L": 0
    }), {("D", "U"), (l_name, "R")})


def down_part(former, site, *, l_name):
    return former.contract(site, {("D", "U"), (l_name, "R")}).edge_rename({
        "D":
        "D",
        "L":
        l_name
    })
