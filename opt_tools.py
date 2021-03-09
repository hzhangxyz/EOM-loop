#!/usr/bin/env python
# -*- coding: utf-8 -*-


def random_gradient(chain, file_name, config):
    import random

    def sign(x):
        if x == 0:
            return 0
        elif x > 0:
            return 1
        else:
            return -1

    delta = float(config[0])

    def update_once():
        chain.set_value([
            x - delta * random.random() * sign(g)
            for x, g in zip(chain.get_value(), chain.gradient())
        ])
        print(chain.energy())
        with open(file_name, "w") as file:
            print(chain.length, chain.depth, 1, file=file)
            print(*chain.get_value(), file=file)
            print(chain.energy(), file=file)

    if len(config) != 1:
        count = int(config[1])
        for _ in range(count):
            update_once()
    else:
        while True:
            update_once()


def line_search(chain, file_name, config):
    from math import sqrt

    delta = float(config[0])
    search_count = int(config[1])

    def line_search_once(count, begin, e_begin, end, e_end):
        if count == search_count:
            if e_begin < e_end:
                chain.set_value(begin)
            else:
                chain.set_value(end)
            return
        point_1 = [(b * 2 + e) / 3 for b, e in zip(begin, end)]
        point_2 = [(b + e * 2) / 3 for b, e in zip(begin, end)]
        e_1 = chain.set_value(point_1).energy()
        e_2 = chain.set_value(point_2).energy()
        print(e_begin, e_1, e_2, e_end)
        if e_1 < e_2:
            line_search_once(count + 1, begin, e_begin, point_2, e_2)
        else:
            line_search_once(count + 1, point_1, e_1, end, e_end)

    def update_once():
        gs = chain.gradient()
        """
        psipsi = complex(chain.get_psiHpsi(0)).real
        if psipsi < 0.2:
            print("ALERT")
            psipsi_gs = chain.psipsi_gradient()
            ab = sum(a * b for a, b in zip(gs, psipsi_gs))
            bb = sum(b * b for b in psipsi_gs)
            gs = [g - p * ab / bb for g, p in zip(gs, psipsi_gs)]
        """

        gs_norm = sqrt(sum(i * i for i in gs))

        begin = chain.get_value()
        end = [
            x - delta * g / gs_norm for x, g in zip(begin, chain.gradient())
        ]
        e_begin = chain.energy()
        e_end = chain.set_value(end).energy()
        line_search_once(0, begin, e_begin, end, e_end)
        e_now = chain.energy()
        print(e_now)
        with open(file_name, "w") as file:
            print(chain.length, chain.depth, 1, file=file)
            print(*chain.get_value(), file=file)
            print(chain.energy(), file=file)

    if len(config) != 2:
        count = int(config[2])
        for _ in range(count):
            update_once()
    else:
        while True:
            update_once()