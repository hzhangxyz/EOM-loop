#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tools import gradient, save_to_file


def axis_descent(chain, file_name, config):

    delta = float(config[0])

    def update_once():
        old_E = chain.energy()
        for i in range(chain.depth * 2):
            for d in [-delta, +delta]:
                xs = chain.get_value()
                xss = xs[:]
                xss[i] += d
                new_E = chain.set_value(xss).energy()
                if new_E > old_E:
                    chain.set_value(xs)

        chain.save_to_file(file_name)
        print(chain.energy())

    if len(config) != 1:
        count = int(config[1])
        for _ in range(count):
            update_once()
    else:
        while True:
            update_once()


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

        chain.save_to_file(file_name)
        print(chain.energy())

    if len(config) != 1:
        count = int(config[1])
        for _ in range(count):
            update_once()
    else:
        while True:
            update_once()


def normal_gradient(chain, file_name, config):
    delta = float(config[0])
    global last_energy
    last_energy = 100

    def update_once():
        chain.set_value([
            x - delta * g for x, g in zip(chain.get_value(), chain.gradient())
        ])
        e = chain.energy()
        global last_energy
        if e < last_energy or True:
            last_energy = e
            print(e)
        else:
            print(e)
            print("!!!")
            exit(-1)
        chain.save_to_file(file_name)

    if len(config) != 1:
        count = int(config[1])
        for _ in range(count):
            update_once()
    else:
        while True:
            update_once()


def calculate_energy(lattice, file_name, config):
    save_to_file(lattice, file_name)
    print(lattice.energy())


def energy_calculator(chain, file_name, config):
    with open(file_name, "w") as file:
        x = float(config[0])
        y = float(config[1])
        print(x, y)
        i = -4
        while i <= +4:
            j = -4
            while j <= +4:
                chain.set_value([x, y, i, j])
                print(i, j, chain.energy(), file=file)
                j += 0.1
            i += 0.1


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
        print(chain.energy())

    if len(config) != 2:
        count = int(config[2])
        for _ in range(count):
            update_once()
    else:
        while True:
            update_once()


def search_nearest(chain, file_name, config):
    size = float(config[0])
    count = int(config[1])
    direction = int(config[2])

    x = chain.get_value()
    #print(x)
    #print(chain.gradient())
    print()

    def try_x(delta):
        xn = x[:]
        xn[direction] += delta
        chain.set_value(xn)
        print(delta, chain.energy())

    for i in range(count):
        try_x(-size / 2**i)
    try_x(0)
    for i in range(count):
        try_x(size / 2**i)


def search_nearest_direction(chain, file_name, config):
    from math import sqrt

    size = float(config[0])
    count = int(config[1])

    xs = chain.get_value()
    gs = chain.gradient()
    gs_norm = sqrt(sum(i * i for i in gs))
    print()

    def try_x(delta):
        chain.set_value([x + delta * g / gs_norm for x, g in zip(xs, gs)])
        print(delta, chain.energy())

    for i in range(count):
        try_x(-size / 2**i)
    try_x(0)
    for i in range(count):
        try_x(size / 2**i)


def sample_line_search(chain, file_name, config):
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
        for i in range(len(chain.get_value())):

            print(i)
            cx = chain.get_value()
            begin = [
                x - delta * (1 if j == i else 0) for j, x in enumerate(cx)
            ]
            end = [x + delta * (1 if j == i else 0) for j, x in enumerate(cx)]
            e_begin = chain.set_value(begin).energy()
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


def beyesian_opt(chain, file_name, config):
    from skopt import gp_minimize
    from skopt.learning import GaussianProcessRegressor

    def get_energy(x):
        chain.set_value(x)
        e = chain.energy()
        return e

    x0 = chain.get_value()

    def callback_function(state, best=[[], +1000]):
        e = chain.set_value(state.x).energy()
        if e < best[1]:
            best[1] = e
            best[0] = state.x
        print(chain.energy())
        chain.set_value(best[0]).save_to_file(file_name)

    space = [(-2., +2.) if i % 2 == 0 else (-6., +6.)
             for i, j in enumerate(x0)]

    res = gp_minimize(
        get_energy,
        space,
        x0=x0,
        # base_estimator = GaussianProcessRegressor(),
        acq_func="EI",
        n_calls=500,
        n_initial_points=100,
        callback=callback_function)
    callback_function(res)

    from skopt.plots import plot_convergence
    plot_convergence(res)
    from matplotlib import pyplot as plt
    plt.show()


def scipy_optimize(chain, file_name, config):
    method = config[0]

    import numpy as np
    from scipy.optimize import minimize

    def get_e_and_g(x):
        chain.set_value(x)
        e = chain.energy()
        g = gradient(chain)
        return e, np.array(g)

    def callback_function(x):
        print(chain.energy())
        save_to_file(chain, file_name)

    minimize(get_e_and_g,
             chain.get_value(),
             method=method,
             jac=True,
             options={"disp": True},
             tol=float(config[1]),
             callback=callback_function)
