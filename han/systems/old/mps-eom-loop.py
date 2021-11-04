import numpy as np
import TAT
from tools import StorageFunction, read_from_file, save_to_file, tensor_U
import sys
import opt_tools
import random
from hamiltonian import get_H
from get_energy import get_energy

Tensor = TAT(float)

delta = 1e-5


@StorageFunction
def get_U(n, r, omega, shrink=tuple()):
    return tensor_U(n, n, r, omega, 0, 0).edge_rename({
        "I1": "L",
        "O1": "R",
        "O2": "U",
        "I2": "D"
    }).to(float).shrink({i: 0 for i in shrink})


def random_uniform(a, b):
    return random.random() * (b - a) + a


class MPS:

    def __init__(self, depth, length, cutoff, cut1, cut2):
        self.H = get_H()
        self.HH = self.H.contract(self.H, {("I1", "O1"), ("I2", "O2")})
        print("H", self.H)
        print("HH", self.HH)
        self.set_shape(depth, length, cutoff, cut1, cut2)

        self._energy = None

        self.set_diff(-1, -1)

    def set_shape(self, depth, length, cutoff, cut1, cut2):
        self.depth = depth
        self.length = length
        self.cutoff = cutoff
        self.cut1 = cut1
        self.cut2 = cut2

        self.parameter = [[[random_uniform(-2, +2),
                            random_uniform(-6, +6)]
                           for _ in range(self.depth)]
                          for _ in range(self.length)]
        self.projector = [[random_uniform(-1, +1)
                           for _ in range(8)]
                          for _ in range(self.length)]

    def get_shape(self):
        return self.depth, self.length, self.cutoff, self.cut1, self.cut2

    def get_shape_size(self):
        return 5

    def get_value_size(self):
        return 8 * self.length + 2 * self.length * self.depth

    def get_value(self):
        xs1 = np.array(self.projector).reshape([-1])  # L * 8
        xs2 = np.array(self.parameter).reshape([-1])  # L * D * 2
        return [*xs1, *xs2]

    def set_value(self, xs):
        self._energy = None
        xs = np.array(xs)
        xs1 = xs[:self.length * 8]
        xs2 = xs[self.length * 8:]
        self.projector = xs1.reshape([self.length, 8]).tolist()
        self.parameter = xs2.reshape([self.length, self.depth, 2]).tolist()
        return self

    def set_diff(self, down, up):
        self._diff_projector = [None, None]
        self._diff_parameter = [None, None]

        for it, index in enumerate([down, up]):
            xs = np.zeros(self.get_value_size())
            if index != -1:
                xs[index] = delta
            self._diff_projector[it] = xs[:self.length * 8].reshape(
                [self.length, 8]).tolist()
            self._diff_parameter[it] = xs[self.length * 8:].reshape(
                [self.length, self.depth, 2]).tolist()

    def __call__(self, *, depth, length):
        # length : 0 ~ L-1
        # depth : 0 ~ D-1
        # when depth >= D, edge name not changed
        down_not_up = True
        down_0_up_1 = 0
        if depth >= self.depth:
            down_not_up = False
            down_0_up_1 = 1
            depth = 2 * self.depth - depth - 1
        shrink = []
        if depth == 0:
            shrink.append("D")
        if length == 0:
            shrink.append("L")
        if length == self.length - 1:
            shrink.append("R")
        result = get_U(
            self.cutoff, self.parameter[length][depth][0] +
            self._diff_parameter[down_0_up_1][length][depth][0],
            self.parameter[length][depth][1] +
            self._diff_parameter[down_0_up_1][length][depth][1], tuple(shrink))
        if depth == self.depth - 1:
            projector = Tensor(["D", "U"], [self.cutoff, 2]).zero()
            projector.blocks[["D", "U"]][:4, :2] = (
                np.array(self.projector[length]) +
                np.array(self._diff_projector[down_0_up_1][length])).reshape(
                    [4, 2])
            result = result.contract(projector, {("U", "D")})
        return result

    def get_energies(self, H=None):
        if H is None:
            H = self.H
        totalE = 0
        totalpsiHpsi = 0
        for i in range(self.length - 1):
            # i and i + 1
            E, psiHpsi, psipsi = get_energy(self.length,
                                            self.depth,
                                            self,
                                            H,
                                            i,
                                            self.cut1,
                                            self.cut2,
                                            double_layer=True)
            totalE += E
            totalpsiHpsi += psiHpsi
        return (totalE / self.length, totalpsiHpsi / self.length, psipsi)

    def energy(self):
        if self._energy is None:
            self.set_diff(-1, -1)
            self._energy = self.get_energies()[0]
        return self._energy

    def norm_proj(self):
        n = np.max(np.abs(self.projector))
        print("norm proj norm", n)
        self.projector = (np.array(self.projector) / n).tolist()


handle = read_from_file(MPS, sys.argv[1])
if sys.argv[2] != "it":
    getattr(opt_tools, sys.argv[2])(handle, sys.argv[1], sys.argv[3:])
    exit()

delta_tau = float(sys.argv[3])
up_bond = 0.05

t = 0
total_time = 0.
while True:
    t += 1
    print("step", t)
    print("saving data...")
    save_to_file(handle, sys.argv[1])
    for apply_position in range(handle.length - 1):
        # apply_position and apply_position + 1
        def get_index(i):
            # (AB) * 8 + (AB) * depth * 2
            if i < 2 * 8:
                "P"
                if i < 8:
                    "A"
                    res = (apply_position - 1) * 8 + i
                else:
                    i -= 8
                    "B"
                    res = (apply_position) * 8 + i
            else:
                i -= 2 * 8
                "U"
                if i < handle.depth * 2:
                    "A"
                    res = (apply_position - 1) * (handle.depth * 2) + i
                else:
                    i -= handle.depth * 2
                    "B"
                    res = (apply_position) * (handle.depth * 2) + i
                res += handle.length * 8
            return res

        print("position", apply_position)
        print("norm proj...")
        handle.norm_proj()
        print("measure...")
        size = (2 * handle.depth + 8) * 2
        #handle.get_value_size() // handle.length
        psiHpsi = [None for i in range(size + 1)]
        psipsi = [[None for _ in range(size + 1)] for _ in range(size + 1)]
        energy, psiHpsi[size], psipsi[size][size] = handle.get_energies()
        print("psipsi", psipsi[size][size])
        print("energy", energy)
        _, psiHHpsi, _ = handle.get_energies(handle.HH)
        with open(sys.argv[1] + ".log", "a") as file:
            print(t, total_time, apply_position, handle.energy(), file=file)
        for i in range(size):
            print(i, end=" ", flush=True)
            handle.set_diff(get_index(i), -1)
            _, psiHpsi[i], psipsi[i][size] = handle.get_energies()
            handle.set_diff(-1, get_index(i))
            _, _, psipsi[size][i] = handle.get_energies()
        for i in range(size):
            for j in range(size):
                print("%d_%d" % (i, j), end=" ", flush=True)
                handle.set_diff(get_index(i), get_index(j))
                _, _, psipsi[i][j] = handle.get_energies()
        print("AC...")
        C = [(psiHpsi[size] - psiHpsi[i]) / delta for i in range(size)]
        A = [[(psipsi[i][j] + psipsi[size][size] - psipsi[i][size] -
               psipsi[size][j]) / (delta**2)
              for j in range(size)]
             for i in range(size)]
        C = np.array(C)
        A = np.array(A)
        print("solving...")
        real_xs = handle.get_value()
        xs = [real_xs[get_index(i)] for i in range(size)]
        xss, residuals, rank, s = np.linalg.lstsq(A, C * delta_tau)
        epsilon = 0.05
        max_singular = np.sort(s)[-1]
        print("max singular", max_singular)
        if np.linalg.norm(xss) > np.linalg.norm(xs) * epsilon:
            idm = np.identity(len(A))
            # find lam s.t. (A+lam id) xss = C delta_tau
            low = 0
            high = max_singular
            while True:
                xss, residuals, rank, s = np.linalg.lstsq(
                    A + high * idm, C * delta_tau)
                if np.linalg.norm(xss) > np.linalg.norm(xs) * epsilon:
                    low = high
                    high *= 2
                else:
                    break
            # between low and high
            while True:
                mid = (low + high) / 2
                xss, residuals, rank, s = np.linalg.lstsq(
                    A + mid * idm, C * delta_tau)
                if np.linalg.norm(xss) > np.linalg.norm(xs) * epsilon:
                    low = mid
                else:
                    high = mid
                if high - low < 1e-6 * max_singular:
                    lam = high
                    xss, residuals, rank, s = np.linalg.lstsq(
                        A + lam * idm, C * delta_tau)
                    break
        else:
            lam = 0
        print("lam", lam)
        print("xs", xs)
        print("C", C)
        print("A", A)
        print("xss", xss)
        print("apply...")

        for i in range(size):
            real_xs[get_index(i)] += xss[i]
        handle.set_value(real_xs)

    total_time += delta_tau
