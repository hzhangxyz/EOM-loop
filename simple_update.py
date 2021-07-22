import TAT
from hamiltonian import get_H
import numpy as np
import sys

Tensor = TAT(float)

D = 6

H = get_H()
U = (-0.01 * H).exponential({("I1", "O1"), ("I2", "O2")})
print(U)

A = Tensor(["P", "L", "R"], [2, D, D]).randn()
B = Tensor(["P", "L", "R"], [2, D, D]).randn()
EAB = None
EBA = None

for t in range(10000):
    for i in range(2):
        if i == 0:
            this_A = A
            this_B = B
            this_EAB = EAB
            this_EBA = EBA
        else:
            this_A = B
            this_B = A
            this_EAB = EBA
            this_EBA = EAB
        big = this_A
        big = big.edge_rename({"P": "PA"})
        if this_EAB:
            big = big.multiple(this_EAB, "R", 'u')
        big = big.contract(this_B, {("R", "L")})
        big = big.edge_rename({"P": "PB"})
        if this_EBA:
            big = big.multiple(this_EBA, "L", 'v')
            big = big.multiple(this_EBA, "R", 'u')
        big = big.contract(U, {("PA", "I1"), ("PB", "I2")})
        u, s, v = big.svd({"L", "O1"}, "R", "L", D)
        this_EAB = s / s.norm_max()
        this_A = u.edge_rename({"O1": "P"})
        this_B = v.edge_rename({"O2": "P"})
        if this_EBA:
            this_A = this_A.multiple(this_EBA, "L", 'v', True)
            this_B = this_B.multiple(this_EBA, "R", 'u', True)
        if i == 0:
            A = this_A
            B = this_B
            EAB = this_EAB
            EBA = this_EBA
        else:
            B = this_A
            A = this_B
            EBA = this_EAB
            EAB = this_EBA
        print(-np.log(s.norm_max()) / 0.01)

from get_energy import get_lattice_energy

print("!!!")

AE = A.multiple(EAB, "R", 'u').edge_rename({"P": "U"})
BE = B.multiple(EBA, "R", 'u').edge_rename({"P": "U"})

shrinker = Tensor(["I"], [D]).randn()
ls = [10, 30, 100, 300, 1000]
for l in ls:

    def get_site(length, depth):
        if length % 2 == 0:
            res = AE
        else:
            res = BE
        if length == 0:
            res = res.contract(shrinker, {("L", "I")})
        if length == l - 1:
            res = res.contract(shrinker, {("R", "I")})
        return res

    E1 = get_lattice_energy(l, 1, get_site, H, l // 2 - 1, 100, 100)
    E2 = get_lattice_energy(l, 1, get_site, H, l // 2, 100, 100)
    print(l, (E1 + E2) / 2)
