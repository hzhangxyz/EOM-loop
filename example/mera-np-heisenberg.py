import pickle
import TAT
from han.systems.mera_eom import Mera_EOM
from han.systems.heisenberg import Heisenberg


class Mera_Heisenberg(Heisenberg, Mera_EOM):
    pass


def create(file_name, layer, D, Dc, seed):
    lattice = Mera_Heisenberg(layer=layer, D=D, Dc=Dc, Tensor=TAT.No.D.Tensor)

    TAT.random.seed(seed)
    uni1 = TAT.random.uniform_real(-1, +1)
    uni2 = TAT.random.uniform_real(-2, +2)
    unipi = TAT.random.uniform_real(-3.14, +3.14)

    LP = 1
    for l1 in range(lattice.L1):
        if l1 % 2 == 0:
            if l1 != 0:
                LP *= 2
        else:
            LP += 1
        for lp in range(LP):
            lattice.parameter[l1, lp, "r"] = uni2()
            lattice.parameter[l1, lp, "omega"] = unipi()

    with open(file_name, "wb") as file:
        pickle.dump(lattice, file)


def update(file_name, count, step):
    with open(file_name, "rb") as file:
        lattice = pickle.load(file)
    for t in range(count):
        lattice.refresh_auxiliaries()
        a, b = lattice._energies()
        print(t, a / b / lattice.L2)
        with open(file_name.replace(".dat", "") + ".log", "a") as file:
            print(t, a / b / lattice.L2, file=file)
        gp = lattice._grad_of_param()
        for k in gp:
            g = gp[k]
            if k[2] == "r":
                if lattice.parameter[k] >= +2 and g < 0:
                    g = 0
                if lattice.parameter[k] <= -2 and g > 0:
                    g = 0
            lattice.parameter[k] -= float(step) * g
        with open(file_name, "wb") as file:
            lattice.auxiliaries = None
            pickle.dump(lattice, file)


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: out.write("\n".join(lines) + "\n")
    fire.Fire()
