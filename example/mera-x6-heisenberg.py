import pickle
import TAT
from han.systems.abstract_system import AbstractHoleSystem
from han.systems.mera_eom_with_x6_post import Mera_EOM_with_x6_post
from han.systems.heisenberg import Heisenberg


class Mera_Heisenberg(Heisenberg, Mera_EOM_with_x6_post, AbstractHoleSystem):
    pass


def create(file_name, layer, D, Dc, seed):
    lattice = Mera_Heisenberg(layer=layer, D=D, Dc=Dc, Tensor=TAT.No.D.Tensor)

    lattice.generate_initial_state(seed)
    with open(file_name, "wb") as file:
        pickle.dump(lattice, file)


def update(file_name, count, step):
    with open(file_name, "rb") as file:
        lattice = pickle.load(file)
    for t in range(count):
        a, b = lattice._energies()
        print(t, a / b / lattice.L2)
        with open(file_name.replace(".dat", "") + ".log", "a") as file:
            print(t, a / b / lattice.L2, file=file)
        gp = lattice.grad_of_param()
        for k in gp:
            g = gp[k]
            if k[2] == "r":
                if lattice.parameter[k] >= +2 and g < 0:
                    g = 0
                if lattice.parameter[k] <= -2 and g > 0:
                    g = 0
            lattice.parameter[k] -= float(step) * g
        with open(file_name, "wb") as file:
            pickle.dump(lattice, file)


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: out.write("\n".join(lines) + "\n")
    fire.Fire()
