import pickle
import TAT
from han.systems.abstract_system import AbstractHoleSystem, r_bound
from han.systems.mps_eom_with_x4_post import MPS_EOM_with_x4_post
from han.systems.ising import Ising


class MPS_Ising(Ising, MPS_EOM_with_x4_post, AbstractHoleSystem):
    pass


def create(file_name, depth, length, D, Dc, seed):
    lattice = MPS_Ising(depth=depth,
                             length=length,
                             D=D,
                             Dc=Dc,
                             Tensor=TAT.No.D.Tensor)

    lattice.generate_initial_state(seed)
    with open(file_name, "wb") as file:
        pickle.dump(lattice, file)


def show(file_name):
    with open(file_name, "rb") as file:
        lattice = pickle.load(file)
    for k, v in lattice.parameter.param.items():
        print(k, v())


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
            lattice.parameter[k] -= float(step) * gp[k]
        lattice.refine_parameters()
        with open(file_name, "wb") as file:
            pickle.dump(lattice, file)


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: out.write("\n".join(lines) + "\n")
    fire.Fire()
