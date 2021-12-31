import pickle
from mpi4py import MPI
import TAT
from han.systems.abstract_system import AbstractSamplingSystem, r_bound
from han.systems.mera_eom_with_x6_post import Mera_EOM_with_x6_post
from han.systems.heisenberg import Heisenberg

rank = MPI.COMM_WORLD.Get_rank()


class Mera_Heisenberg(Heisenberg, Mera_EOM_with_x6_post,
                      AbstractSamplingSystem):
    pass


def create(file_name, layer, D, Dc, seed):
    lattice = Mera_Heisenberg(layer=layer, D=D, Dc=Dc, Tensor=TAT.No.D.Tensor)
    lattice.generate_initial_state(seed)
    with open(file_name, "wb") as file:
        pickle.dump(lattice, file)


def show(file_name):
    with open(file_name, "rb") as file:
        lattice = pickle.load(file)
    for k, v in lattice.parameter.param.items():
        print(k, v())


def update(file_name, count, step, sampling, seed):
    with open(file_name, "rb") as file:
        lattice = pickle.load(file)
    TAT.random.seed(seed + rank)
    gen01 = TAT.random.uniform_real(0, 1)
    for t in range(count):
        ss = lattice.get_configurations(gen01, sampling)
        e, branchs = lattice.energy(ss)
        gp = lattice.grad_of_param(ss, e, branchs)

        for k in gp:
            lattice.parameter[k] -= float(step) * gp[k]
        lattice.refine_parameters()
        if rank == 0:
            print(t, e / lattice.L2)
            with open(file_name.replace(".dat", "") + ".log", "a") as file:
                print(t, e / lattice.L2, file=file)
            with open(file_name, "wb") as file:
                pickle.dump(lattice, file)


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: out.write("\n".join(lines) + "\n")
    fire.Fire()
