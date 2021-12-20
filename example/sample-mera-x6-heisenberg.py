import pickle
from mpi4py import MPI
import TAT
from han.systems.abstract_system import AbstractSamplingSystem
from han.systems.mera_eom_with_x6_post import Mera_EOM_with_x6_post
from han.systems.heisenberg import Heisenberg


class Mera_Heisenberg(Heisenberg, Mera_EOM_with_x6_post,
                      AbstractSamplingSystem):
    pass


def create(file_name, layer, D, Dc, seed):
    lattice = Mera_Heisenberg(layer=layer, D=D, Dc=Dc, Tensor=TAT.No.D.Tensor)
    lattice.generate_initial_state(seed)
    with open(file_name, "wb") as file:
        pickle.dump(lattice, file)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def _sum_dict(d1, d2):
    return {k: d1[k] + d2[k] for k in d1}


def _mpi_mean(gp):
    result = comm.allreduce(gp, _sum_dict)
    return {k: v / size for k, v in result.items()}


def update(file_name, count, step, sampling, seed):
    with open(file_name, "rb") as file:
        lattice = pickle.load(file)
    TAT.random.seed(seed + rank)
    gen01 = TAT.random.uniform_real(0, 1)
    for t in range(count):
        ss = lattice.get_configurations(gen01, sampling)
        e, branchs = lattice.energy(ss)
        gp = lattice.grad_of_param(ss, e, branchs)
        gp["e"] = e

        gp = _mpi_mean(gp)

        for k in gp:
            if k != "e":
                g = gp[k]
                if k[2] == "r":
                    if lattice.parameter[k] >= +2 and g < 0:
                        g = 0
                    if lattice.parameter[k] <= -2 and g > 0:
                        g = 0
                lattice.parameter[k] -= float(step) * g
        if rank == 0:
            print(t, gp["e"] / lattice.L2)
            with open(file_name.replace(".dat", "") + ".log", "a") as file:
                print(t, gp["e"] / lattice.L2, file=file)
            with open(file_name, "wb") as file:
                pickle.dump(lattice, file)


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: out.write("\n".join(lines) + "\n")
    fire.Fire()
