import pickle
import TAT
from han.systems.abstract_system import AbstractSamplingSystem
from han.systems.mps_eom_with_x4_post import MPS_EOM_with_x4_post
from han.systems.heisenberg import Heisenberg


class MPS_Heisenberg(Heisenberg, MPS_EOM_with_x4_post, AbstractSamplingSystem):
    pass


def create(file_name, depth, length, D, Dc, seed):
    lattice = MPS_Heisenberg(depth=depth,
                             length=length,
                             D=D,
                             Dc=Dc,
                             Tensor=TAT.No.D.Tensor)
    lattice.generate_initial_state(seed)
    with open(file_name, "wb") as file:
        pickle.dump(lattice, file)


def update(file_name, count, step, sampling, seed):
    with open(file_name, "rb") as file:
        lattice = pickle.load(file)
    TAT.random.seed(seed)
    gen01 = TAT.random.uniform_real(0, 1)
    for t in range(count):
        ss = lattice.get_configurations(gen01, sampling)
        e = lattice.energy(ss)
        print(t, e / lattice.L2)
        with open(file_name.replace(".dat", "") + ".log", "a") as file:
            print(t, e / lattice.L2, file=file)
        gp = lattice.grad_of_param(ss, e)
        for k in gp:
            lattice.parameter[k] -= float(step) * gp[k]
        with open(file_name, "wb") as file:
            pickle.dump(lattice, file)


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: out.write("\n".join(lines) + "\n")
    fire.Fire()
