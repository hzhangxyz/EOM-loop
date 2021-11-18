import pickle
import TAT
from han.systems.mera_eom import Mera_EOM
from han.systems.inner_product import inner_product


def fidelity(r, layer, D1, D2, Dc):
    psi1 = Mera_EOM(layer=layer, D=D1, Dc=Dc, Tensor=TAT.No.D.Tensor)
    psi1.d = 2
    psi2 = Mera_EOM(layer=layer, D=D2, Dc=Dc, Tensor=TAT.No.D.Tensor)
    psi2.d = 2

    LP = 1
    for l1 in range(psi1.L1):
        if l1 % 2 == 0:
            if l1 != 0:
                LP *= 2
        else:
            LP += 1
        for lp in range(LP):
            psi1.parameter[l1, lp, "r"] = r
            psi2.parameter[l1, lp, "r"] = r
            psi1.parameter[l1, lp, "omega"] = 0
            psi2.parameter[l1, lp, "omega"] = 0

    psi1psi2 = inner_product(psi1, psi2)
    psi1psi1 = inner_product(psi1, psi1)
    psi2psi2 = inner_product(psi2, psi2)

    result = (psi1psi2 * psi1psi2) / (psi1psi1 * psi2psi2)
    return result


def main(r, layer, D1, D2):
    Dc = 10
    last = None
    while True:
        result = fidelity(r, layer, D1, D2, Dc)
        print(Dc, "%.12f" % result)
        if last is not None:
            if abs(last - result) < 1e-12:
                return
        if Dc == 64:
            return
        last = result
        Dc += 1


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: out.write("\n".join(lines) + "\n")
    fire.Fire(main)
