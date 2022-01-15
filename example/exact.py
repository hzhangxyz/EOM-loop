import numpy as np
import torch
from TAT.No.D import Tensor
from han.utility.tensor_U import tensor_U

n = 4


def get_U(r, omega):
    result = tensor_U(n, n, r, omega, 0, 0).edge_rename({
        "I1": "L",
        "O1": "R",
        "O2": "D",
        "I2": "U"
    })
    return torch.tensor(result.to(float).blocks[["L", "R", "U", "D"]],
                        requires_grad=True)


parameter = np.random.randn(4, 2)
clas = [torch.randn(4, 4, requires_grad=True, dtype=float) for i in range(4)]

p40 = torch.tensor([1, 0, 0, 0], dtype=float)

# I0, I1, O0, O1
H = torch.zeros(2, 2, 2, 2, dtype=float)
H[0, 0, 0, 0] = 1 / 4.
H[0, 1, 0, 1] = -1 / 4.
H[1, 0, 1, 0] = -1 / 4.
H[1, 1, 1, 1] = 1 / 4.
H[1, 0, 0, 1] = 2 / 4.
H[0, 1, 1, 0] = 2 / 4.

# while True:
for _ in range(10000):
    quan = [get_U(*i) for i in parameter]

    psiq = torch.einsum("abcd,befg,ehij,hklm,a,c,f,i,l,k->dgjm", quan[0],
                        quan[1], quan[2], quan[3], p40, p40, p40, p40, p40, p40)

    clas_q = [
        torch.linalg.qr(c, mode="reduced")[0].reshape([4, 2, 2]) for c in clas
    ]
    psi = torch.einsum("abcd,aex,bfy,cgz,dhw->exfygzhw", psiq, clas_q[0],
                       clas_q[1], clas_q[2], clas_q[3])

    num1 = torch.einsum("xycdefgh,xyzw,zwcdefgh->", psi, H, psi)
    num2 = torch.einsum("axydefgh,xyzw,azwdefgh->", psi, H, psi)
    num3 = torch.einsum("abxyefgh,xyzw,abzwefgh->", psi, H, psi)
    num4 = torch.einsum("abcxyfgh,xyzw,abczwfgh->", psi, H, psi)
    num5 = torch.einsum("abcdxygh,xyzw,abcdzwgh->", psi, H, psi)
    num6 = torch.einsum("abcdexyh,xyzw,abcdezwh->", psi, H, psi)
    num7 = torch.einsum("abcdefxy,xyzw,abcdefzw->", psi, H, psi)
    # 1...7
    den = torch.einsum("abcdefgh,abcdefgh->", psi, psi)
    energy = (num1 + num2 + num3 + num4 + num5 + num6 + num7) / (den * 8)

    with open("data.log", "a") as file:
        print(float(energy), file=file)
    energy.backward()

    with torch.no_grad():
        delta = 1e-5
        step_size = 0.001
        for i in range(4):
            clas[i] = torch.tensor(
                clas[i] -
                step_size * np.sign(clas[i].grad) * np.random.rand(4, 4),
                requires_grad=True)

            modified = get_U(parameter[i, 0] + delta, parameter[i, 1])
            tensor_diff = (modified - quan[i]) / delta
            param_grad = torch.sum(tensor_diff * quan[i].grad,
                                   axis=[0, 1, 2, 3])
            parameter[i,
                      0] -= step_size * np.sign(param_grad) * np.random.rand()
            if parameter[i, 0] > +1:
                parameter[i, 0] = +1
            if parameter[i, 0] < -1:
                parameter[i, 0] = -1

            modified = get_U(parameter[i, 0], parameter[i, 1] + delta)
            tensor_diff = (modified - quan[i]) / delta
            param_grad = torch.sum(tensor_diff * quan[i].grad,
                                   axis=[0, 1, 2, 3])
            parameter[i,
                      1] -= step_size * np.sign(param_grad) * np.random.rand()
