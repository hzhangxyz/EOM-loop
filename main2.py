import math
import TAT
from han2.tensor_U import tensor_U
from measure_device import device_tensor

Tensor = TAT.No.Z.Tensor


def get_central(u2, m):
    result = u2.edge_rename({"I1": "c1", "I2": "c2"})
    m1, m2 = m
    if m1 is not None:
        result = result.contract(m1.edge_rename({"t": "O1"}), set(), {"O1"})
    if m2 is not None:
        result = result.contract(m2.edge_rename({"t": "O2"}), set(), {"O2"})
    result = result.contract(
        u2.edge_rename({
            "I1": "c1'",
            "I2": "c2'"
        }).conjugate(),
        {("O1", "O1"), ("O2", "O2")},
    )
    return result


def main(D, r1, omega1, phi1, psi1, r2, omega2, phi2, psi2, measure):
    u1 = tensor_U(D, D, r=r1, omega=omega1, phi=phi1, psi=psi1)
    u1h = u1.shrink({"I2": 0})
    u2 = tensor_U(D, D, r=r2, omega=omega2, phi=phi2, psi=psi2)
    L = len(measure)
    central = [get_central(u2, m) for m in measure]
    result = Tensor(["n1", "n2", "n1'", "n2'"], [D, D, D, D]).zero()
    result[{"n1": 0, "n2": 0, "n1'": 0, "n2'": 0}] = 1
    for i, c in enumerate(central):
        if i != L - 1:
            result = result.contract(u1h, {("n1", "I1")}).edge_rename({"O1": "n1", "O2": "c1"})
            result = result.contract(u1h, {("n2", "I1")}).edge_rename({"O1": "n2", "O2": "c2"})
        else:
            result = result.edge_rename({"n1": "c1", "n2": "c2"})
        if "i" in c.names:
            result = result.contract(c, {("c1", "c1"), ("c2", "c2"), ("o", "i")})
        else:
            result = result.contract(c, {("c1", "c1"), ("c2", "c2")})
        if i != L - 1:
            result = result.contract(u1h.conjugate(), {("c1'", "O2"), ("n1'", "I1")}).edge_rename({"O1": "n1'"})
            result = result.contract(u1h.conjugate(), {("c2'", "O2"), ("n2'", "I1")}).edge_rename({"O1": "n2'"})
        else:
            result = result.trace({("c1'", "n1'"), ("c2'", "n2'")})
    if result.norm_num() == 1:
        return complex(result)
    else:
        return result


def unit(i, d):
    result = Tensor(["t"], [d]).zero()
    result[{"t": i}] = 1
    return result


def sum_matrix(d1, d2):
    result = Tensor(["O", "I1", "I2"], [d1 + d2 - 1, d1, d2]).zero()
    for i1 in range(d1):
        for i2 in range(d2):
            result[{"O": i1 + i2, "I1": i1, "I2": i2}] = 1
    return result


def possibility(D, r1, omega1, phi1, psi1, r2, omega2, phi2, psi2, d_out, pd, dc, ns):
    if isinstance(ns, str):
        ns = [int(n) for n in ns.split(",")]
    device = device_tensor(D, d_out, pd, dc).edge_rename({"I": "t"})
    num = main(
        D=D,
        #
        r1=r1,
        omega1=omega1,
        phi1=phi1,
        psi1=psi1,
        #
        r2=r2,
        omega2=omega2,
        phi2=phi2,
        psi2=psi2,
        #
        measure=[(None, device.shrink({"O": n})) for n in ns],
    )
    den = main(
        D=D,
        #
        r1=r1,
        omega1=omega1,
        phi1=phi1,
        psi1=psi1,
        #
        r2=r2,
        omega2=omega2,
        phi2=phi2,
        psi2=psi2,
        #
        measure=[(None, None) for n in ns],
    )
    return (num / den).real, den.real


def ideal_possibility(D, r1, omega1, phi1, psi1, r2, omega2, phi2, psi2, ns):
    if isinstance(ns, str):
        ns = [int(n) for n in ns.split(",")]
    num = main(
        D=D,
        #
        r1=r1,
        omega1=omega1,
        phi1=phi1,
        psi1=psi1,
        #
        r2=r2,
        omega2=omega2,
        phi2=phi2,
        psi2=psi2,
        #
        measure=[(None, unit(n, D)) for n in ns],
    )
    den = main(
        D=D,
        #
        r1=r1,
        omega1=omega1,
        phi1=phi1,
        psi1=psi1,
        #
        r2=r2,
        omega2=omega2,
        phi2=phi2,
        psi2=psi2,
        #
        measure=[(None, None) for n in ns],
    )
    return (num / den).real, den.real


def parity(L, D, r1, omega1, phi1, psi1, r2, omega2, phi2, psi2, d_out, pd, dc, masks):
    if isinstance(masks, str):
        if masks == "":
            masks = []
        else:
            masks = [int(n) for n in masks.split(",")]
    device = device_tensor(D, d_out, pd, dc)
    parity = Tensor(["P"], [d_out])
    for i in range(d_out):
        parity.storage[i] = +1 if i % 2 == 0 else -1
    device_parity = device.contract(parity, {("O", "P")}).edge_rename({"I": "t"})
    num = main(
        D=D,
        #
        r1=r1,
        omega1=omega1,
        phi1=phi1,
        psi1=psi1,
        #
        r2=r2,
        omega2=omega2,
        phi2=phi2,
        psi2=psi2,
        #
        measure=[(None, None if i in masks else device_parity) for i in range(L)],
    )
    den = main(
        D=D,
        #
        r1=r1,
        omega1=omega1,
        phi1=phi1,
        psi1=psi1,
        #
        r2=r2,
        omega2=omega2,
        phi2=phi2,
        psi2=psi2,
        #
        measure=[(None, None) for _ in range(L)],
    )
    return (num / den).real, den.real


def ideal_parity(L, D, r1, omega1, phi1, psi1, r2, omega2, phi2, psi2, masks):
    if isinstance(masks, str):
        if masks == "":
            masks = []
        else:
            masks = [int(n) for n in masks.split(",")]
    parity = Tensor(["t"], [D])
    for i in range(D):
        parity.storage[i] = +1 if i % 2 == 0 else -1
    num = main(
        D=D,
        #
        r1=r1,
        omega1=omega1,
        phi1=phi1,
        psi1=psi1,
        #
        r2=r2,
        omega2=omega2,
        phi2=phi2,
        psi2=psi2,
        #
        measure=[(None, None if i in masks else parity) for i in range(L)],
    )
    den = main(
        D=D,
        #
        r1=r1,
        omega1=omega1,
        phi1=phi1,
        psi1=psi1,
        #
        r2=r2,
        omega2=omega2,
        phi2=phi2,
        psi2=psi2,
        #
        measure=[(None, None) for _ in range(L)],
    )
    return (num / den).real, den.real


def count(L, D, r1, omega1, phi1, psi1, r2, omega2, phi2, psi2, d_out, pd, dc, masks):
    if isinstance(masks, str):
        if masks == "":
            masks = []
        else:
            masks = [int(n) for n in masks.split(",")]
    device = device_tensor(D, d_out, pd, dc)
    last = None
    former_d = 0
    counters = []
    for i in range(L):
        if i in masks:
            counters.append(None)
        else:
            if former_d == 0:
                counters.append(Tensor(["t", "o"], [D, D]).identity({("t", "o")}))
                former_d = D
            else:
                counters.append(sum_matrix(former_d, D).edge_rename({"I1": "i", "I2": "t", "O": "o"}))
                former_d += D - 1
            counters[-1] = counters[-1].contract(device, {("t", "O")}).edge_rename({"I": "t"})
            last = i
    if last is not None:
        number = Tensor(["i"], [former_d]).zero()
        for i in range(former_d):
            number[{"i": i}] = i
        counters[last] = counters[last].contract(number, {("o", "i")})
    num = main(
        D=D,
        #
        r1=r1,
        omega1=omega1,
        phi1=phi1,
        psi1=psi1,
        #
        r2=r2,
        omega2=omega2,
        phi2=phi2,
        psi2=psi2,
        #
        measure=[(None, counters[i]) for i in range(L)],
    )
    den = main(
        D=D,
        #
        r1=r1,
        omega1=omega1,
        phi1=phi1,
        psi1=psi1,
        #
        r2=r2,
        omega2=omega2,
        phi2=phi2,
        psi2=psi2,
        #
        measure=[(None, None) for _ in range(L)],
    )
    return (num / den).real, den.real


def ideal_count(L, D, r1, omega1, phi1, psi1, r2, omega2, phi2, psi2, masks):
    if isinstance(masks, str):
        if masks == "":
            masks = []
        else:
            masks = [int(n) for n in masks.split(",")]
    last = None
    former_d = 0
    counters = []
    for i in range(L):
        if i in masks:
            counters.append(None)
        else:
            if former_d == 0:
                counters.append(Tensor(["t", "o"], [D, D]).identity({("t", "o")}))
                former_d = D
            else:
                counters.append(sum_matrix(former_d, D).edge_rename({"I1": "i", "I2": "t", "O": "o"}))
                former_d += D - 1
            last = i
    if last is not None:
        number = Tensor(["i"], [former_d]).zero()
        for i in range(former_d):
            number[{"i": i}] = i
        counters[last] = counters[last].contract(number, {("o", "i")})
    num = main(
        D=D,
        #
        r1=r1,
        omega1=omega1,
        phi1=phi1,
        psi1=psi1,
        #
        r2=r2,
        omega2=omega2,
        phi2=phi2,
        psi2=psi2,
        #
        measure=[(None, counters[i]) for i in range(L)],
    )
    den = main(
        D=D,
        #
        r1=r1,
        omega1=omega1,
        phi1=phi1,
        psi1=psi1,
        #
        r2=r2,
        omega2=omega2,
        phi2=phi2,
        psi2=psi2,
        #
        measure=[(None, None) for _ in range(L)],
    )
    return (num / den).real, den.real


import gradio as gr

io0 = gr.Interface(
    fn=possibility,
    inputs=[
        gr.Slider(1, 100, 10, step=1),
        gr.Slider(-math.pi * 2, +math.pi * 2, 2),
        gr.Slider(-math.pi * 2, +math.pi * 2, 0),
        gr.Slider(-math.pi * 2, +math.pi * 2, math.pi),
        gr.Slider(-math.pi * 2, +math.pi * 2, 0),
        gr.Slider(-math.pi * 2, +math.pi * 2, 0),
        gr.Slider(-math.pi * 2, +math.pi * 2, math.pi / 2),
        gr.Slider(-math.pi * 2, +math.pi * 2, 0),
        gr.Slider(-math.pi * 2, +math.pi * 2, math.pi / 2),
        gr.Slider(1, 100, 11, step=1),
        gr.Slider(0, 1, 0.8),
        gr.Slider(0, 1, 0.0),
        gr.Textbox("0, 0"),
    ],
    outputs=[gr.Number(label="possibility"), gr.Number(label="normalization")],
    allow_flagging="never",
    api_name="possibility",
)
io1 = gr.Interface(
    fn=ideal_possibility,
    inputs=[
        gr.Slider(1, 100, 10, step=1),
        gr.Slider(-math.pi * 2, +math.pi * 2, 2),
        gr.Slider(-math.pi * 2, +math.pi * 2, 0),
        gr.Slider(-math.pi * 2, +math.pi * 2, math.pi),
        gr.Slider(-math.pi * 2, +math.pi * 2, 0),
        gr.Slider(-math.pi * 2, +math.pi * 2, 0),
        gr.Slider(-math.pi * 2, +math.pi * 2, math.pi / 2),
        gr.Slider(-math.pi * 2, +math.pi * 2, 0),
        gr.Slider(-math.pi * 2, +math.pi * 2, math.pi / 2),
        gr.Textbox("0, 0"),
    ],
    outputs=[gr.Number(label="possibility"), gr.Number(label="normalization")],
    allow_flagging="never",
    api_name="possibility",
)
io2 = gr.Interface(
    fn=parity,
    inputs=[
        gr.Slider(2, 100, 2, step=1),
        gr.Slider(1, 100, 10, step=1),
        gr.Slider(-math.pi * 2, +math.pi * 2, 2),
        gr.Slider(-math.pi * 2, +math.pi * 2, 0),
        gr.Slider(-math.pi * 2, +math.pi * 2, math.pi),
        gr.Slider(-math.pi * 2, +math.pi * 2, 0),
        gr.Slider(-math.pi * 2, +math.pi * 2, 0),
        gr.Slider(-math.pi * 2, +math.pi * 2, math.pi / 2),
        gr.Slider(-math.pi * 2, +math.pi * 2, 0),
        gr.Slider(-math.pi * 2, +math.pi * 2, math.pi / 2),
        gr.Slider(1, 100, 11, step=1),
        gr.Slider(0, 1, 0.8),
        gr.Slider(0, 1, 0.0),
        gr.Textbox(""),
    ],
    outputs=[gr.Number(label="parity"), gr.Number(label="normalization")],
    allow_flagging="never",
    api_name="parity",
)
io3 = gr.Interface(
    fn=ideal_parity,
    inputs=[
        gr.Slider(2, 100, 2, step=1),
        gr.Slider(1, 100, 10, step=1),
        gr.Slider(-math.pi * 2, +math.pi * 2, 2),
        gr.Slider(-math.pi * 2, +math.pi * 2, 0),
        gr.Slider(-math.pi * 2, +math.pi * 2, math.pi),
        gr.Slider(-math.pi * 2, +math.pi * 2, 0),
        gr.Slider(-math.pi * 2, +math.pi * 2, 0),
        gr.Slider(-math.pi * 2, +math.pi * 2, math.pi / 2),
        gr.Slider(-math.pi * 2, +math.pi * 2, 0),
        gr.Slider(-math.pi * 2, +math.pi * 2, math.pi / 2),
        gr.Textbox(""),
    ],
    outputs=[gr.Number(label="parity"), gr.Number(label="normalization")],
    allow_flagging="never",
    api_name="ideal_parity",
)
io4 = gr.Interface(
    fn=count,
    inputs=[
        gr.Slider(2, 100, 2, step=1),
        gr.Slider(1, 100, 10, step=1),
        gr.Slider(-math.pi * 2, +math.pi * 2, 2),
        gr.Slider(-math.pi * 2, +math.pi * 2, 0),
        gr.Slider(-math.pi * 2, +math.pi * 2, math.pi),
        gr.Slider(-math.pi * 2, +math.pi * 2, 0),
        gr.Slider(-math.pi * 2, +math.pi * 2, 0),
        gr.Slider(-math.pi * 2, +math.pi * 2, math.pi / 2),
        gr.Slider(-math.pi * 2, +math.pi * 2, 0),
        gr.Slider(-math.pi * 2, +math.pi * 2, math.pi / 2),
        gr.Slider(1, 100, 11, step=1),
        gr.Slider(0, 1, 0.8),
        gr.Slider(0, 1, 0.0),
        gr.Textbox(""),
    ],
    outputs=[gr.Number(label="count"), gr.Number(label="normalization")],
    allow_flagging="never",
    api_name="count",
)
io5 = gr.Interface(
    fn=ideal_count,
    inputs=[
        gr.Slider(2, 100, 2, step=1),
        gr.Slider(1, 100, 10, step=1),
        gr.Slider(-math.pi * 2, +math.pi * 2, 2),
        gr.Slider(-math.pi * 2, +math.pi * 2, 0),
        gr.Slider(-math.pi * 2, +math.pi * 2, math.pi),
        gr.Slider(-math.pi * 2, +math.pi * 2, 0),
        gr.Slider(-math.pi * 2, +math.pi * 2, 0),
        gr.Slider(-math.pi * 2, +math.pi * 2, math.pi / 2),
        gr.Slider(-math.pi * 2, +math.pi * 2, 0),
        gr.Slider(-math.pi * 2, +math.pi * 2, math.pi / 2),
        gr.Textbox(""),
    ],
    outputs=[gr.Number(label="count"), gr.Number(label="normalization")],
    allow_flagging="never",
    api_name="ideal_count",
)
gr.TabbedInterface([io0, io1, io2, io3, io4, io5], ["Possibility", "Ideal Possibility", "Parity", "Ideal Parity", "Count", "Ideal Count"], theme=gr.themes.Monochrome()).launch(server_port=2333,)
