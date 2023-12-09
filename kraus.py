import TAT

Tensor = TAT.No.Z.Tensor


def factor(k):
    if k == 0:
        return 1
    else:
        return factor(k - 1) * k


def kraus(D, k, eta):
    result = Tensor(["k", "O", "I"], [k, D, D]).zero()
    for i in range(k):
        for a in range(D):
            b = a - i
            if b >= 0:
                param = ((1 - eta)**(i / 2)) * (factor(a) / factor(i) / factor(a - i))**(1 / 2)
                result[{"k": i, "I": a, "O": b}] = eta**(b / 2) * param
    return result


# print(kraus(2,2,0.5))
