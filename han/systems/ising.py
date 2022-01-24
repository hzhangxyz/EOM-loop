# Copyright (C) 2020-2021 Hao Zhang<zh970205@mail.ustc.edu.cn>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

from .abstract_system import AbstractSystem


class Ising(AbstractSystem):

    def __init__(self, *args, **kwargs):
        self.d = 2
        super(Ising, self).__init__(*args, **kwargs)
        self._set_hamiltonian()

    def _set_hamiltonian(self):
        sigmazsigmaz = self.Tensor(["I0", "I1", "O0", "O1"],
                                   [2, 2, 2, 2]).zero()
        block = sigmazsigmaz.blocks[sigmazsigmaz.names]
        block[0, 0, 0, 0] = 1
        block[0, 1, 0, 1] = -1
        block[1, 0, 1, 0] = -1
        block[1, 1, 1, 1] = 1
        for i in range(0, self.L2 - 2, 2):
            self.hamiltonians[(i, i + 2)] = sigmazsigmaz

        sigmax = self.Tensor(["I0", "O0"], [2, 2]).zero()
        block = sigmax.blocks[sigmax.names]
        block[1, 0] = 1
        block[0, 1] = 1
        for i in range(0, self.L2, 2):
            self.hamiltonians[(i,)] = sigmax
