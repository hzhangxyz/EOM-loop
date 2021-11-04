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


class Heisenberg(AbstractSystem):

    def __init__(self, *args, **kwargs):
        super(Heisenberg, self).__init__(*args, **kwargs)
        self.d = 2
        self._set_hamiltonian()

    def _set_hamiltonian(self):
        H = self.Tensor(["I0", "I1", "O0", "O1"], [2, 2, 2, 2]).zero()
        block = H.blocks[H.names]
        block[0, 0, 0, 0] = 1 / 4.
        block[0, 1, 0, 1] = -1 / 4.
        block[1, 0, 1, 0] = -1 / 4.
        block[1, 1, 1, 1] = 1 / 4.
        block[1, 0, 0, 1] = 2 / 4.
        block[0, 1, 1, 0] = 2 / 4.
        for i in range(self.L2 - 1):
            self.hamiltonians[(i, i + 1)] = H
