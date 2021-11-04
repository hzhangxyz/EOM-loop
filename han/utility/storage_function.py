#!/usr/bin/env python
# -*- coding: utf-8 -*-
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


class StorageFunction:

    def __init__(self, func):
        self.func = func
        self.storage = {}

    def __call__(self, *args, **kwargs):
        kwtuple = tuple(kwargs.items())
        if (args, kwtuple) not in self.storage:
            self.storage[(args, kwtuple)] = self.func(*args, **kwargs)
        return self.storage[args, kwtuple]
