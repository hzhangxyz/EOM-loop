/**
 * Copyright (C) 2021 Hao Zhang<zh970205@mail.ustc.edu.cn>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <cmath>
#include <complex>
#include <vector>

using complex = std::complex<double>;

struct matrix_A {
   int n; // 矩阵截断
   int c; // 物理截断
   int k; // 丢失粒子数
   double eta;

   std::vector<complex> matrix;
   matrix_A(int _n, int _c, int _k, double _eta) : n(_n), c(_c), k(_k), eta(_eta) {
      matrix = std::vector<complex>(n * n);
      for (auto i = k; i < c; i++) {
         get_element(i - k, i) = create_element(i, k, eta);
      }
   }

   complex& get_element(int i, int j, std::vector<complex>* p = nullptr) {
      if (!p) {
         p = &matrix;
      }
      return matrix[i * n + j];
   }

   static int binomial(int n, int k) {
      if (k == 0 || k == n) {
         return 1;
      } else {
         return binomial(n - 1, k - 1) + binomial(n - 1, k);
      }
   }

   static complex create_element(int n, int k, double eta) {
      return std::sqrt(binomial(n, k)) * std::pow(eta, (n - k) / 2.) * std::pow(1 - eta, k / 2.);
   }
};

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(matrix_A, matrix_A_m) {
   matrix_A_m.doc() = "generate matrix A for EOM loop";
   py::class_<matrix_A>(matrix_A_m, "matrix_A", "A matrix for EOM loop, dimension is [O, I]", py::buffer_protocol())
         .def(py::init<>([](int n, int c, int k, double eta) { return matrix_A(n, c, k, eta); }),
              py::arg("cutoff_matrix"),
              py::arg("cutoff_physics"),
              py::arg("particle_number_lost"),
              py::arg("eta"))
         .def_buffer([](matrix_A& A) {
            auto n = A.n;
            return py::buffer_info{
                  A.matrix.data(),
                  sizeof(complex),
                  py::format_descriptor<complex>::format(),
                  2,
                  std::vector<int>{n, n},
                  std::vector<int>{sizeof(complex) * n, sizeof(complex)}};
         });
}
