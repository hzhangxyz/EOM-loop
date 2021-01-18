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

/**
 * 用于描述粒子透过率的矩阵
 *
 * 参见PHYSICAL REVIEW A70, 042308(2004) 公式13
 */
struct matrix_A {
   // 为了后续操作简单, 这里有两个截断, 原则上系统是无穷维的
   // 但是实际上我们只保留到第c位, 描述他的矩阵可以是c维的
   // 也可以是大于c维的矩阵, 即n维, 多余的维度填0
   int n; // 矩阵截断
   int c; // 物理截断
   int k; // 丢失粒子数
   double eta;

   std::vector<complex> matrix;
   matrix_A(int _n, int _c, int _k, double _eta) : n(_n), c(_c), k(_k), eta(_eta) {
      // 截断为n, 矩阵大小为n*n
      matrix.resize(n * n);
      for (auto i = k; i < c; i++) {
         // Ak矩阵的|i-k><i|项, i应取k到无穷大, 但是这里只取到物理截断
         get_element(i - k, i) = create_element(i, k, eta);
      }
   }

   /**
    * 获取n*n矩阵的某个元素
    */
   complex& get_element(int i, int j, std::vector<complex>* p = nullptr) {
      if (!p) {
         p = &matrix;
      }
      return matrix[i * n + j];
   }

   /**
    * 获得组合数
    */
   static int binomial(int n, int k) {
      if (k == 0 || k == n) {
         return 1;
      } else {
         // 递归求值
         return binomial(n - 1, k - 1) + binomial(n - 1, k);
      }
   }

   static complex create_element(int n, int k, double eta) {
      // 参见公式13
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
            std::size_t n = A.n;
            return py::buffer_info{
                  A.matrix.data(),
                  sizeof(complex),
                  py::format_descriptor<complex>::format(),
                  2,
                  std::vector<std::size_t>{n, n},
                  std::vector<std::size_t>{sizeof(complex) * n, sizeof(complex)}};
         });
}
