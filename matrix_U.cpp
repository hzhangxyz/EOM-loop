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
#include <iomanip>
#include <iostream>
#include <vector>

using namespace std::literals::complex_literals;
using complex = std::complex<double>;

void mv(int n, const complex* A, const complex* x, complex* y) {
   for (auto i = 0; i < n; i++) {
      y[i] = 0;
      for (auto j = 0; j < n; j++) {
         y[i] += A[j * n + i] * x[j];
      }
   }
}

struct U_matrix {
   int n; // 截断
   std::vector<complex> matrix;
   std::vector<complex> adding_1;
   std::vector<complex> adding_2;

   complex r;
   complex omega;
   complex phi;
   complex psi;

   complex sigma1;
   complex tau1;
   complex mu1;
   complex nu1;
   complex sigma2;
   complex tau2;
   complex mu2;
   complex nu2;

   void generate_parameter() {
      using namespace std;
      sigma1 = cos(omega / 2.) * cosh(r / 2.);
      tau1 = exp(-1i * (phi + psi)) * sin(omega / 2.) * sinh(r / 2.);
      mu1 = exp(-1i * psi) * sin(omega / 2.) * cosh(r / 2.);
      nu1 = exp(-1i * phi) * cos(omega / 2.) * sinh(r / 2.);
      sigma2 = -exp(1i * psi) * sin(omega / 2.) * cosh(r / 2.);
      tau2 = exp(-1i * phi) * cos(omega / 2.) * sinh(r / 2.);
      mu2 = cos(omega / 2.) * cosh(r / 2.);
      nu2 = -exp(-1i * (phi - psi)) * sin(omega / 2.) * sinh(r / 2.);
   }

   void generate_adding() {
      std::vector<complex> a_1_dagger(n * n * n * n);
      std::vector<complex> a_2_dagger(n * n * n * n);
      std::vector<complex> a_1(n * n * n * n);
      std::vector<complex> a_2(n * n * n * n);
      // a_1_dagger
      for (auto p2 = 0; p2 < n; p2++) {
         for (auto i1 = 0; i1 < n - 1; i1++) {
            auto o1 = i1 + 1;
            get_element(o1, p2, i1, p2, &a_1) = get_element(i1, p2, o1, p2, &a_1_dagger) = std::sqrt(o1);
         }
      }
      // a_2_dagger
      for (auto p1 = 0; p1 < n; p1++) {
         for (auto i2 = 0; i2 < n - 1; i2++) {
            auto o2 = i2 + 1;
            get_element(p1, o2, p1, i2, &a_2) = get_element(p1, i2, p1, o2, &a_2_dagger) = std::sqrt(o2);
         }
      }
#if 0
    std::cout << "a1dagger\n";
    show_matrix(&a_1_dagger);
    std::cout << "a1\n";
    show_matrix(&a_1);
    std::cout << "a2dagger\n";
    show_matrix(&a_2_dagger);
    std::cout << "a2\n";
    show_matrix(&a_2);
#endif
      // adding 1 = tau1* a1 + sigma1* a1dagger + nu1* a2 + mu1* a2dagger
      for (auto i = 0; i < n * n * n * n; i++) {
         adding_1[i] = std::conj(tau1) * a_1[i] + std::conj(sigma1) * a_1_dagger[i] + std::conj(nu1) * a_2[i] + std::conj(mu1) * a_2_dagger[i];
      }
      // adding 2 = tau2* a1 + sigma2* a1dagger + nu2* a2 + mu2* a2dagger
      for (auto i = 0; i < n * n * n * n; i++) {
         adding_2[i] = std::conj(tau2) * a_1[i] + std::conj(sigma2) * a_1_dagger[i] + std::conj(nu2) * a_2[i] + std::conj(mu2) * a_2_dagger[i];
      }
   }

   void create_matrix() {
      matrix = decltype(matrix)(n * n * n * n);
      adding_1 = decltype(adding_1)(n * n * n * n);
      adding_2 = decltype(adding_2)(n * n * n * n);
   }
   void show_matrix(std::vector<complex>* m = nullptr) {
      if (m == nullptr) {
         m = &matrix;
      }
      for (auto i = 0; i < n * n; i++) {
         for (auto j = 0; j < n * n; j++) {
            if (j != 0) {
               std::cout << ", ";
            }
            std::cout << (*m)[j * n * n + i];
         }
         std::cout << "\n";
      }
   }
   std::complex<double>& get_element(long in1, long in2, long out1, long out2, std::vector<complex>* m = nullptr) {
      if (m == nullptr) {
         m = &matrix;
      }
      long offset = 0;
      offset = offset * n + in1;
      offset = offset * n + in2;
      offset = offset * n + out1;
      offset = offset * n + out2;
      return (*m)[offset];
   }

   void norm_column(int a, int b) {
      double square_sum = 0;
      for (auto i = 0; i < n; i++) {
         for (auto j = 0; j < n; j++) {
            auto this_element = get_element(a, b, i, j);
            square_sum += std::norm(this_element);
         }
      }
      auto parameter = std::sqrt(square_sum);
      for (auto i = 0; i < n; i++) {
         for (auto j = 0; j < n; j++) {
            get_element(a, b, i, j) /= parameter;
         }
      }
   }

   void generate_00() {
      for (auto i = 0; i < n; i++) {
         get_element(0, 0, i, i) = std::pow(-std::exp(-1i * phi) * std::tanh(r / 2.), i) / std::cosh(r / 2.);
      }
      norm_column(0, 0);
   }

   void generate_other() {
      // get_element(0, 0, ..., ...) --adding 2-->  get_element(0, 1, ..., ...);
      for (auto i = 1; i < n; i++) {
         mv(n * n, adding_2.data(), &get_element(0, i - 1, 0, 0), &get_element(0, i, 0, 0));
         norm_column(0, i);
      }
      for (auto j = 1; j < n; j++) {
         for (auto i = 0; i < n; i++) {
            mv(n * n, adding_1.data(), &get_element(j - 1, i, 0, 0), &get_element(j, i, 0, 0));
            norm_column(j, i);
         }
      }
   }

   void generate_all(int _n, double _r, double _omega, double _phi, double _psi) {
      n = _n;
      create_matrix();
      r = _r;
      omega = _omega;
      phi = _phi;
      psi = _psi;
      generate_00();
      generate_parameter();
      generate_adding();
      generate_other();
   }
};

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(matrix_U, matrix_U_m) {
   matrix_U_m.doc() = "generate matrix U for EOM loop";
   py::class_<U_matrix>(matrix_U_m, "matrix_U", "U matrix for EOM loop, dimension is [I1, I2, O1, O2]", py::buffer_protocol())
         .def(py::init<>([](int n, double r, double omega, double phi, double psi) {
                 auto U = U_matrix();
                 U.generate_all(n, r, omega, phi, psi);
                 return U;
              }),
              py::arg("n"),
              py::arg("r"),
              py::arg("omega"),
              py::arg("phi"),
              py::arg("psi"))
         .def_buffer([](U_matrix& U) {
            auto n = U.n;
            return py::buffer_info{
                  U.matrix.data(),
                  sizeof(complex),
                  py::format_descriptor<complex>::format(),
                  4,
                  std::vector<int>{n, n, n, n},
                  std::vector<int>{sizeof(complex) * n * n * n, sizeof(complex) * n * n, sizeof(complex) * n, sizeof(complex)}};
         });
}