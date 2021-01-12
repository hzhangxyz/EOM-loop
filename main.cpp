#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace std::literals::complex_literals;
using complex = std::complex<double>;
extern "C" void zgemv_(
      const char* trans,
      const int* m,
      const int* n,
      const complex* alpha,
      const complex* a,
      const int* lda,
      const complex* x,
      const int* incx,
      const complex* beta,
      complex* y,
      const int* incy);

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

#include <TAT/TAT.hpp>

auto get_U_tensor(int n, double r, double omega, double phi, double psi) {
   auto result = TAT::Tensor<complex>({"I1", "I2", "O1", "O2"}, {n, n, n, n});
   auto U = U_matrix();
   U.generate_all(n, r, omega, phi, psi);
#if 0
  std::cout << "U matrix is\n";
  U.show_matrix(&U.matrix);
  std::cout << "adding 1 matrix is\n";
  U.show_matrix(&U.adding_1);
  std::cout << "adding 2 matrix is\n";
  U.show_matrix(&U.adding_2);
#endif
   result.block() = U.matrix;
   return result;
}

auto get_site(std::vector<double> p, std::vector<TAT::Tensor<complex>> Us) {
   auto get_UUn = [&](int n) {
      auto U1 = Us[n].edge_rename({{"I1", "1.I1"}, {"I2", "1.I2"}, {"O1", "1.O1"}, {"O2", "1.O2"}}).shrink({{"1.I2", 0}});
      auto U2 = Us[n].conjugate().edge_rename({{"I1", "2.I1"}, {"I2", "2.I2"}, {"O1", "2.O1"}, {"O2", "2.O2"}}).shrink({{"2.I2", 0}});
      return p[n] * U1.contract(U2, {});
   };
   auto result = get_UUn(0);
   for (auto i = 1; i < p.size(); i++) {
      result += get_UUn(i);
   }
   return result;
}

complex trace_mps(int l, const TAT::Tensor<complex>& a) {
   auto as = a.shrink({{"1.I1", 0}, {"2.I1", 0}}).edge_rename({{"1.O2", "A.1.O2"}, {"2.O2", "A.2.O2"}});
   auto result = as.trace({{"1.O1", "2.O1"}});
   for (auto t = 0; t < l - 1; t++) {
      result = result.contract(a.edge_rename({{"1.O2", "A.1.O2"}, {"2.O2", "A.2.O2"}}), {{"A.1.O2", "1.I1"}, {"A.2.O2", "2.I1"}})
                     .trace({{"1.O1", "2.O1"}});
   }
   result = result.contract(a.edge_rename({{"1.O2", "A.1.O2"}, {"2.O2", "A.2.O2"}}), {{"A.1.O2", "1.I1"}, {"A.2.O2", "2.I1"}})
                  .trace({{"1.O1", "2.O1"}, {"A.1.O2", "A.2.O2"}});
   return result;
}
complex contract_mps(int l, const TAT::Tensor<complex>& a, const TAT::Tensor<complex>& b) {
   auto as = a.shrink({{"1.I1", 0}, {"2.I1", 0}}).edge_rename({{"1.O2", "A.1.O2"}, {"2.O2", "A.2.O2"}});
   auto bs = b.shrink({{"1.I1", 0}, {"2.I1", 0}}).edge_rename({{"1.O2", "B.1.O2"}, {"2.O2", "B.2.O2"}});
   auto result = as.contract(bs, {{"1.O1", "1.O1"}, {"2.O1", "2.O1"}});
   for (auto t = 0; t < l - 1; t++) {
      result = result.contract(a.edge_rename({{"1.O2", "A.1.O2"}, {"2.O2", "A.2.O2"}}), {{"A.1.O2", "1.I1"}, {"A.2.O2", "2.I1"}})
                     .contract(
                           b.edge_rename({{"1.O2", "B.1.O2"}, {"2.O2", "B.2.O2"}}),
                           {{"B.1.O2", "1.I1"}, {"B.2.O2", "2.I1"}, {"1.O1", "1.O1"}, {"2.O1", "2.O1"}});
   }
   result = result.contract(a.edge_rename({{"1.O2", "A.1.O2"}, {"2.O2", "A.2.O2"}}), {{"A.1.O2", "1.I1"}, {"A.2.O2", "2.I1"}})
                  .contract(
                        b.edge_rename({{"1.O2", "B.1.O2"}, {"2.O2", "B.2.O2"}}),
                        {{"B.1.O2", "1.I1"}, {"B.2.O2", "2.I1"}, {"1.O1", "1.O1"}, {"2.O1", "2.O1"}, {"A.1.O2", "B.1.O2"}, {"A.2.O2", "B.2.O2"}});
   return result;
}

extern "C" void check(int l, int n, double r, double omega, double phi, double psi, double delta, int delta_which) {
   auto U = get_U_tensor(n, r, omega, phi, psi);
#if 0
  std::cout << "U is " << U << "\n";
#endif
   auto S = get_site({1}, {U});
   std::vector<double> p;
   std::vector<TAT::Tensor<complex>> Us;
   int sample = 5;
   for (auto i = -sample; i <= +sample; i++) {
      p.push_back(1. / (sample * 2 + 1));
      switch (delta_which) {
         case 0:
            Us.push_back(get_U_tensor(n, r * (1 + i * delta / sample), omega, phi, psi));
            break;
         case 1:
            Us.push_back(get_U_tensor(n, r, omega * (1 + i * delta / sample), phi, psi));
            break;
         case 2:
            Us.push_back(get_U_tensor(n, r, omega, phi * (1 + i * delta / sample), psi));
            break;
         case 3:
            Us.push_back(get_U_tensor(n, r, omega, phi, psi * (1 + i * delta / sample)));
            break;
         default:
            std::cerr << "Wrong delta which\n";
            exit(-1);
      }
   }
   auto Ss = get_site(p, Us);
#if 0
  std::cout << "UU is " << S << "\n";
  std::cout << "UpU is " << Ss << "\n";
#endif
   S /= S.norm<-1>();
   Ss /= Ss.norm<-1>();
   auto tracerhorhos = contract_mps(l, S, Ss);
   auto tracerho = trace_mps(l, S);
   auto tracerhos = trace_mps(l, Ss);
#if 0
  std::cout << "tr(rho rho') is " << tracerhorhos << "\n";
  std::cout << "tr(rho) is " << tracerho << "\n";
  std::cout << "tr(rho') is " << tracerhos << "\n";
#endif
   auto f = tracerhorhos / (tracerho * tracerhos);
   std::cout << "Fidelity is " << std::setprecision(20) << f.real() << "\n";
}

#include <fire.hpp>

int fired_main(
      int l = fire::arg({"-L", "--loop", "system loop number"}),
      int n = fire::arg({"-N", "--n-cut", "particle number cutoff"}),
      double r = fire::arg({"-R", "--r-value"}),
      double omega = fire::arg({"-O", "--omega"}),
      double phi = fire::arg({"-H", "--phi"}),
      double psi = fire::arg({"-S", "--psi"}),
      double delta = fire::arg({"-D", "--delta", "relative error of specified parameter"}),
      int delta_which = fire::arg(
            {"-W",
             "--which",
             "check which parameter, 0 for r, 1 for "
             "omega, 2 for phi, 3 for psi"})) {
   check(l, n, r, omega, phi, psi, delta, delta_which);
   return 0;
}

FIRE(fired_main)
