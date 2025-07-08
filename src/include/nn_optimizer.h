//
// Created by valma on 6/29/2025.
//


#ifndef NN_OPTIMIZER_H
#define NN_OPTIMIZER_H

#include "tensor.h"
#include "nn_interfaces.h"
#include "nn_activation.h"
#include <unordered_map>


namespace utec::neural_network {
    using namespace algebra;
    template <typename T>
    class IOptimizer {
    public:
        virtual void update(Tensor<T,2>& param, const Tensor<T,2>& grad) = 0;
        virtual void step() {}
        virtual ~IOptimizer() = default;
    };

    template<typename T>
    class SGD final : public IOptimizer<T> {
        T lr_;
    public:
        explicit SGD(T learning_rate = 0.01): lr_(learning_rate) {}
        void update(Tensor<T, 2>& param, const Tensor<T, 2>& grad) override {
            auto shape = param.shape();
            for (size_t i = 0; i < shape[0]; ++i) {
                for (size_t j = 0; j < shape[1]; ++j) {
                    param(i, j) -= lr_ * grad(i, j);
                }
            }
        }
    };

    template<typename T>
    class Adam final : public IOptimizer<T> {
        T lr_, beta1_, beta2_, epsilon_;
        size_t t_;
        std::unordered_map<Tensor<T, 2>*, Tensor<T, 2>> m_;
        std::unordered_map<Tensor<T, 2>*, Tensor<T, 2>> v_;
    public:
        explicit Adam(T learning_rate = 0.001, T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8): lr_(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), t_(0) {}

        void update(Tensor<T, 2>& param, const Tensor<T, 2>& grad) override {
            ++t_;  // ⬅️ esto debe ir al inicio

            auto& m = m_[&param];
            auto& v = v_[&param];

            const auto shape = param.shape();
            const size_t rows = shape[0], cols = shape[1];

            if (m.shape() != shape) {
                m = Tensor<T, 2>(shape);
                for (size_t i = 0; i < rows; ++i)
                    for (size_t j = 0; j < cols; ++j)
                        m(i,j) = 0.0;
            }

            if (v.shape() != shape) {
                v = Tensor<T, 2>(shape);
                for (size_t i = 0; i < rows; ++i)
                    for (size_t j = 0; j < cols; ++j)
                        v(i,j) = 0.0;
            }

            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    m(i,j) = beta1_ * m(i,j) + (1 - beta1_) * grad(i,j);
                    v(i,j) = beta2_ * v(i,j) + (1 - beta2_) * grad(i,j) * grad(i,j);

                    T m_hat = m(i,j) / (1 - std::pow(beta1_, t_));
                    T v_hat = v(i,j) / (1 - std::pow(beta2_, t_));

                    param(i,j) -= lr_ * m_hat / (std::sqrt(v_hat) + epsilon_);
                }
            }
        }

        void step() override {
            ++t_;
        }
    };
}
#endif //NN_OPTIMIZER_H
