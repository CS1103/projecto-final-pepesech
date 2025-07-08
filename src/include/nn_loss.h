//
// Created by valma on 6/17/2025.
//

#ifndef LOSS_H
#define LOSS_H
#include "nn_interfaces.h"
#include <cmath>
namespace utec::neural_network {
    using namespace algebra;
    template<typename T>
    class MSELoss final : public ILoss<T, 2> {
            Tensor<T, 2> y_pred, y_true;

        public:
            MSELoss(const Tensor<T,2>& y_prediction, const Tensor<T,2>& y_target)
                : y_pred(y_prediction), y_true(y_target) {}

            T loss() const override {
                T sum = T{};
                for (size_t i = 0; i < y_pred.shape()[0]; ++i) {
                    for (size_t j = 0; j < y_pred.shape()[1]; ++j) {
                        T diff = y_pred(i,j) - y_true(i,j);
                        sum += diff * diff;
                    }
                }
                return sum / static_cast<T>(Tensor<T,2>::total_size(y_pred.shape()));
            }

            Tensor<T,2> loss_gradient() const override {
                Tensor<T,2> grad(y_pred.shape());
                T scale = 2 / static_cast<T>(Tensor<T,2>::total_size(y_pred.shape()));
                for (size_t i = 0; i < grad.shape()[0]; ++i) {
                    for (size_t j = 0; j < grad.shape()[1]; ++j) {
                        grad(i,j) = scale * (y_pred(i,j) - y_true(i,j));
                    }
                }
                return grad;
            }
    };

        template<typename T>
        class BCELoss final : public ILoss<T, 2> {
            Tensor<T, 2> y_pred, y_true;

        public:
            BCELoss(const Tensor<T,2>& y_prediction, const Tensor<T,2>& y_target)
            : y_pred(y_prediction), y_true(y_target) {}

            T loss() const override {
                T total = T{};
                const T epsilon = 1e-10;
                for (size_t i = 0; i < y_pred.shape()[0]; ++i) {
                    for (size_t j = 0; j < y_pred.shape()[1]; ++j) {
                        T y = y_true(i,j);
                        T p = std::clamp(y_pred(i,j), epsilon, T(1) - epsilon);
                        total += -y * std::log(p) - (1 - y) * std::log(1 - p);
                    }
                }
                return total / static_cast<T>(Tensor<T,2>::total_size(y_pred.shape()));
            }

            Tensor<T,2> loss_gradient() const override {
                Tensor<T,2> grad(y_pred.shape());
                const T epsilon = 1e-10;
                for (size_t i = 0; i < grad.shape()[0]; ++i) {
                    for (size_t j = 0; j < grad.shape()[1]; ++j) {
                        T y = y_true(i,j);
                        T p = std::clamp(y_pred(i,j), epsilon, T(1) - epsilon);
                        grad(i,j) = (p - y) / (p * (1 - p) * static_cast<T>(Tensor<T,2>::total_size(y_pred.shape())));
                    }
                }
                return grad;
            }
    };

}
#endif //LOSS_H
