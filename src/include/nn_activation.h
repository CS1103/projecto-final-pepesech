//
// Created by valma on 6/17/2025.
//


#ifndef ACTIVATION_H
#define ACTIVATION_H

#include"nn_interfaces.h"
namespace utec::neural_network {
    using namespace algebra;
    template<typename T>
    class ReLU : public ILayer<T> {
        Tensor<T,2> mask; //para almacenar donde x>0
    public:
        Tensor<T,2> forward(const Tensor<T,2>& x) override {
            mask = Tensor<T,2>(x.shape());
            Tensor<T,2> out (x.shape());
            for( size_t i = 0; i < x.shape()[0]; i++) {
                for(size_t j = 0; j< x.shape()[1]; j++) {
                    if(x(i,j) > 0) {
                        out(i, j) = x(i, j);
                        mask(i, j) = 1;
                    } else {
                        out(i, j) = 0;
                        mask(i, j) = 0;
                    }
                }
            }
            return out;
        }
        Tensor<T,2> backward(const Tensor<T,2>& grad) override {
            Tensor<T,2> out(grad.shape());
            for (size_t i = 0; i < grad.shape()[0]; ++i) {
                for (size_t j = 0; j < grad.shape()[1]; ++j) {
                    out(i,j) = grad(i,j) * mask(i,j);
                }
            }
            return out;
        }

        void update_params(T l) override {
        }
    };

    template<typename T>
class Sigmoid : public ILayer<T> {
        Tensor<T,2> last_output;

    public:
        Tensor<T,2> forward(const Tensor<T,2>& x) override {
            last_output = Tensor<T,2>(x.shape());
            Tensor<T,2> out(x.shape());

            for (size_t i = 0; i < x.shape()[0]; ++i) {
                for (size_t j = 0; j < x.shape()[1]; ++j) {
                    T z = x(i, j);
                    T s = 1 / (1 + std::exp(-z));
                    out(i, j) = s;
                    last_output(i, j) = s;
                }
            }

            return out;
        }

        Tensor<T,2> backward(const Tensor<T,2>& grad) override {

            Tensor<T,2> dx(grad.shape());
            for (size_t i = 0; i < grad.shape()[0]; ++i) {
                for (size_t j = 0; j < grad.shape()[1]; ++j) {
                    T s = last_output(i, j);
                    dx(i, j) = grad(i, j) * s * (T(1) - s);
                }
            }
            return dx;
        }

        void update_params(T l) override {

        }
    };
}
#endif //ACTIVATION_H
