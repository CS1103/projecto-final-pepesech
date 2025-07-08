//
// Created by valma on 6/17/2025.
//

#ifndef DENSE_H
#define DENSE_H

#include"nn_interfaces.h"

namespace utec::neural_network{
    using namespace algebra;

    template<typename T>
    class Dense final : public ILayer<T> {
        Tensor<T, 2> W, dW;
        Tensor<T, 1> b, db;
        Tensor<T, 2> last_x;

    public:

        template<typename InitWFun, typename InitBFun>
        Dense(size_t in_features, size_t out_features, InitWFun init_w_fun, InitBFun init_b_fun)
            : W({in_features, out_features}),
              dW({in_features, out_features}),
              b({out_features}),
              db({out_features}) {


            init_w_fun(W);


            Tensor<T, 2> b_temp({1, out_features});
            init_b_fun(b_temp);


            for (size_t i = 0; i < out_features; ++i) {
                b(i) = b_temp(0, i);
            }
        }


        Tensor<T, 2> forward(const Tensor<T, 2>& x) override {

            last_x = x;

            std::cout << "[Dense] X shape: " << x.shape()[0] << "x" << x.shape()[1] << "\n";
            std::cout << "[Dense] W shape: " << W.shape()[0] << "x" << W.shape()[1] << "\n";
            Tensor<T, 2> Y = matrix_product(x, W);


            const auto Y_shape = Y.shape();
            for (size_t i = 0; i < Y_shape[0]; ++i) {
                for (size_t j = 0; j < Y_shape[1]; ++j) {
                    Y(i, j) += b(j);
                }
            }

            return Y;
        }


        Tensor<T, 2> backward(const Tensor<T, 2>& dZ) override {


            const auto batch_size = dZ.shape()[0];
            const auto out_features = dZ.shape()[1];
            const auto in_features = last_x.shape()[1];


            Tensor<T, 2> X_T = transpose_2d(last_x);
            dW = matrix_product(X_T, dZ);


            for (size_t j = 0; j < out_features; ++j) {
                T sum = T{0};
                for (size_t i = 0; i < batch_size; ++i) {
                    sum += dZ(i, j);
                }
                db(j) = sum;
            }


            Tensor<T, 2> W_T = transpose_2d(W);
            Tensor<T, 2> dX = matrix_product(dZ, W_T);

            return dX;
        }


        void update_params(T learning_rate) override {

            for (size_t i = 0; i < W.shape()[0]; ++i) {
                for (size_t j = 0; j < W.shape()[1]; ++j) {
                    W(i, j) -= learning_rate * dW(i, j);
                }
            }


            for (size_t i = 0; i < b.shape()[0]; ++i) {
                b(i) -= learning_rate * db(i);
            }
        }


        const Tensor<T, 2>& get_weights() const { return W; }
        const Tensor<T, 1>& get_bias() const { return b; }
        const Tensor<T, 2>& get_dW() const { return dW; }
        const Tensor<T, 1>& get_db() const { return db; }
    };
}

#endif //DENSE_H
