//
// Created by valma on 6/17/2025.
//

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H
#include <memory>
#include "nn_interfaces.h"
#include "nn_activation.h"
#include <vector>
#include "nn_loss.h"
#include "nn_optimizer.h"

namespace utec::neural_network {

    template<typename T>
    class NeuralNetwork {
        std::vector<std::unique_ptr<ILayer<T>>> layers_;
    public:
        void add_layer(std::unique_ptr<ILayer<T>> layer) { layers_.emplace_back(std::move(layer));}


        template <template <typename ...> class LossType>
        void train(const Tensor<T,2>& X, const Tensor<T,2>& Y, const size_t epochs, const size_t batch_size, T learning_rate) {
            const size_t samples = X.shape()[0];

            for (size_t epoch = 0; epoch < epochs; ++epoch) {
                for (size_t i = 0; i < samples; i += batch_size) {
                    const size_t row_start = i;
                    const size_t row_end = i + std::min(batch_size, samples - i);

                    Tensor<T,2> x_batch = X.slice(row_start, row_end);
                    Tensor<T,2> y_batch = Y.slice(row_start, row_end);

                    // Forward
                    Tensor<T,2> prediction = x_batch;
                    for (auto& layer : layers_)
                        prediction = layer->forward(prediction);

                    // Create loss with prediction and ground truth
                    LossType<T> loss(prediction, y_batch);
                    T current_loss = loss.loss();

                    // Backward
                    Tensor<T,2> grad = loss.loss_gradient();
                    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it)
                        grad = (*it)->backward(grad);

                    // Update parameters using learning rate directly
                    for (auto& layer : layers_)
                        layer->update_params(learning_rate);
                }
            }
        }
        Tensor<T,2> predict(const Tensor<T,2>& X) {
            Tensor<T,2> result = X;
            for (auto& layer : layers_)
                result = layer->forward(result);
            return result;
        }
    };
}

#endif //NEURAL_NETWORK_H
