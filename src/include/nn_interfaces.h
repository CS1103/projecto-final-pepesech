//
// Created by valma on 6/17/2025.
//

#ifndef LAYER_H
#define LAYER_H
#include "tensor.h"
#include <cmath>

namespace utec::neural_network {
    template<typename T>
    class ILayer {
        public :
        virtual ~ ILayer () = default ;
        // Forward : recibe batch x features , devuelve batch x units
        virtual algebra::Tensor <T ,2 > forward ( const algebra::Tensor <T ,2 >& x) = 0;
        // Backward : recibe gradiente de salida , devuelve gradiente de entrada
        virtual algebra::Tensor <T ,2 > backward ( const algebra::Tensor <T ,2 >& grad ) = 0;
        virtual void update_params(T learning_rate) = 0;
    };

    template<typename T, size_t Rank>
    class ILoss {
        public:
            virtual ~ILoss() = default;
            virtual T loss() const = 0;
            virtual algebra::Tensor<T, Rank> loss_gradient() const = 0;
    };
}

#endif //LAYER_H
