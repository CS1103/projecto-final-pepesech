//
// Created by valma on 7/7/2025.
//
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <tuple>
#include <random>
#include <chrono>
#include <algorithm>
#include <memory>

#include "include/tensor.h"
#include "include/neural_network.h"
#include "include/nn_loss.h"
#include "include/nn_dense.h"
#include "include/nn_activation.h"
#include "include/nn_optimizer.h"
#include "include/nn_interfaces.h"

using T = double;

std::vector<std::tuple<T,T,T>> load_dataset(const std::string& path) {
    std::ifstream file(path);
    std::string line;
    std::getline(file, line);

    std::vector<std::tuple<T,T,T>> data;
    while(std::getline(file, line)) {
        std::stringstream ss(line);
        std::string val;
        std::vector<T> fila;
        while (std::getline(ss, val, ',')) {
            fila.push_back(std::stod(val));
        }
        data.emplace_back(fila[0], fila[1], fila[2]);
    }

    return data;
}

void to_tensor(const std::vector<std::tuple<T,T,T>>& data, utec::algebra::Tensor<T,2>& X, utec::algebra::Tensor<T,2>& Y) {
    size_t n = data.size();
    X = utec::algebra::Tensor<T,2>({n,2});
    Y = utec::algebra::Tensor<T,2>({n,1});
    for(size_t i = 0; i < n; i++) {
        X(i,0) = std::get<0>(data[i]);
        X(i,1) = std::get<1>(data[i]);
        Y(i,0) = std::get<2>(data[i]);
    }
}

double accuracy(const utec::algebra::Tensor<T,2>& y_pred, const utec::algebra::Tensor<T,2>& y_true) {
    size_t aciertos = 0;
    for(size_t i =0; i< y_true.shape(0); i++) {
        int pred = y_pred(i,0) > 0.5 ? 1 : 0;
        int real = y_true(i,0) > 0.5 ? 1 : 0;
        if (pred == real){ aciertos++;}
    }

    return static_cast<T>(aciertos)/y_true.shape(0);
}

std::mt19937 gen(42);
auto xavier_init = [](auto& M) {
    const auto shape = M.shape();
    double limit = std::sqrt(6.0/(shape[0]+shape[1]));
    std::uniform_real_distribution<> dist(-limit, limit);
    for(auto& v : M) v = dist(gen);
};

auto init_zero = [](auto& M) {
    for (auto& v : M) v = 0.0;
};

int main(){
    auto data = load_dataset("C:/Users/valma/Downloads/projecto-final-pepesech/data/dataset.csv");
    std::cout << "Datos cargados: " << data.size() << std::endl;

    std::shuffle(data.begin(), data.end(), std::mt19937{42});

    size_t total = data.size();
    auto train_size = static_cast<size_t>(total*0.8);

    std::vector<std::tuple<T, T, T>> train_data(data.begin(), data.begin() + train_size);
    std::vector<std::tuple<T, T, T>> test_data(data.begin() + train_size, data.end());

    utec::algebra::Tensor<T,2> X_train, Y_train, X_test, Y_test;
    to_tensor(train_data, X_train, Y_train);
    to_tensor(test_data, X_test, Y_test);

    utec::neural_network::NeuralNetwork<T> net;
    net.add_layer(std::make_unique<utec::neural_network::Dense<T>>(2, 8, xavier_init, init_zero));
    net.add_layer(std::make_unique<utec::neural_network::ReLU<T>>());
    net.add_layer(std::make_unique<utec::neural_network::Dense<T>>(8, 1, xavier_init, init_zero));
    net.add_layer(std::make_unique<utec::neural_network::Sigmoid<T>>());

    size_t epochs = 100;
    size_t batch_size = 32;
    T learning_rate = 0.001;

    std::cout << "X_train shape: " << X_train.shape(0) << " x " << X_train.shape(1) << "\n";
    std::cout << "Y_train shape: " << Y_train.shape(0) << " x " << Y_train.shape(1) << "\n";

    auto start = std::chrono::high_resolution_clock::now();

    net.train<utec::neural_network::BCELoss>(X_train, Y_train, epochs, batch_size, learning_rate);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> tiempo = end - start;

    std::cout << "Tiempo total de entrenamiento: " << tiempo.count() << " segundos\n";

    utec::neural_network::Tensor<T,2> y_pred = net.predict(X_test);
    double acc = accuracy(y_pred, Y_test);
    std::cout << "Precisión en conjunto de prueba: " << acc * 100.0 << "%\n";
    return 0;
}