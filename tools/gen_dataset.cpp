//
// Created by valma on 7/7/2025.
//

#include <iostream>
#include <fstream>
#include <random>
#include <cmath>

int main() {
    const int num_samples = 10000;
    const double r_min = 0.5;
    const double r_max = 1.0;

    std::ofstream file("C:/Users/valma/Downloads/projecto-final-pepesech/data/dataset.csv");
    if (!file.is_open()) {
        std::cerr << "No se pudo abrir el archivo para escribir.\n";
        return 1;
    }

    file << "x,y,label\n";

    std::mt19937 gen(42);
    std::uniform_real_distribution<> dist(-1.5, 1.5);

    for (int i = 0; i < num_samples; ++i) {
        double x = dist(gen);
        double y = dist(gen);
        double r2 = x * x + y * y;
        int label = (r2 >= r_min * r_min && r2 <= r_max * r_max) ? 1 : 0;
        file << x << "," << y << "," << label << "\n";
    }
    file.close();
    return 0;
}