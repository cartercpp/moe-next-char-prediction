//
// Created by cartercpp on 10/24/26.
//

#pragma once

#include <initializer_list>
#include <vector>
#include <string>
#include <memory_resource>
#include <cstddef>
#include "math_vector.hpp"
#include "matrix.hpp"

enum class ACTIVATION_FUNCTION{LINEAR, RELU, SIGMOID};

class neural_network
{
public:

    // CONSTRUCTORS

    explicit neural_network(
        std::initializer_list<std::size_t> neuronsPerLayer,
        std::initializer_list<ACTIVATION_FUNCTION> activationFunctions,
        double learningRate,
        bool randomlyInitWeights = true,
        double gradientDecay = 0.9,
        double squaredGradientDecay = 0.999,
        double epsilon = 1e-8
    )
        pre(neuronsPerLayer.size() >= 3)
        pre(activationFunctions.size() == neuronsPerLayer.size() - 1)
        pre((learningRate > 0) && (learningRate < 1))
        pre((gradientDecay > 0) && (gradientDecay < 1))
        pre((squaredGradientDecay > 0) && (squaredGradientDecay < 1))
        pre((epsilon > 0) && (epsilon < 0.01));

    // METHODS

    [[nodiscard]] std::pmr::vector<math_vector<double>> forward(
        const math_vector<double>& input,
        std::pmr::memory_resource* resource = std::pmr::get_default_resource()
    ) const
        pre(input.size() == m_neuronsPerLayer[0])
        post(output : output.size() == m_neuronsPerLayer.size());

    math_vector<double> backward(
        const std::pmr::vector<math_vector<double>>& activations,
        math_vector<double> gradient,
        bool gradientIsWrtPreactivation
    );

    void update_weights_and_biases();
    void load_weights(const std::string& filepath);
    void load_biases(const std::string& filepath);

private:

    static math_vector<double> Relu(math_vector<double>);
    static math_vector<double> ReluDerivative(math_vector<double>);
    static math_vector<double> Sigmoid(math_vector<double>);
    static math_vector<double> SigmoidDerivative(math_vector<double>);

    std::vector<matrix<double>> m_weightMatrices,
                                m_weightGradientsEMA,
                                m_squaredWeightGradientsEMA;
    std::vector<math_vector<double>> m_biasVectors,
                                     m_biasGradientsEMA,
                                     m_squaredBiasGradientsEMA;
    std::vector<std::size_t> m_neuronsPerLayer;
    std::vector<ACTIVATION_FUNCTION> m_activationFunctions;
    double m_learningRate,
           m_gradientDecay,
           m_squaredGradientDecay,
           m_epsilon;
    std::size_t m_timestep;
};