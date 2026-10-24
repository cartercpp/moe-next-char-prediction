//
// Created by cartercpp on 10/24/26.
//

#include "neural_network.hpp"
#include <initializer_list>
#include <vector>
#include <string>
#include <fstream>
#include <memory_resource>
#include <random>
#include <cstddef>
#include <cmath>
#include "math_vector.hpp"
#include "matrix.hpp"

neural_network::neural_network(
    std::initializer_list<std::size_t> neuronsPerLayer,
    std::initializer_list<ACTIVATION_FUNCTION> activationFunctions,
    double learningRate,
    bool randomlyInitWeights,
    double gradientDecay,
    double squaredGradientDecay,
    double epsilon
)
    : m_neuronsPerLayer{neuronsPerLayer},
    m_activationFunctions{activationFunctions},
    m_learningRate{learningRate},
    m_gradientDecay{gradientDecay},
    m_squaredGradientDecay{squaredGradientDecay},
    m_epsilon{epsilon},
    m_timestep{1}
{
    m_weightMatrices.reserve(m_neuronsPerLayer.size() - 1);
    m_weightGradientsEMA.reserve(m_neuronsPerLayer.size() - 1);
    m_squaredWeightGradientsEMA.reserve(m_neuronsPerLayer.size() - 1);

    m_biasVectors.reserve(m_neuronsPerLayer.size() - 1);
    m_biasGradientsEMA.reserve(m_neuronsPerLayer.size() - 1);
    m_squaredBiasGradientsEMA.reserve(m_neuronsPerLayer.size() - 1);

    std::mt19937_64 rng{std::random_device{}()};

    for (std::size_t i = 1; i < m_neuronsPerLayer.size(); ++i)
    {
        const std::size_t layerSize = m_neuronsPerLayer[i],
                          prevLayerSize = m_neuronsPerLayer[i - 1];

        matrix<double> weights(layerSize, prevLayerSize, 0);
        if (randomlyInitWeights)
            switch (m_activationFunctions[i - 1])
            {
                case ACTIVATION_FUNCTION::RELU:
                {
                    std::normal_distribution<double> dist(0, std::sqrt(2.0 / prevLayerSize));

                    for (std::size_t row = 0; row < layerSize; ++row)
                        for (std::size_t column = 0; column < prevLayerSize; ++column)
                            weights[row, column] = dist(rng);

                    break;
                }
                default:
                {
                    std::uniform_real_distribution<double> dist(-0.5, 0.5);

                    for (std::size_t row = 0; row < layerSize; ++row)
                        for (std::size_t column = 0; column < prevLayerSize; ++column)
                            weights[row, column] = dist(rng);

                    break;
                }
            }

        m_weightMatrices.emplace_back(std::move(weights));
        m_weightGradientsEMA.emplace_back(layerSize, prevLayerSize, 0);
        m_squaredWeightGradientsEMA.emplace_back(layerSize, prevLayerSize, 0);
        m_biasVectors.emplace_back(layerSize, 0);
        m_biasGradientsEMA.emplace_back(layerSize, 0);
        m_squaredBiasGradientsEMA.emplace_back(layerSize, 0);
    }
}

std::pmr::vector<math_vector<double>> neural_network::forward(
    const math_vector<double>& input,
    std::pmr::memory_resource* resource
) const
{
    std::pmr::vector<math_vector<double>> activations{resource};
    activations.reserve(m_neuronsPerLayer.size());
    activations.push_back(input);

    for (std::size_t i = 0; i < m_weightMatrices.size(); ++i)
    {
        math_vector<double> layerActivations{m_weightMatrices[i] * activations[i] + m_biasVectors[i], resource};

        switch (m_activationFunctions[i])
        {
            case ACTIVATION_FUNCTION::RELU:
                layerActivations = Relu(std::move(layerActivations));
                break;
            case ACTIVATION_FUNCTION::SIGMOID:
                layerActivations = Sigmoid(std::move(layerActivations));
                break;
        }

        activations.emplace_back(std::move(layerActivations));
    }

    return activations;
}

math_vector<double> neural_network::backward(
    const std::pmr::vector<math_vector<double>>& activations,
    math_vector<double> gradient,
    bool gradientIsWrtPreactivation
)
{
    for (std::size_t i = m_weightMatrices.size(); i-- > 0;)
    {
        if ((i < m_weightMatrices.size() - 1) || ((i == m_weightMatrices.size() - 1) && !gradientIsWrtPreactivation))
            switch (m_activationFunctions[i])
            {
                case ACTIVATION_FUNCTION::RELU:
                    gradient = gradient.multiply(ReluDerivative(activations[i + 1]));
                    break;
                case ACTIVATION_FUNCTION::SIGMOID:
                    gradient = gradient.multiply(SigmoidDerivative(activations[i + 1]));
                    break;
            }

        const matrix<double> weightGradient{outer_product(gradient, activations[i])};

        m_weightGradientsEMA[i]
            = m_gradientDecay * m_weightGradientsEMA[i] + (1 - m_gradientDecay) * weightGradient;

        m_squaredWeightGradientsEMA[i]
            = m_squaredGradientDecay * m_squaredWeightGradientsEMA[i]
            + (1 - m_squaredGradientDecay) * weightGradient.squared();

        m_biasGradientsEMA[i]
            = m_gradientDecay * m_biasGradientsEMA[i] + (1 - m_gradientDecay) * gradient;

        m_squaredBiasGradientsEMA[i]
            = m_squaredGradientDecay * m_squaredBiasGradientsEMA[i]
            + (1 - m_squaredGradientDecay) * gradient.multiply(gradient);

        gradient = gradient * m_weightMatrices[i];
    }

    return gradient;
}

void neural_network::update_weights_and_biases()
{
    for (std::size_t i = 0; i < m_weightGradientsEMA.size(); ++i)
    {
        const matrix<double>& weightGradientEMA{m_weightGradientsEMA[i]};
        const matrix<double>& squaredWeightGradientEMA{m_squaredWeightGradientsEMA[i]};

        const matrix<double> correctedWeightGradientEMA
            = weightGradientEMA / (1.0 - std::pow(m_gradientDecay, m_timestep));

        const matrix<double> correctedSquaredWeightGradientEMA
            = squaredWeightGradientEMA / (1.0 - std::pow(m_squaredGradientDecay, m_timestep));

        matrix<double>& weightMatrixRef{m_weightMatrices[i]};
        for (std::size_t row = 0; row < weightMatrixRef.rows(); ++row)
            for (std::size_t column = 0; column < weightMatrixRef.columns(); ++column)
                weightMatrixRef[row, column]
                    -= m_learningRate * correctedWeightGradientEMA[row, column]
                        / (std::sqrt(correctedSquaredWeightGradientEMA[row, column]) + m_epsilon);
    }

    for (std::size_t i = 0; i < m_biasGradientsEMA.size(); ++i)
    {
        const math_vector<double>& biasGradientEMA{m_biasGradientsEMA[i]};
        const math_vector<double>& squaredBiasGradientEMA{m_squaredBiasGradientsEMA[i]};

        const math_vector<double> correctedBiasGradientEMA
            = biasGradientEMA / (1.0 - std::pow(m_gradientDecay, m_timestep));

        const math_vector<double> correctedSquaredBiasGradientEMA
            = squaredBiasGradientEMA / (1.0 - std::pow(m_squaredGradientDecay, m_timestep));

        math_vector<double>& biasVectorRef{m_biasVectors[i]};
        for (std::size_t i = 0; i < biasVectorRef.size(); ++i)
            biasVectorRef[i] -= m_learningRate * correctedBiasGradientEMA[i]
                                / (std::sqrt(correctedSquaredBiasGradientEMA[i]) + m_epsilon);
    }

    ++m_timestep;
}

math_vector<double> neural_network::Relu(math_vector<double> vec)
{
    for (std::size_t i = 0; i < vec.size(); ++i)
        vec[i] = (vec[i] > 0) * vec[i];

    return vec;
}

math_vector<double> neural_network::ReluDerivative(math_vector<double> vec)
{
    for (std::size_t i = 0; i < vec.size(); ++i)
        vec[i] = vec[i] > 0;

    return vec;
}

math_vector<double> neural_network::Sigmoid(math_vector<double> vec)
{
    for (std::size_t i = 0; i < vec.size(); ++i)
        vec[i] = 1.0 / (1 + std::exp(-vec[i]));

    return vec;
}

math_vector<double> neural_network::SigmoidDerivative(math_vector<double> vec)
{
    for (std::size_t i = 0; i < vec.size(); ++i)
        vec[i] = vec[i] * (1 - vec[i]);

    return vec;
}

void neural_network::load_weights(const std::string& filepath)
{
    std::ifstream file{filepath};

    for (std::size_t i = 0; i < m_weightMatrices.size(); ++i)
    {
        matrix<double>& weightMatrixRef{m_weightMatrices[i]};

        for (std::size_t row = 0; row < weightMatrixRef.rows(); ++row)
            for (std::size_t column = 0; column < weightMatrixRef.columns(); ++column)
                file >> weightMatrixRef[row, column];
    }
}

void neural_network::load_biases(const std::string& filepath)
{
    std::ifstream file{filepath};

    for (std::size_t i = 0; i < m_biasVectors.size(); ++i)
    {
        math_vector<double>& biasVectorRef{m_biasVectors[i]};

        for (std::size_t i2 = 0; i2 < biasVectorRef.size(); ++i2)
            file >> biasVectorRef[i2];
    }
}