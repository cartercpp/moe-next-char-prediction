//
// Created by cartercpp on 10/24/26.
//

#include "mixture_of_experts.hpp"
#include <string>
#include <initializer_list>
#include <vector>
#include <memory_resource>
#include <ranges>
#include <algorithm>
#include <utility>
#include <limits>
#include <cstddef>
#include <cmath>
#include "neural_network.hpp"

mixture_of_experts::mixture_of_experts(
    std::size_t experts,
    std::size_t topK,
    std::initializer_list<std::size_t> expertNeuronsPerLayer,
    std::initializer_list<std::size_t> routerNeuronsPerLayer,
    std::initializer_list<ACTIVATION_FUNCTION> expertActivationFunctions,
    std::initializer_list<ACTIVATION_FUNCTION> routerActivationFunctions,
    double learningRate,
    bool randomlyInitWeights,
    double gradientDecay,
    double squaredGradientDecay,
    double epsilon
)
    : m_routerNetwork(routerNeuronsPerLayer, routerActivationFunctions, learningRate, randomlyInitWeights, gradientDecay, squaredGradientDecay, epsilon),
    m_topK{topK},
    m_expertOutputSize{*(expertNeuronsPerLayer.begin() + expertNeuronsPerLayer.size() - 1)}
{
    m_expertNetworks.reserve(experts);
    for (std::size_t i = 0; i < experts; ++i)
        m_expertNetworks.emplace_back(
            expertNeuronsPerLayer,
            expertActivationFunctions,
            learningRate,
            randomlyInitWeights,
            gradientDecay,
            squaredGradientDecay,
            epsilon
        );
}

void mixture_of_experts::load_router_weights(const std::string& filepath)
{
    m_routerNetwork.load_weights(filepath);
}

void mixture_of_experts::load_router_biases(const std::string& filepath)
{
    m_routerNetwork.load_biases(filepath);
}

void mixture_of_experts::load_expert_weights(std::size_t index, const std::string& filepath)
{
    m_expertNetworks[index].load_weights(filepath);
}

void mixture_of_experts::load_expert_biases(std::size_t index, const std::string& filepath)
{
    m_expertNetworks[index].load_biases(filepath);
}

math_vector<double> mixture_of_experts::Softmax(const math_vector<double>& logits)
{
    double maxRouterLogit = std::numeric_limits<double>::lowest();
    for (std::size_t i = 0; i < logits.size(); ++i)
        maxRouterLogit = std::max(maxRouterLogit, logits[i]);

    math_vector<double> probabilities(logits.size(), 0);
    double divideBy = 0;

    for (std::size_t i = 0; i < probabilities.size(); ++i)
    {
        probabilities[i] = std::exp(logits[i] - maxRouterLogit);
        divideBy += probabilities[i];
    }

    for (std::size_t i = 0; i < probabilities.size(); ++i)
        probabilities[i] = probabilities[i] / divideBy;

    return probabilities;
}

math_vector<double> mixture_of_experts::predict(
    const math_vector<double>& input,
    std::pmr::memory_resource* resource
) const
{
    // pass input through router:
    const math_vector<double> routerLogits{m_routerNetwork.forward(input, resource).back()},
                              expertProbabilities{Softmax(routerLogits)};

    // sort experts by router output:
    std::vector<std::size_t> sortedExpertIndices(expertProbabilities.size(), 0);
    for (std::size_t i = 1; i < m_expertNetworks.size(); ++i)
        sortedExpertIndices[i] = i;

    std::ranges::sort(sortedExpertIndices, [&](std::size_t lExpertIndex, std::size_t rExpertIndex) {
        return expertProbabilities[lExpertIndex] > expertProbabilities[rExpertIndex];
    });

    // compute logits, weighted by expert contribution:
    double topKProbabilitySum = 0;
    for (std::size_t i = 0; i < m_topK; ++i)
    {
        const std::size_t expertIndex = sortedExpertIndices[i];
        topKProbabilitySum += expertProbabilities[expertIndex];
    }

    math_vector<double> weightedExpertLogits(m_expertOutputSize, 0);
    for (std::size_t i = 0; i < m_topK; ++i)
    {
        const std::size_t expertIndex = sortedExpertIndices[i];
        const double weight = expertProbabilities[expertIndex] / topKProbabilitySum;
        const neural_network& expertNetwork{m_expertNetworks[expertIndex]};
        const math_vector<double> expertLogits{expertNetwork.forward(input, resource).back()};

        weightedExpertLogits += weight * expertLogits;
    }

    // probabilistic output:
    return Softmax(weightedExpertLogits);
}

void mixture_of_experts::fit(
    const math_vector<double>& input,
    const math_vector<double>& target,
    std::pmr::memory_resource* resource
)
{
    // pass input through router:
    const std::pmr::vector<math_vector<double>> routerActivations{m_routerNetwork.forward(input, resource)};
    const math_vector<double>& routerLogits{routerActivations.back()};
    const math_vector<double> expertProbabilities{Softmax(routerLogits)};

    // sort experts by router output:
    std::vector<std::size_t> sortedExpertIndices(expertProbabilities.size(), 0);
    for (std::size_t i = 1; i < m_expertNetworks.size(); ++i)
        sortedExpertIndices[i] = i;

    std::ranges::sort(sortedExpertIndices, [&](std::size_t lExpertIndex, std::size_t rExpertIndex) {
        return expertProbabilities[lExpertIndex] > expertProbabilities[rExpertIndex];
    });

    // compute logits weighted by expert contribution, and store activations:
    double topKProbabilitySum = 0;
    for (std::size_t i = 0; i < m_topK; ++i)
    {
        const std::size_t expertIndex = sortedExpertIndices[i];
        topKProbabilitySum += expertProbabilities[expertIndex];
    }

    std::vector<std::pmr::vector<math_vector<double>>> topKExpertActivations;
    topKExpertActivations.reserve(m_topK);

    for (std::size_t i = 0; i < m_topK; ++i)
    {
        const std::size_t expertIndex = sortedExpertIndices[i];
        const neural_network& expertNetwork{m_expertNetworks[expertIndex]};
        topKExpertActivations.emplace_back(expertNetwork.forward(input, resource));
    }

    math_vector<double> weightedExpertLogits(m_expertOutputSize, 0);
    for (std::size_t i = 0; i < m_topK; ++i)
    {
        const math_vector<double>& expertLogits{topKExpertActivations[i].back()};
        const std::size_t expertIndex = sortedExpertIndices[i];
        const double weight = expertProbabilities[expertIndex] / topKProbabilitySum;

        weightedExpertLogits += weight * expertLogits;
    }

    // compute loss gradient:
    const math_vector<double> prediction{Softmax(weightedExpertLogits)},
                              lossGradient{prediction - target};

    // backpropagate loss gradient through experts:
    for (std::size_t i = 0; i < m_topK; ++i)
    {
        const std::pmr::vector<math_vector<double>>& expertActivations{topKExpertActivations[i]};
        const std::size_t expertIndex = sortedExpertIndices[i];

        neural_network& expertNetworkRef{m_expertNetworks[expertIndex]};
        const double weight = expertProbabilities[expertIndex] / topKProbabilitySum;
        const math_vector<double> expertGradient{lossGradient * weight};

        expertNetworkRef.backward(expertActivations, expertGradient, false);
        expertNetworkRef.update_weights_and_biases();
    }

    // backpropagate loss gradient through router:
    math_vector<double> routerGradient(m_expertNetworks.size(), 0);

    for (std::size_t i = 0; i < m_topK; ++i)
    {
        const math_vector<double>& expertLogits{topKExpertActivations[i].back()};
        const std::size_t expertIndex = sortedExpertIndices[i];

        const double weight = expertProbabilities[expertIndex] / topKProbabilitySum;

        routerGradient[expertIndex] = weight * (lossGradient * expertLogits - lossGradient * weightedExpertLogits);
    }

    m_routerNetwork.backward(routerActivations, routerGradient, false);
    m_routerNetwork.update_weights_and_biases();
}