//
// Created by cartercpp on 10/24/26.
//

#pragma once

#include <string>
#include <initializer_list>
#include <vector>
#include <memory_resource>
#include <cstddef>
#include "neural_network.hpp"

class mixture_of_experts
{
public:

    // CONSTRUCTORS

    explicit mixture_of_experts(
        std::size_t experts,
        std::size_t topK,
        std::initializer_list<std::size_t> expertNeuronsPerLayer,
        std::initializer_list<std::size_t> routerNeuronsPerLayer,
        std::initializer_list<ACTIVATION_FUNCTION> expertActivationFunctions,
        std::initializer_list<ACTIVATION_FUNCTION> routerActivationFunctions,
        double learningRate,
        bool randomlyInitWeights = true,
        double gradientDecay = 0.9,
        double squaredGradientDecay = 0.999,
        double epsilon = 1e-8
    )
        pre(experts > 0)
        pre((topK > 0) && (topK <= experts))
        pre(expertNeuronsPerLayer.size() >= 3)
        pre(routerNeuronsPerLayer.size() >= 3)
        pre(expertNeuronsPerLayer.size() - 1 == expertActivationFunctions.size())
        pre(routerNeuronsPerLayer.size() - 1 == routerActivationFunctions.size())
        pre(routerNeuronsPerLayer.end()[-1] == experts)
        pre(expertActivationFunctions.end()[-1] == ACTIVATION_FUNCTION::LINEAR)
        pre(routerActivationFunctions.end()[-1] == ACTIVATION_FUNCTION::LINEAR)
        pre(*expertNeuronsPerLayer.begin() == *routerNeuronsPerLayer.begin());

    // METHODS

    void load_router_weights(const std::string& filepath);

    void load_router_biases(const std::string& filepath);

    void load_expert_weights(std::size_t index, const std::string& filepath)
        pre(index < m_expertNetworks.size());

    void load_expert_biases(std::size_t index, const std::string& filepath)
        pre(index < m_expertNetworks.size());

    math_vector<double> predict(
        const math_vector<double>& input,
        std::pmr::memory_resource* resource = std::pmr::get_default_resource()
    ) const;

    void fit(
        const math_vector<double>& input,
        const math_vector<double>& target,
        std::pmr::memory_resource* resource = std::pmr::get_default_resource()
    );

private:

    static math_vector<double> Softmax(const math_vector<double>& logits);

    std::vector<neural_network> m_expertNetworks;
    neural_network m_routerNetwork;
    std::size_t m_topK,
                m_expertOutputSize;
};