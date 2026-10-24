#include <iostream>
#include <print>
#include <string>
#include <string_view>
#include <set>
#include <map>
#include <fstream>
#include <memory_resource>
#include <chrono>
#include <thread>
#include <stop_token>
#include <cstddef>
#include "mixture_of_experts.hpp"

int main()
{
    std::ifstream file{"/home/cartercpp/Documents/C++/MixtureOfExpertsCharPrediction/lyrics.txt"};
    const std::string fileContent(std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{});
    const std::set<char> uniqueChars(fileContent.begin(), fileContent.end());
    std::map<char, std::size_t> charToIndex;

    {
        std::size_t index = 0;
        for (const char c : uniqueChars)
        {
            charToIndex[c] = index;
            ++index;
        }
    }

    constexpr std::size_t experts = 4,
                          topK = 2,
                          contextLength = 16; // 16 characters
    const std::size_t inputSize = uniqueChars.size() * contextLength;

    mixture_of_experts nextCharMOE(
        experts,
        topK,
        {inputSize, 64, 64, uniqueChars.size()},
        {inputSize, 32, experts},
        {ACTIVATION_FUNCTION::RELU, ACTIVATION_FUNCTION::RELU, ACTIVATION_FUNCTION::LINEAR},
        {ACTIVATION_FUNCTION::RELU, ACTIVATION_FUNCTION::LINEAR},
        0.001
    );

    // training thread:
    {
        std::jthread trainingThread{[&](std::stop_token st) {
            std::byte buffer[6'000'000];
            std::size_t epochs = 0;
            double averageLoss = 0;

            while (!st.stop_requested())
            {
                for (std::size_t start = 0; start + contextLength < fileContent.size(); ++start)
                {
                    // get context:
                    const std::string_view context(fileContent.data() + start, contextLength);

                    // convert context to one-hot encoding:
                    math_vector<double> oneHotEncodingInput(inputSize, 0);
                    for (std::size_t i = 0; i < contextLength; ++i)
                    {
                        const char ch = context[i];
                        const std::size_t index = i * uniqueChars.size() + charToIndex[ch];
                        oneHotEncodingInput[index] = 1;
                    }

                    // get target:
                    const char succeedingChar = fileContent[start + contextLength];
                    math_vector<double> target(uniqueChars.size(), 0);
                    target[charToIndex[succeedingChar]] = 1;

                    // fit:
                    std::pmr::monotonic_buffer_resource resource{buffer, sizeof(buffer)};
                    nextCharMOE.fit(oneHotEncodingInput, target, &resource);
                    ++epochs;

                    // compute average loss:
                    math_vector<double> prediction{nextCharMOE.predict(oneHotEncodingInput)};
                    averageLoss += (prediction - target) * (prediction - target) / 500;

                    // display loss every now and then:
                    if (epochs % 500 == 0)
                    {
                        std::println("Loss: {:.2f}", averageLoss);
                        averageLoss = 0;
                    }
                }
            }
        }};
        std::cin.get();
    }

    // display:
    std::cout << "\033[2J\033[0H";

    std::string output;
    output.reserve(fileContent.size());
    output.insert(output.end(), fileContent.begin(), fileContent.begin() + contextLength);

    std::cout << output;

    for (std::size_t start = 0; start + contextLength < fileContent.size(); ++start)
    {
        // get context:
        const std::string_view context(output.end() - contextLength, output.end());

        // convert context to one-hot encoding:
        math_vector<double> oneHotEncodingInput(inputSize, 0);
        for (std::size_t i = 0; i < contextLength; ++i)
        {
            const char ch = context[i];
            const std::size_t index = i * uniqueChars.size() + charToIndex[ch];
            oneHotEncodingInput[index] = 1;
        }

        // get char probabilities:
        const math_vector<double> charProbabilities{nextCharMOE.predict(oneHotEncodingInput)};
        double maxProbability = 0;
        char nextChar = ' ';

        for (const auto [ch, chIndex]: charToIndex)
            if (charProbabilities[chIndex] > maxProbability)
            {
                maxProbability = charProbabilities[chIndex];
                nextChar = ch;
            }

        output.push_back(nextChar);
        std::cout << nextChar << std::flush;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}