# Mixture of Experts — Next-Character Prediction

A **character-level Mixture of Experts (MoE)** model written from scratch in **C++26**. It learns to predict the next character from a short text context and then generates text autoregressively, one character at a time.

The neural networks, routing, backpropagation, and Adam optimizer are implemented without PyTorch, TensorFlow, or other machine-learning libraries.

## Architecture

The vocabulary is built from the unique characters in the training file, so the input and output sizes adapt to the text.

| Component | Configuration |
| --- | --- |
| Context | 16 characters |
| Encoding | One-hot (16 × vocabulary size) |
| Experts | 4 feedforward neural networks |
| Active experts | Top 2 per input |
| Each expert | Input → 64 (ReLU) → 64 (ReLU) → vocabulary size (linear) |
| Router | Input → 32 (ReLU) → 4 (linear) |
| Output | Softmax over the combined expert logits |
| Optimizer | Adam, learning rate 0.001 |

With a 34-character vocabulary, each context becomes a **544-element input vector**, and the model predicts one of 34 possible next characters.

## How it works

1. **Prepare data:** Slide a 16-character window across a text file. One-hot encode the context and use the following character as the target.
2. **Route:** The router produces four scores. Softmax converts them to probabilities, and the two highest-scoring experts are selected.
3. **Predict:** The selected experts produce character logits. Their outputs are combined using renormalized routing weights and passed through softmax.
4. **Train:** Backpropagate through the active experts and the router; update parameters using a from-scratch Adam implementation.
5. **Generate:** Seed the model with the first 16 characters of the file. Repeatedly choose the highest-probability next character, append it, and use the latest 16 characters as the next input.

The terminal reports the **mean squared prediction error** every 500 training updates. This displayed metric is separate from the softmax/cross-entropy-style output gradient used during training.

Because the model is trained on a single short text, it can **memorize much of that text**. It is a small character-level demonstration, not a pretrained general-purpose language model.

## Build and run

**Requirements:** GCC 16 with C++26 contracts support (or a compatible compiler), and a terminal.

This repository does **not** include the training lyrics. Supply your own plain-text training file.

1. Create a `lyrics.txt` file with the text you want to train on.
2. In `main.cpp`, change the **hard-coded absolute path** in the `std::ifstream file{...}` declaration to the path of your file. For example, use `"lyrics.txt"` if running from the project directory.
3. Compile:

```bash
g++-16 -std=gnu++26 -O3 -march=native -pthread -DNDEBUG \
    -fcontracts -fcontract-evaluation-semantic=ignore \
    main.cpp mixture_of_experts.cpp neural_network.cpp \
    -o moe
```

4. Run:

```bash
./moe
```

The program trains until you **press Enter**, then stops training and generates characters in the terminal with a short typing delay. It does not currently save a model checkpoint, so a new run starts with newly initialized weights.

> The sample program uses a 6 MB stack buffer inside its training thread. On systems with smaller thread stacks, you may need to move this buffer to heap storage.

## Source files

- `main.cpp` — dataset preparation, training loop, and autoregressive generation
- `mixture_of_experts.hpp/.cpp` — softmax router, Top-2 expert selection, and MoE training
- `neural_network.hpp/.cpp` — feedforward networks, backpropagation, and Adam
- `math_vector.hpp` and `matrix.hpp` — custom numerical types

The project also experiments with modern C++ features such as **C++26 contracts**, `std::jthread`, `std::string_view`, ranges, and polymorphic memory resources.
