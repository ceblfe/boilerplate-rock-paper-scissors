# Rock Paper Scissors

Rock, Paper, Scissors. This project is the first project to get the Machine Learning with Python Certification from freeCodeCamp.


## Overview

This project implements a Rock, Paper, Scissors (RPS) game in Python, designed as a challenge where a custom `player` function competes against four bot opponents (`quincy`, `mrugesh`, `kris`, `abbey`). The goal is to achieve at least a 60% win rate against each bot over 1000 games. The `player` function uses a neural network (TensorFlow) and pattern detection to predict and counter opponent moves, with additional logic to identify specific bot strategies.

Key features:
- Simulates RPS games with a `play` function that tracks wins, losses, and ties.
- Includes four bot opponents with distinct strategies (e.g., `quincy` cycles through a fixed sequence, `abbey` uses opponent history patterns).
- Implements a `player` function combining machine learning (neural network) and rule-based detection to counter bots.
- Provides an interactive mode to play as a human against bots.
- Includes unit tests to verify the `player` function's performance.

The project achieves the required ≥60% win rate against all bots, as validated by the unit tests in `test_module.py`.

## Files

- **main.py**: Entry point for the project. Runs the `player` function against each bot for 1000 games, includes options for interactive human play, and executes unit tests.
- **RPS_game.py**: Core game logic. Defines the `play` function and the four bot strategies (`quincy`, `mrugesh`, `kris`, `abbey`), plus `human` and `random_player` for interactive/testing modes. **Note**: This file should not be modified.
- **RPS.py**: Contains the custom `player` function, which uses a TensorFlow neural network to predict opponent moves based on game history. Includes detection logic for each bot's strategy and a pre-training step with random data.
- **test_module.py**: Unit tests to verify that the `player` function achieves at least a 60% win rate against each bot over 1000 games.

## Requirements

- **Python**: 3.8+ (tested in Python 3.x environments).
- **Libraries**:
  - TensorFlow 2.x (for neural network in `player`)
  - NumPy
  - Random (standard library)
- Install dependencies via pip:
  ```bash
  pip install tensorflow numpy
  ```

## Usage

1. **Setup**:
   - Ensure all files (`main.py`, `RPS_game.py`, `RPS.py`, `test_module.py`) are in the same directory.
   - Install required libraries (see above).

2. **Running the Game**:
   - Execute `main.py` to run the `player` function against each bot for 1000 games:
     ```bash
     python main.py
     ```
   - Output includes final results (wins, losses, ties) and the `player` win rate for each bot.

3. **Interactive Play**:
   - Uncomment the line in `main.py` for human play:
     ```python
     play(human, abbey, 20, verbose=True)
     ```
   - Run `main.py` and input `R`, `P`, or `S` to play 20 games against `abbey` (or modify to play against other bots).

4. **Random Opponent**:
   - Uncomment the line in `main.py` to play against a random bot:
     ```python
     play(human, random_player, 1000)
     ```

5. **Unit Tests**:
   - Uncomment the line in `main.py` to run tests:
     ```python
     main(module='test_module', exit=False)
     ```
   - Tests verify that the `player` function achieves ≥60% win rate against each bot.

Example output from `main.py`:
```
Final results: {'p1': 650, 'p2': 300, 'tie': 50}
Player 1 win rate: 68.42%
```

## How It Works

- **Game Logic (`RPS_game.py`)**:
  - The `play` function runs a specified number of RPS games, tracking moves and determining winners (Rock beats Scissors, Scissors beats Paper, Paper beats Rock).
  - Bots have unique strategies:
    - `quincy`: Cycles through ["R", "R", "P", "P", "S"].
    - `mrugesh`: Counters the most frequent move in the opponent's last 10 plays.
    - `kris`: Counters the opponent's previous move.
    - `abbey`: Predicts based on the opponent's last two moves and counters the most likely next move.

- **Player Strategy (`RPS.py`)**:
  - Uses a neural network to predict the opponent's next move based on the last 5 moves of both players (encoded as one-hot vectors).
  - Detects specific bots by simulating their strategies and comparing with the opponent's history.
  - Falls back to neural network predictions if no bot is detected.
  - Pre-trains the model with 50 random samples for better initial performance.

- **Testing (`test_module.py`)**:
  - Runs 1000 games against each bot and checks if the win rate is ≥60%.

## Results

- The `player` function consistently achieves >60% win rate against all bots, passing all unit tests.
- Performance can vary slightly due to randomness in the neural network and initial moves.
- The neural network improves over time as it trains on game history, supplemented by bot-specific strategies for faster wins.

## Troubleshooting

- **TensorFlow Errors**: Ensure TensorFlow is installed correctly. If GPU issues arise, run on CPU or check CUDA/cuDNN compatibility.
- **Low Win Rate**: If tests fail, verify the neural network training in `RPS.py` (e.g., increase `num_samples` in `simple_pretrain` or adjust model architecture).
- **Bot Detection Issues**: Ensure opponent history is long enough (e.g., ≥10 for `quincy`, `abbey`) for accurate detection.

## Improvements

- Increase the neural network's capacity (e.g., add layers or units) for better prediction.
- Extend the history length in `get_state` (e.g., from 5 to 10 moves) for more context.
- Add more pre-training data or epochs in `simple_pretrain` for better initial performance.
- Experiment with learning rates or optimizers in the neural network.

## License

This project is for educational purposes, likely part of a freeCodeCamp or similar coding challenge. No explicit license is provided, but the code is intended for learning and modification (except `RPS_game.py`).

## Credits

- Built with Python, TensorFlow, and NumPy.
- Inspired by RPS challenges from platforms like freeCodeCamp.

For questions or contributions, contact the author or fork the repository. Last updated: September 10, 2025.





