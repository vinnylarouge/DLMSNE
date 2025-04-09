# DLMSNE

Deep Learning for Mixed Strategy Nash Equilibria.

This project provides a tool to find Mixed Strategy Nash Equilibria (MSNE) for N-player normal-form games using gradient-based optimization methods inspired by deep learning techniques. It includes a graphical user interface (GUI) for setting up games, running the optimization, and visualizing the results.

## Features

*   Finds approximate MSNE for N-player games.
*   Uses PyTorch for gradient calculations and optimization.
*   PyQt6 GUI for easy game setup and interaction.
*   Visualization of strategy evolution on the simplex (for 2-5 actions or selected subsets).
*   Visualization of player losses over iterations.
*   Customizable number of players and actions.
*   Random payoff initialization for quick setup.

## How to Use

### 1. Installation

Clone the repository and navigate into the project directory:

```bash
git clone <your-repository-url> DLMSNE
cd DLMSNE
```

Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```
*(Note: You might need to create a `requirements.txt` file first if you don't have one. A basic one would include `numpy`, `torch`, `matplotlib`, `PyQt6`)*

### 2. Running the GUI

Launch the main graphical interface:

```bash
python gui.py
```

### 3. Using the Interface

1.  **Game Definition:**
    *   Select the **Number of Players** and **Number of Actions** using the spin boxes. Player names and default action names will populate automatically.
    *   You can edit the player names and the comma-separated action names in the text fields if needed.
2.  **Setup Game Structure:**
    *   Click the "Setup Game Structure" button. This validates the player/action setup and creates the payoff table below, initializing payoffs randomly between -10 and 10.
    *   The Game Definition section will hide.
    *   **Important:** If the combination of players and actions leads to too many possible outcomes (currently > 50,000), setup will be prevented to avoid performance issues.
3.  **Verify/Edit Payoffs:**
    *   Review the randomly generated payoffs in the table. Edit any values as needed by double-clicking the cells. The table is scrollable if it's large.
4.  **Initialize Training:**
    *   Click the "Initialize Training" button. This creates the internal game representation and prepares the optimization algorithm. The payoff table becomes read-only.
5.  **Run Training:**
    *   Click the "▶ Start" button to begin the optimization process.
    *   The status label will show the current iteration, and the console log will update periodically (default: every 100 iterations).
    *   The Loss Plot and Strategy Simplex Plot (if applicable) will update live.
6.  **Controls:**
    *   **❚❚ Pause / ▶ Resume:** Pause or resume the optimization.
    *   **■ Reset:** Stop the current training run completely, clear results, and return to the Game Definition stage. This requires re-initializing before starting again.
    *   **▶ Continue (+N iters):** After a run finishes, this button appears. Clicking it runs the optimization for an additional N iterations (default: 1000) from the current state.
7.  **Action Selection (for >5 Actions):**
    *   If the game has more than 5 actions, the "Select Actions for Plot" button appears after initialization.
    *   Click it to open a dialog where you can choose 2 to 5 actions to visualize on the simplex plot. The plot will update upon selection.

## TODO

*   [ ] Implement payoff input validation (e.g., ensure numbers are entered).
*   [ ] Add option to save/load game definitions (players, actions, payoffs).
*   [ ] Explore alternative optimization algorithms (e.g., SGD, RMSprop).
*   [ ] Add more sophisticated simplex visualization for > 3 dimensions (e.g., PCA projection).
*   [ ] Improve performance for handling very large payoff tables (e.g., lazy loading).
*   [ ] Add convergence criteria/stopping conditions besides max iterations.
*   [ ] Package the application for easier distribution.
*   [ ] Write more comprehensive tests.

## Development Note

This codebase was initially generated through interaction with Google's Gemini 2.5 Advanced.

1.  The process started from a high-level PDF sketch outlining the desired functionality (see `spec/` folder).
2.  Gemini was prompted to propose a modular software architecture based on the sketch, providing justifications for the design choices (the resulting architecture description PDF is also in `spec/`).
3.  Code for the individual Python modules (`data_structures.py`, `game_theory_core.py`, `optimizer.py`, `visualization.py`, `gui.py`) was then generated iteratively by Gemini.
4.  Subsequent development, debugging, GUI refinement, and feature additions were performed using Cursor, interacting with its AI features.
