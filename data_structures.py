# module: data_structures

import torch
import numpy as np
from typing import List, Dict, NamedTuple, Optional

# Recommended: Use dataclasses if Python 3.7+ is guaranteed
# from dataclasses import dataclass

class Player:
    """
    Represents a player in the game. [cite: 31]

    Attributes:
        name (str): Unique identifier for the player. [cite: 31]
        actions (List[str]): List of action names available to the player. [cite: 32]
        num_actions (int): The number of actions. [cite: 32]
        strategy_params (torch.Tensor): Learnable parameters in Euclidean space,
                                       requires_grad=True. Shape: (num_actions,). [cite: 33]
    """
    def __init__(self, name: str, actions: List[str]):
        if not name:
            raise ValueError("Player name must be non-empty.") # Invariant [cite: 34]
        if not actions:
            raise ValueError("Player actions list must not be empty.") # Invariant [cite: 34]

        self.name: str = name
        self.actions: List[str] = actions
        self.num_actions: int = len(actions)
        # Initialize parameters randomly, requires grad [cite: 33]
        self.strategy_params: torch.Tensor = torch.randn(self.num_actions, requires_grad=True)
        # Ensure it's a 1D tensor of the correct size [cite: 35]
        assert self.strategy_params.shape == (self.num_actions,) and self.strategy_params.dim() == 1

    def __repr__(self) -> str:
        return f"Player(name='{self.name}', num_actions={self.num_actions})"

# Using NamedTuple as a lightweight data structure, similar to dataclass [cite: 35]
class Strategy(NamedTuple):
    """
    Represents a concrete probability distribution over a player's actions. [cite: 35]
    Detached from the computation graph. [cite: 36]

    Attributes:
        player_name (str): The name of the player this strategy belongs to. [cite: 35]
        probabilities (np.ndarray): Probability distribution over actions. Shape: (num_actions,). [cite: 35]
    """
    player_name: str
    probabilities: np.ndarray

    def __post_init__(self):
        # Invariant checks [cite: 36]
        if not np.isclose(np.sum(self.probabilities), 1.0):
            raise ValueError(f"Strategy probabilities for {self.player_name} must sum to 1. Got: {np.sum(self.probabilities)}")
        if not np.all((self.probabilities >= 0) & (self.probabilities <= 1)):
            raise ValueError(f"Strategy probabilities for {self.player_name} must be between 0 and 1.")

class Game:
    """
    Represents the N-player game structure including players and payoff tensors. [cite: 37]

    Attributes:
        players (List[Player]): List of participating players. [cite: 37]
        payoff_tensors (Dict[str, torch.Tensor]): Maps player names to their payoff tensors (converted to torch tensors). [cite: 38]
                                                 Shape depends on all players' action counts. [cite: 39, 40]
    """
    def __init__(self, players: List[Player], payoff_tensors: Dict[str, np.ndarray]):
        if not players:
            raise ValueError("Game must contain at least one player.") # Invariant [cite: 41]

        self.players: List[Player] = players
        self._player_map: Dict[str, Player] = {p.name: p for p in players}

        if set(payoff_tensors.keys()) != set(self._player_map.keys()):
             raise ValueError("payoff_tensors must contain an entry for every player.") # Invariant [cite: 42]

        # Convert payoff tensors from NumPy to PyTorch Tensors during initialization
        self.payoff_tensors: Dict[str, torch.Tensor] = {
            name: torch.from_numpy(tensor).float()
            for name, tensor in payoff_tensors.items()
        }

        # Validate payoff tensor dimensions [cite: 43]
        expected_shape = tuple(p.num_actions for p in self.players) # Shape (k_1, k_2, ..., k_N) [cite: 40]
        for player_name, tensor in self.payoff_tensors.items(): # Now checking torch.Tensor shapes
            if tensor.shape != expected_shape:
                raise ValueError(f"Payoff tensor shape for player '{player_name}' is {tensor.shape}, "
                                 f"but expected {expected_shape} based on player action counts.")

    def get_player_by_name(self, name: str) -> Optional[Player]:
        """Retrieves a player object by their unique name. [cite: 41]"""
        return self._player_map.get(name)

    def __repr__(self) -> str:
        player_names = [p.name for p in self.players]
        return f"Game(players={player_names})"

# Using NamedTuple for simplicity [cite: 44]
class TrainingStatus(NamedTuple):
    """
    Encapsulates the state of the training at a specific iteration. [cite: 48]

    Attributes:
        iteration (int): Current training step number. [cite: 44]
        player_losses (Dict[str, float]): Maps player names to their individual losses for this iteration. [cite: 46]
        current_strategies (Dict[str, Strategy]): Snapshot of all players' strategies at this iteration. [cite: 47]
    """
    iteration: int
    player_losses: Dict[str, float]
    current_strategies: Dict[str, Strategy]

# Example Usage (Optional - for demonstration)
if __name__ == '__main__':
    # Setup a simple 2-player game (e.g., Matching Pennies)
    player1 = Player(name="P1", actions=["Heads", "Tails"])
    player2 = Player(name="P2", actions=["Heads", "Tails"])
    players = [player1, player2]

    # Payoff matrix for P1: [[1, -1], [-1, 1]]
    # Payoff matrix for P2: [[-1, 1], [1, -1]]
    # Combined payoff tensor shape: (P1_actions, P2_actions) -> (2, 2)
    payoffs_p1 = np.array([[1, -1], [-1, 1]])
    payoffs_p2 = np.array([[-1, 1], [1, -1]])
    payoff_tensors = {"P1": payoffs_p1, "P2": payoffs_p2} # [cite: 38]

    try:
        game = Game(players=players, payoff_tensors=payoff_tensors) # [cite: 37]
        print(f"Game created: {game}")
        print(f"Player 1 details: {player1}")
        print(f"Player 1 initial strategy params (requires_grad={player1.strategy_params.requires_grad}): {player1.strategy_params}") # [cite: 33]

        # Example Strategy (assuming probabilities derived from params later)
        strategy_p1 = Strategy(player_name="P1", probabilities=np.array([0.5, 0.5])) # [cite: 35]
        strategy_p2 = Strategy(player_name="P2", probabilities=np.array([0.5, 0.5]))

        # Example TrainingStatus without global_loss
        status = TrainingStatus(iteration=0,
                                player_losses={"P1": 0.0, "P2": 0.0}, # [cite: 46]
                                current_strategies={"P1": strategy_p1, "P2": strategy_p2}) # [cite: 47]
        print(f"Initial Training Status: {status}")

    except ValueError as e:
        print(f"Error setting up game: {e}")