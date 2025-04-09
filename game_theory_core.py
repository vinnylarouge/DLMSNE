# module: game_theory_core

import torch
import numpy as np
from typing import Dict

# Assuming data_structures module is available and contains the Game class
# from data_structures import Game
# If running independently, include the Game definition or a placeholder:
try:
    from data_structures import Game, Player
except ImportError:
    # Placeholder if data_structures isn't directly importable
    # Define necessary structures minimally for type hinting
    class Player:
        def __init__(self, name: str, num_actions: int):
            self.name = name
            self.num_actions = num_actions
            self.strategy_params = torch.randn(num_actions, requires_grad=True)

    class Game:
        def __init__(self, players: list[Player], payoff_tensors: Dict[str, np.ndarray]):
            self.players = players
            self.payoff_tensors = {name: torch.from_numpy(tensor).float() for name, tensor in payoff_tensors.items()} # Convert to torch tensors
            self._player_map = {p.name: p for p in players}

        def get_player_by_name(self, name: str) -> Player | None:
             return self._player_map.get(name)


def project_to_simplex(params: torch.Tensor) -> torch.Tensor:
    """
    Projects raw strategy parameters onto the probability simplex using Softmax.

    Ensures the output is a valid probability distribution (non-negative, sums to 1)
    and remains differentiable. [cite: 54, 55]

    Args:
        params: A 1D torch.Tensor representing raw strategy parameters.

    Returns:
        A 1D torch.Tensor representing the probability distribution on the simplex. [cite: 53]

    Raises:
        ValueError: If the input tensor is not 1D.
    """
    if params.dim() != 1:
        raise ValueError(f"Input tensor must be 1D, but got {params.dim()} dimensions.") # Condition
    # Softmax ensures non-negativity and sum-to-one, and is differentiable [cite: 55]
    probabilities = torch.softmax(params, dim=0)
    # Check invariants (optional, softmax guarantees this)
    # assert torch.all(probabilities >= 0)
    # assert torch.isclose(torch.sum(probabilities), torch.tensor(1.0))
    return probabilities #


def calculate_expected_payoffs(game: Game, strategy_profile: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Calculates the expected payoff for each player given a full strategy profile.

    Uses tensor contraction (einsum) for efficient computation. [cite: 59, 60]

    Args:
        game: The Game object containing players and payoff tensors (as torch.Tensors).
        strategy_profile: A dictionary mapping player names to their current
                          PROJECTED strategy tensors (probability distributions). [cite: 57]

    Returns:
        A dictionary mapping player names to their scalar expected payoff
        (as a torch.Tensor of size 1 to preserve gradient information). [cite: 58]

    Raises:
        ValueError: If strategy_profile doesn't contain strategies for all players.
    """
    player_names = [p.name for p in game.players]
    if set(strategy_profile.keys()) != set(player_names):
        raise ValueError("strategy_profile must contain strategies for all players in the game.") # Condition [cite: 62]

    num_players = len(game.players)
    strategy_tensors = [strategy_profile[p.name] for p in game.players] # Ordered list of strategy tensors

    expected_payoffs = {}

    # Einsum notation:
    # Payoff tensors have indices corresponding to player actions (e.g., ijk for 3 players)
    # Strategy tensors have one index (e.g., i for player 1, j for player 2, k for player 3)
    # Expected payoff = Sum_{i,j,k,...} Payoff(i,j,k,...) * Strat_1(i) * Strat_2(j) * Strat_3(k) * ...
    # The einsum string dynamically builds this contraction.
    indices = "abcdefghijklmnopqrstuvwxyz"[:num_players]
    strategy_indices = [f"{indices[i]}" for i in range(num_players)]
    einsum_str = f"{indices}," + ",".join(strategy_indices) + "->" # Contract payoff with all strategies

    for i, player in enumerate(game.players):
        player_name = player.name
        payoff_tensor = game.payoff_tensors[player_name] # Already a torch.Tensor

        # Check dimensions match (optional, should be guaranteed by Game init)
        # expected_shape = tuple(p.num_actions for p in game.players)
        # assert payoff_tensor.shape == expected_shape
        # for j, strat_tensor in enumerate(strategy_tensors):
        #     assert strat_tensor.shape == (game.players[j].num_actions,)

        # Perform the tensor contraction [cite: 59, 60]
        # Operands for einsum: Payoff tensor followed by all strategy tensors
        operands = [payoff_tensor] + strategy_tensors
        expected_payoff = torch.einsum(einsum_str, operands) # Results in a scalar tensor

        # Store the scalar tensor (maintains gradient info) [cite: 58]
        expected_payoffs[player_name] = expected_payoff.unsqueeze(0) # Ensure it's size [1] technically

    return expected_payoffs


# Example Usage (Optional - for demonstration)
if __name__ == '__main__':
    # Re-setup the simple 2-player game (Matching Pennies)
    player1 = Player(name="P1", num_actions=2)
    player2 = Player(name="P2", num_actions=2)
    players = [player1, player2]

    payoffs_p1_np = np.array([[1.0, -1.0], [-1.0, 1.0]])
    payoffs_p2_np = np.array([[-1.0, 1.0], [1.0, -1.0]])
    payoff_tensors_np = {"P1": payoffs_p1_np, "P2": payoffs_p2_np}

    game = Game(players=players, payoff_tensors=payoff_tensors_np)

    # --- Test project_to_simplex ---
    raw_params_p1 = torch.tensor([0.0, 0.0], dtype=torch.float32) # Should yield [0.5, 0.5]
    proj_strat_p1 = project_to_simplex(raw_params_p1)
    print(f"Projected strategy for P1 (params {raw_params_p1}): {proj_strat_p1}")
    print(f"Sum: {torch.sum(proj_strat_p1)}")

    raw_params_p2 = torch.tensor([10.0, -10.0], dtype=torch.float32) # Should yield [~1, ~0]
    proj_strat_p2 = project_to_simplex(raw_params_p2)
    print(f"Projected strategy for P2 (params {raw_params_p2}): {proj_strat_p2}")
    print(f"Sum: {torch.sum(proj_strat_p2)}")

    # --- Test calculate_expected_payoffs ---
    # Example profile: P1 plays [0.5, 0.5], P2 plays [0.5, 0.5]
    strategy_prof_5050 = {
        "P1": torch.tensor([0.5, 0.5], dtype=torch.float32),
        "P2": torch.tensor([0.5, 0.5], dtype=torch.float32)
    }
    expected_payoffs_5050 = calculate_expected_payoffs(game, strategy_prof_5050)
    print(f"\nExpected Payoffs for profile {strategy_prof_5050}:")
    for name, payoff in expected_payoffs_5050.items():
        print(f"  {name}: {payoff.item():.4f}") # Expect 0.0 for both in Matching Pennies

    # Example profile: P1 plays [1.0, 0.0], P2 plays [0.0, 1.0]
    strategy_prof_corners = {
        "P1": torch.tensor([1.0, 0.0], dtype=torch.float32),
        "P2": torch.tensor([0.0, 1.0], dtype=torch.float32)
    }
    expected_payoffs_corners = calculate_expected_payoffs(game, strategy_prof_corners)
    print(f"\nExpected Payoffs for profile {strategy_prof_corners}:")
    for name, payoff in expected_payoffs_corners.items():
        # P1 plays H, P2 plays T -> P1 gets -1, P2 gets 1
        print(f"  {name}: {payoff.item():.4f}")

    # --- Test Gradient Flow (Simple) ---
    params_p1_grad = torch.tensor([0.1, -0.1], dtype=torch.float32, requires_grad=True)
    params_p2_grad = torch.tensor([0.2, 0.3], dtype=torch.float32, requires_grad=True)

    strat_p1_grad = project_to_simplex(params_p1_grad)
    strat_p2_grad = project_to_simplex(params_p2_grad)

    profile_grad = {"P1": strat_p1_grad, "P2": strat_p2_grad}
    payoffs_grad = calculate_expected_payoffs(game, profile_grad)

    # Calculate gradient of P1's payoff w.r.t P1's params
    payoffs_grad["P1"].backward()
    print(f"\nGradient of P1's payoff w.r.t P1's params: {params_p1_grad.grad}")
    print(f"Gradient of P1's payoff w.r.t P2's params (should be None): {params_p2_grad.grad}")