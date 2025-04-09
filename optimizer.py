# module: optimizer

import torch
import torch.optim as optim
from typing import Dict, Type

# Assuming data_structures and game_theory_core modules are available
# from data_structures import Game, Player, Strategy, TrainingStatus
# from game_theory_core import project_to_simplex, calculate_expected_payoffs

# If running independently, include necessary definitions or placeholders:
try:
    from data_structures import Game, Player, Strategy, TrainingStatus
    from game_theory_core import project_to_simplex, calculate_expected_payoffs
except ImportError:
    # Minimal placeholders for type hinting
    import numpy as np
    from typing import NamedTuple, Optional, List

    class Player:
        def __init__(self, name: str, num_actions: int):
            self.name = name
            self.num_actions = num_actions
            self.strategy_params = torch.randn(num_actions, requires_grad=True)

    class Game:
        def __init__(self, players: list[Player], payoff_tensors: Dict[str, np.ndarray]):
            self.players = players
            self.payoff_tensors = {name: torch.from_numpy(tensor).float() for name, tensor in payoff_tensors.items()}
            self._player_map = {p.name: p for p in players}
        def get_player_by_name(self, name: str) -> Player | None: return self._player_map.get(name)

    class Strategy(NamedTuple):
        player_name: str
        probabilities: np.ndarray

    class TrainingStatus(NamedTuple):
        iteration: int
        player_losses: Dict[str, float]
        current_strategies: Dict[str, Strategy]

    def project_to_simplex(params: torch.Tensor) -> torch.Tensor:
        return torch.softmax(params, dim=0)

    def calculate_expected_payoffs(game: Game, strategy_profile: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Dummy implementation for placeholder
        payoffs = {}
        for p_name in strategy_profile:
             payoffs[p_name] = torch.tensor([0.0]) # Placeholder
        return payoffs


class MSNEFinder:
    """
    Implements the iterative algorithm to find Mixed Strategy Nash Equilibria (MSNE)
    using backpropagation, treating opponent strategies as fixed during each player's update.

    Attributes:
        game (Game): The game object.
        learning_rate (float): The learning rate for the optimizers.
        optimizers (Dict[str, optim.Optimizer]): Optimizers for each player's strategy parameters.
        _iteration_count (int): Internal counter for iterations.
    """
    def __init__(self, game: Game, learning_rate: float, optimizer_cls: Type[optim.Optimizer] = optim.Adam):
        """
        Initializes the MSNEFinder.

        Args:
            game: The Game object.
            learning_rate: The learning rate for optimization.
            optimizer_cls: The PyTorch optimizer class to use (default: Adam).
        """
        self.game: Game = game
        self.learning_rate: float = learning_rate
        self.optimizers: Dict[str, optim.Optimizer] = {}
        self._iteration_count: int = 0

        # Create an optimizer for each player, linked ONLY to their strategy_params
        for player in self.game.players:
            if not player.strategy_params.requires_grad:
                # Ensure parameters require gradients
                player.strategy_params.requires_grad_(True)
            self.optimizers[player.name] = optimizer_cls([player.strategy_params], lr=self.learning_rate) # Pass params as a list

        # Invariant check: Verify optimizers are linked correctly (optional)
        # for name, opt in self.optimizers.items():
        #     player = self.game.get_player_by_name(name)
        #     assert len(opt.param_groups) == 1
        #     assert len(opt.param_groups[0]['params']) == 1
        #     assert opt.param_groups[0]['params'][0] is player.strategy_params # Check object identity


    def step(self) -> TrainingStatus:
        """
        Performs one full update iteration for all players.

        Calculates loss for each player assuming others' strategies are fixed (using detach),
        computes gradients only for the current player's parameters, and updates parameters.

        Returns:
            TrainingStatus: An object containing the status after this iteration.
        """
        # 1. Get current projected strategies for all players (still attached to graph here)
        current_projections = {
            p.name: project_to_simplex(p.strategy_params) for p in self.game.players
        }

        player_losses_dict = {}

        # 2. Iterate through each player to calculate their individual loss and gradients
        for current_player in self.game.players:
            player_name = current_player.name
            optimizer = self.optimizers[player_name]

            # 2a. Create the specific strategy profile for this player's calculation
            profile_for_player = {}
            for other_player in self.game.players:
                other_name = other_player.name
                # Use the projected strategy
                strategy_tensor = current_projections[other_name]

                # *** CRUCIAL STEP: Detach opponent strategies *** [cite: 75, 76]
                # If it's not the current player, detach the tensor from the computation graph.
                # This treats their strategy as a fixed constant for the current player's gradient calculation.
                if other_name != player_name:
                    profile_for_player[other_name] = strategy_tensor.detach()
                else:
                    # Use the current player's strategy directly (it needs to stay attached)
                    profile_for_player[other_name] = strategy_tensor

            # 2b. Calculate expected payoffs using this tailored profile
            # Payoffs are calculated based on the perspective where only the current player's
            # strategy influences the gradients.
            expected_payoffs = calculate_expected_payoffs(self.game, profile_for_player)

            # 2c. Define loss as negative expected payoff (for maximization via gradient descent) [cite: 78]
            loss = -expected_payoffs[player_name]
            player_losses_dict[player_name] = loss.item() # Store scalar loss value

            # 2d. Backpropagate the loss for the current player
            optimizer.zero_grad() # Clear previous gradients for this player ONLY [cite: 79]
            # Compute gradients of the loss w.r.t. the current player's strategy_params ONLY,
            # because other strategies were detached. [cite: 79, 80]
            loss.backward()

            # Gradient clipping (optional, can help stability)
            # torch.nn.utils.clip_grad_norm_(current_player.strategy_params, max_norm=1.0)

        # 3. Update all players' parameters using their respective optimizers [cite: 81]
        # This step applies the gradients calculated in the loop above.
        for player_name, optimizer in self.optimizers.items():
            optimizer.step()

        # 4. Prepare and return TrainingStatus [cite: 82]
        self._iteration_count += 1
        final_strategies = self.get_current_strategies() # Get detached strategies for reporting

        status = TrainingStatus(
            iteration=self._iteration_count,
            player_losses=player_losses_dict,
            current_strategies=final_strategies
        )
        return status


    def get_current_strategies(self) -> Dict[str, Strategy]:
        """
        Gets the current probability distributions for all players, detached from the computation graph.

        Returns:
            A dictionary mapping player names to Strategy objects containing NumPy arrays.
        """
        strategies = {}
        with torch.no_grad(): # Ensure no gradients are computed here
            for player in self.game.players:
                # Project current parameters to simplex
                probabilities_tensor = project_to_simplex(player.strategy_params)
                # Detach and convert to NumPy array for the Strategy object [cite: 83]
                probabilities_np = probabilities_tensor.detach().cpu().numpy()
                strategies[player.name] = Strategy(
                    player_name=player.name,
                    probabilities=probabilities_np
                )
        return strategies


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

    # Initialize the finder
    finder = MSNEFinder(game=game, learning_rate=0.05, optimizer_cls=optim.Adam)

    print("Initial Strategies:")
    initial_strats = finder.get_current_strategies()
    for name, strat in initial_strats.items():
        print(f"  {name}: {strat.probabilities.round(3)}")

    # Run a few optimization steps
    print("\nRunning optimization steps...")
    num_steps = 200
    for i in range(num_steps):
        status = finder.step()
        if (i + 1) % 50 == 0:
             print(f"--- Iteration {status.iteration} ---")
             print(f"  Losses: P1={status.player_losses['P1']:.4f}, P2={status.player_losses['P2']:.4f}")
             current_strats = status.current_strategies
             print(f"  Strategies: P1={current_strats['P1'].probabilities.round(3)}, "
                   f"P2={current_strats['P2'].probabilities.round(3)}")

    print("\nFinal Strategies (after {} steps):".format(num_steps))
    final_strats = finder.get_current_strategies()
    for name, strat in final_strats.items():
        # For Matching Pennies, expect strategies close to [0.5, 0.5]
        print(f"  {name}: {strat.probabilities.round(3)}")