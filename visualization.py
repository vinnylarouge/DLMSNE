# module: visualization

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
from typing import List, Dict, Optional, Sequence, Tuple

# Assuming data_structures module is available
# from data_structures import TrainingStatus, Strategy

# If running independently, include necessary definitions or placeholders:
try:
    from data_structures import TrainingStatus, Strategy
except ImportError:
    from typing import NamedTuple
    class Strategy(NamedTuple):
        player_name: str
        probabilities: np.ndarray
    class TrainingStatus(NamedTuple):
        iteration: int
        player_losses: Dict[str, float]
        current_strategies: Dict[str, Strategy]
        global_loss: Optional[float] = None


# Helper function for ternary plot transformation (for 3 actions)
def _to_ternary_coords(points: np.ndarray) -> np.ndarray:
    """Converts 3D simplex points (summing to 1) to 2D coordinates for plotting."""
    # Ensure points sum to 1 (or normalize)
    with np.errstate(divide='ignore', invalid='ignore'): # Ignore potential divide by zero if sum is zero
        points_sum = points.sum(axis=1, keepdims=True)
        safe_sum = np.where(points_sum == 0, 1, points_sum) # Avoid division by zero
        normalized_points = points / safe_sum
    # Transformation to equilateral triangle coordinates
    # See: https://en.wikipedia.org/wiki/Ternary_plot#Plotting_a_ternary_diagram
    x = 0.5 * (2 * normalized_points[:, 1] + normalized_points[:, 2])
    y = (np.sqrt(3) / 2) * normalized_points[:, 2]
    return np.stack([x, y], axis=-1)

# Helper function to setup ternary axes
def _setup_ternary_axis(ax: Axes, labels: Sequence[str] = ('A1', 'A2', 'A3')):
    """Sets up the visual boundaries and labels for a ternary plot."""
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-0.05, 1.05) # Give a bit of padding
    ax.set_ylim(-0.05, np.sqrt(3) / 2 + 0.05)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off') # Remove box frame

    # Vertices
    corners = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
    # Draw triangle border
    triangle = plt.Polygon(corners, edgecolor='k', facecolor='none', linewidth=1)
    ax.add_patch(triangle)

    # Labels at corners (adjusted positions)
    ax.text(corners[0, 0] - 0.03, corners[0, 1] - 0.02, labels[0], ha='right', va='top', fontsize=9)
    ax.text(corners[1, 0] + 0.03, corners[1, 1] - 0.02, labels[1], ha='left', va='top', fontsize=9)
    ax.text(corners[2, 0], corners[2, 1] + 0.03, labels[2], ha='center', va='bottom', fontsize=9)
    ax.set_title("Strategy Simplex (3 Actions)")


class PlotGenerator:
    """
    Handles the generation and updating of plots for strategy visualization and loss tracking.
    Uses Matplotlib.
    """

    def __init__(self):
        # Use a more vibrant color palette (e.g., tab10)
        # self.colors = plt.cm.viridis(np.linspace(0, 1, 10)) # Old palette
        self.colors = plt.cm.tab10(np.linspace(0, 1, 10)) # Use tab10 for up to 10 distinct colors
        # If more than 10 players are needed, consider tab20 or custom list
        self.markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', '+'] # Cycle through markers

    def create_loss_plot(self) -> Tuple[Figure, Axes]:
        """Creates the initial figure and axes for the loss plot."""
        fig, ax = plt.subplots()
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss (Negative Expected Payoff)")
        ax.set_title("Player Losses vs. Iteration")
        ax.grid(True, linestyle='--', alpha=0.6)
        # Adjust layout slightly to prevent labels cutting off later
        fig.tight_layout(pad=1.5)
        return fig, ax

    def update_loss_plot(self, ax: Axes, loss_history: List[TrainingStatus], player_names: List[str]):
        """
        Updates the loss plot axes with data from the loss_history.

        Args:
            ax: The Matplotlib Axes object for the loss plot.
            loss_history: A list of TrainingStatus objects over iterations.
            player_names: Ordered list of player names.
        """
        ax.clear() # Clear previous lines
        iterations = [status.iteration for status in loss_history]

        if not iterations: # No data yet
             ax.set_title("Player Losses vs. Iteration (Waiting for data...)")
             ax.grid(True, linestyle='--', alpha=0.6)
             ax.figure.canvas.draw_idle() # Redraw empty state
             return

        plotted_lines = [] # For legend
        for i, name in enumerate(player_names):
            player_losses = [status.player_losses.get(name, np.nan) for status in loss_history]
            color_idx = i % len(self.colors)
            marker_idx = i % len(self.markers)
            line, = ax.plot(iterations, player_losses, label=f"{name} Loss", # Use label here
                            color=self.colors[color_idx], marker=self.markers[marker_idx], markersize=3, linestyle='-')
            plotted_lines.append(line)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss (Negative Expected Payoff)")
        ax.set_title("Player Losses vs. Iteration")
        # Position legend outside plot area using handles
        if plotted_lines:
            ax.legend(handles=plotted_lines, fontsize='small', bbox_to_anchor=(1.04, 1), loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.relim() # Recalculate limits
        ax.autoscale_view() # Autoscale
        # Ensure layout accommodates the legend
        ax.figure.tight_layout(pad=1.5)
        ax.figure.canvas.draw_idle()

    def create_simplex_plot(self, num_actions: int, action_names: Optional[List[str]] = None,
                             action_indices_to_plot: Optional[Sequence[int]] = None) -> Tuple[Optional[Figure], Optional[Axes]]:
        """
        Creates the initial figure and axes for a simplex plot.
        Handles 2, 3, 4, 5 actions, or selected actions if > 5.

        Args:
            num_actions: The total number of actions for the players being plotted.
            action_names: Optional list of action names for labeling axes.
            action_indices_to_plot: Optional list of indices (0-based) of actions to plot
                                     if num_actions > 5. Must have 2 <= len <= 5.

        Returns:
            A tuple (Figure, Axes) or (None, None) if visualization is not supported.
        """
        # Create figure first
        fig = plt.figure()

        effective_num_actions = num_actions
        effective_action_names = action_names

        if action_indices_to_plot:
             if not (2 <= len(action_indices_to_plot) <= 5):
                 plt.close(fig) # Close the unused figure
                 return None, None
                 # raise ValueError("action_indices_to_plot must contain between 2 and 5 indices.")
             effective_num_actions = len(action_indices_to_plot)
             if action_names:
                 effective_action_names = [action_names[i] for i in action_indices_to_plot]

        ax = fig.add_subplot(111) # Add axes to the figure

        if effective_num_actions == 2:
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 0.1) # Essentially a line
            ax.set_xlabel(f"Probability({effective_action_names[0] if effective_action_names else 'A1'})")
            ax.set_yticks([])
            ax.set_title("Strategy Simplex (2 Actions)")
            ax.grid(True, axis='x', linestyle='--', alpha=0.6)
        elif effective_num_actions == 3:
             # Use helper to setup ternary axes
             labels = effective_action_names if effective_action_names and len(effective_action_names)==3 else ('A1','A2','A3')
             _setup_ternary_axis(ax, labels)
             # Title set within helper
        elif effective_num_actions == 4 or effective_num_actions == 5:
            # Simple 2D projection: Plot Prob(Action 1) vs Prob(Action 2)
            idx1, idx2 = (0, 1) # Plot first two selected actions by default
            x_label = f"Prob({effective_action_names[idx1] if effective_action_names else f'Action {idx1+1}'})"
            y_label = f"Prob({effective_action_names[idx2] if effective_action_names else f'Action {idx2+1}'})"
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(f"Strategy Simplex ({effective_num_actions} Actions - Projection)")
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.1)
            ax.grid(True, linestyle='--', alpha=0.6)
        else: # num_actions < 2 or (num_actions > 5 and not action_indices_to_plot)
             plt.close(fig) # Close the unused figure
             return None, None # Indicate no plot generated

        # Adjust layout to potentially make space for legend later
        fig.tight_layout(rect=[0, 0, 0.85, 1], pad=1.5) # Leave space on the right (rect=[left, bottom, right, top])

        return fig, ax

    def update_simplex_plot(self, ax: Axes, history: Dict[str, List[np.ndarray]], player_names: List[str],
                             action_indices_to_plot: Optional[Sequence[int]] = None):
        """
        Updates the simplex plot axes with strategy paths from history.
        Handles different dimensionalities (2, 3, 4, 5 actions) or selected subsets.
        Places the legend outside the plot area.
        """
        if ax is None: # Plotting might be disabled for this dimension count
             return

        ax.clear() # Clear previous paths

        if not history or not any(history.values()): # No data
            ax.set_title("Strategy Simplex (Waiting for data...)")
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.figure.canvas.draw_idle() # Redraw empty state
            return

        # Determine dimensions from the first player's history
        first_player_name = next(iter(history)) # Assume at least one player if history is not empty
        # Find first player with actual data
        while not history[first_player_name] and player_names:
            try:
                 first_player_name = player_names[player_names.index(first_player_name) + 1]
            except (ValueError, IndexError): # Reached end or name not found
                 ax.set_title("Strategy Simplex (No data...)")
                 ax.grid(True, linestyle='--', alpha=0.6)
                 ax.figure.canvas.draw_idle()
                 return

        if not history[first_player_name]: # Still no data after checking all
             ax.set_title("Strategy Simplex (No data...)")
             ax.grid(True, linestyle='--', alpha=0.6)
             ax.figure.canvas.draw_idle()
             return

        num_actions_total = history[first_player_name][0].shape[0]
        effective_num_actions = num_actions_total
        indices = list(range(num_actions_total))
        # Assume action names exist matching num_actions_total length (fetch from somewhere if needed)
        action_names_total = [f"A{i+1}" for i in range(num_actions_total)] # Placeholder names

        if action_indices_to_plot:
             if not (2 <= len(action_indices_to_plot) <= 5):
                 print("Warning: action_indices_to_plot invalid in update_simplex_plot.")
                 return
             effective_num_actions = len(action_indices_to_plot)
             indices = action_indices_to_plot
             effective_action_names = [action_names_total[i] for i in indices]
        elif num_actions_total > 5:
             ax.clear()
             ax.text(0.5, 0.5, f"{num_actions_total} actions.\nPlease select 2-5 to visualize.",
                     ha='center', va='center', transform=ax.transAxes)
             ax.set_xticks([])
             ax.set_yticks([])
             ax.figure.canvas.draw_idle()
             return
        else: # 2 <= num_actions_total <= 5
            effective_action_names = action_names_total


        # Resetup axes based on effective dimensions
        if effective_num_actions == 2:
            ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.1, 0.1); ax.set_yticks([])
            ax.set_title("Strategy Simplex (2 Actions)")
            ax.grid(True, axis='x', linestyle='--', alpha=0.6)
            ax.set_xlabel(f"Prob({effective_action_names[0]})") # Update label
        elif effective_num_actions == 3:
             _setup_ternary_axis(ax, effective_action_names)
        elif effective_num_actions == 4 or effective_num_actions == 5:
            ax.set_title(f"Strategy Simplex ({effective_num_actions} Actions - Projection)")
            ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.1, 1.1)
            ax.grid(True, linestyle='--', alpha=0.6)
            idx1, idx2 = 0, 1
            ax.set_xlabel(f"Prob({effective_action_names[idx1]})") # Update labels
            ax.set_ylabel(f"Prob({effective_action_names[idx2]})")
        else: # Should not be reached
            return

        plotted_lines = [] # To store handles for the legend
        for i, name in enumerate(player_names):
            if name not in history or not history[name]:
                continue

            player_history = np.array(history[name]) # Shape (iterations, num_actions_total)
            player_history_selected = player_history[:, indices] # Shape (iterations, effective_num_actions)

            row_sums = player_history_selected.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            player_history_normalized = player_history_selected / row_sums

            color_idx = i % len(self.colors)
            marker_idx = i % len(self.markers)

            x_coords, y_coords = None, None

            if effective_num_actions == 2:
                x_coords = player_history_normalized[:, 0]
                y_coords = np.zeros_like(x_coords)
            elif effective_num_actions == 3:
                coords_2d = _to_ternary_coords(player_history_normalized)
                x_coords = coords_2d[:, 0]
                y_coords = coords_2d[:, 1]
            elif effective_num_actions == 4 or effective_num_actions == 5:
                idx1, idx2 = 0, 1
                x_coords = player_history_normalized[:, idx1]
                y_coords = player_history_normalized[:, idx2]

            if x_coords is not None and y_coords is not None:
                line, = ax.plot(x_coords, y_coords, label=name, color=self.colors[color_idx],
                                marker=None, linestyle='-', linewidth=1.0, alpha=0.7)
                plotted_lines.append(line) # Store line handle for legend

                if len(x_coords) > 0:
                    # Plot start/end points without adding extra legend entries
                    ax.scatter(x_coords[0], y_coords[0], color=self.colors[color_idx],
                               marker=self.markers[marker_idx], s=50, edgecolor='black', zorder=5) # Start
                    ax.scatter(x_coords[-1], y_coords[-1], color=self.colors[color_idx],
                               marker=self.markers[marker_idx], s=100, edgecolor='black', zorder=5) # End


        if plotted_lines: # Add legend only if something was plotted
             # Position legend outside the main axes area
             ax.legend(handles=plotted_lines, fontsize='small',
                       bbox_to_anchor=(1.04, 1), # Position relative to axes (slightly outside, top)
                       loc='upper left',          # Anchor point of the legend box
                       borderaxespad=0.)          # Padding between axes and legend box


        # Redraw the canvas
        ax.figure.tight_layout(rect=[0, 0, 0.85, 1], pad=1.5) # Ensure space is maintained
        ax.figure.canvas.draw_idle()


# Example Usage (Optional - for demonstration)
if __name__ == '__main__':
    plot_gen = PlotGenerator()

    # --- Loss Plot Example ---
    players = ["P1", "P2"]
    history = [
        TrainingStatus(0, {"P1": 0.5, "P2": 0.6}, {"P1": Strategy("P1", np.array([0.5,0.5])), "P2": Strategy("P2", np.array([0.5,0.5]))}, 0.6),
        TrainingStatus(1, {"P1": 0.4, "P2": 0.45}, {"P1": Strategy("P1", np.array([0.6,0.4])), "P2": Strategy("P2", np.array([0.4,0.6]))}, 0.45),
        TrainingStatus(2, {"P1": 0.3, "P2": 0.33}, {"P1": Strategy("P1", np.array([0.7,0.3])), "P2": Strategy("P2", np.array([0.3,0.7]))}, 0.33),
    ]
    fig_loss, ax_loss = plot_gen.create_loss_plot()
    plot_gen.update_loss_plot(ax_loss, history, players)
    print("Loss plot generated (display not shown in text).")
    # plt.show() # Uncomment to display if running locally

    # --- Simplex Plot Examples ---

    # 1. Two Actions
    sim_history_2 = {
        "P1": [np.array([0.5, 0.5]), np.array([0.6, 0.4]), np.array([0.7, 0.3])],
        "P2": [np.array([0.5, 0.5]), np.array([0.4, 0.6]), np.array([0.3, 0.7])]
    }
    fig_sim2, ax_sim2 = plot_gen.create_simplex_plot(num_actions=2, action_names=["H", "T"])
    if ax_sim2:
        plot_gen.update_simplex_plot(ax_sim2, sim_history_2, players)
        print("2-Action Simplex plot generated.")
        # fig_sim2.show()

    # 2. Three Actions (Ternary)
    sim_history_3 = {
        "P1": [np.array([0.33, 0.33, 0.34]), np.array([0.5, 0.3, 0.2]), np.array([0.7, 0.2, 0.1])],
        "P2": [np.array([0.33, 0.33, 0.34]), np.array([0.2, 0.5, 0.3]), np.array([0.1, 0.7, 0.2])]
    }
    fig_sim3, ax_sim3 = plot_gen.create_simplex_plot(num_actions=3, action_names=["R", "P", "S"])
    if ax_sim3:
        plot_gen.update_simplex_plot(ax_sim3, sim_history_3, players)
        print("3-Action Simplex plot generated.")
        # fig_sim3.show()


    # 3. Five Actions (Projection)
    sim_history_5 = {
        "P1": [np.array([0.2]*5), np.array([0.4, 0.3, 0.1, 0.1, 0.1]), np.array([0.6, 0.2, 0.05, 0.05, 0.1])],
        "P2": [np.array([0.2]*5), np.array([0.1, 0.1, 0.4, 0.3, 0.1]), np.array([0.05, 0.05, 0.6, 0.2, 0.1])]
    }
    fig_sim5, ax_sim5 = plot_gen.create_simplex_plot(num_actions=5, action_names=["A1", "A2", "A3", "A4", "A5"])
    if ax_sim5:
        plot_gen.update_simplex_plot(ax_sim5, sim_history_5, players)
        print("5-Action Simplex plot generated (projection).")
        # fig_sim5.show()


    # 4. Six Actions (No plot by default)
    fig_sim6, ax_sim6 = plot_gen.create_simplex_plot(num_actions=6)
    if ax_sim6 is None:
        print("6-Action Simplex plot correctly not generated by default.")
    else:
        print("Error: 6-Action plot was generated unexpectedly.")

    # 5. Six Actions (Plotting selected 3)
    indices_to_plot = [0, 2, 4] # Select 1st, 3rd, 5th actions
    sim_history_6 = {
        "P1": [np.random.rand(6) / 3, np.random.rand(6)/3, np.random.rand(6)/3], # Dummy data
        "P2": [np.random.rand(6) / 3, np.random.rand(6)/3, np.random.rand(6)/3]
    }
    # Normalize dummy data to sum to 1 for plotting
    for p in sim_history_6: sim_history_6[p] = [v/v.sum() for v in sim_history_6[p]]

    fig_sim6_sel, ax_sim6_sel = plot_gen.create_simplex_plot(num_actions=6, action_indices_to_plot=indices_to_plot)
    if ax_sim6_sel:
        # We know it will be 3D effective, so setup ternary axis (already done in create)
        plot_gen.update_simplex_plot(ax_sim6_sel, sim_history_6, players, action_indices_to_plot=indices_to_plot)
        print(f"6-Action Simplex plot generated for selected actions {indices_to_plot}.")
        # fig_sim6_sel.show() #

    # plt.show() # Show all plots if uncommented