# module: gui

import sys
import time
import numpy as np
import torch # Needed for optimizer definition if defaults are used
from typing import List, Optional, Tuple
import random # Add import for random numbers
from itertools import product # Make sure product is imported at the top level if not already

# PyQt6 imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox, QGridLayout, QTextEdit,
    QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView, QDialog,
    QCheckBox, QScrollArea, QGroupBox
)
from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot, Qt
from PyQt6.QtGui import QDoubleValidator

# Matplotlib embedding imports
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt # Usually implicitly imported by canvas

# --- Assuming previous modules are available ---
# Define a placeholder optimizer first
class _BaseOptimizerPlaceholder:
    """Minimal placeholder for optimizer if torch isn't available."""
    def __init__(self, params, lr=0.01): # Add expected signature
        print(f"Warning: Using placeholder optimizer. Learning rate: {lr}, Params: {len(list(params)) if hasattr(params, '__iter__') else 'N/A'}")
        pass
    def step(self): pass
    def zero_grad(self): pass

DefaultOptimizer = _BaseOptimizerPlaceholder # Assign the placeholder initially

try:
    from data_structures import Game, Player, Strategy, TrainingStatus
    from game_theory_core import project_to_simplex, calculate_expected_payoffs
    from optimizer import MSNEFinder
    from visualization import PlotGenerator
except ImportError as e:
    print(f"Error importing backend modules: {e}")
    print("Please ensure data_structures.py, game_theory_core.py, optimizer.py, "
          "and visualization.py are in the Python path.")
    # Minimal placeholders to allow GUI structure compilation
    class Game: pass
    class Player: pass
    class Strategy: pass
    class TrainingStatus: pass
    class MSNEFinder: pass
    class PlotGenerator:
         def create_loss_plot(self): return plt.figure(), plt.axes()
         def update_loss_plot(self, ax, history, names): pass
         def create_simplex_plot(self, num_actions, action_names=None, action_indices_to_plot=None):
             if num_actions < 2 or (num_actions > 5 and not action_indices_to_plot): return None, None
             return plt.figure(), plt.axes()
         def update_simplex_plot(self, ax, history, names, action_indices_to_plot=None): pass

# Try to import the real optimizer and overwrite the placeholder
try:
    from torch.optim import Adam as TorchAdamOptimizer
    DefaultOptimizer = TorchAdamOptimizer # Overwrite placeholder if import succeeds
    print("Successfully imported torch.optim.Adam as DefaultOptimizer.")
except ImportError:
    print("torch.optim.Adam not found. Using placeholder DefaultOptimizer.")
except Exception as e: # Catch other potential import errors
    print(f"An unexpected error occurred during torch import: {e}. Using placeholder DefaultOptimizer.")


# == Worker Thread for Optimization ==
class TrainingWorker(QObject):
    """
    Runs the MSNE optimization loop in a separate thread.
    Emits signals to update the GUI without blocking it.
    """
    # Signal payload: TrainingStatus object
    status_updated = pyqtSignal(TrainingStatus)
    # Signal payload: str (error message)
    error_occurred = pyqtSignal(str)
    # Signal payload: None (indicates completion or stop)
    finished = pyqtSignal()

    def __init__(self, msne_finder: MSNEFinder, max_iterations: int = 10000, update_interval_ms: int = 50):
        super().__init__()
        self.msne_finder = msne_finder
        # max_iterations now represents the target iteration count for this run
        self.max_iterations = max_iterations
        self.update_interval_ms = update_interval_ms / 1000.0 # Convert ms to seconds for time.sleep
        self._running = False
        self._paused = False

    @pyqtSlot()
    def run(self):
        """Starts the optimization loop."""
        self._running = True
        self._paused = False
        # Use the finder's internal count to check progress against the target
        iteration = self.msne_finder._iteration_count # Start from finder's current state
        start_iter = iteration
        try:
            while self._running and iteration < self.max_iterations:
                if self._paused:
                    time.sleep(0.1) # Sleep briefly when paused
                    continue

                status = self.msne_finder.step() # This increments the finder's internal count
                self.status_updated.emit(status) # Send status back to main thread
                iteration = status.iteration # Use iteration count from status

                # Control update frequency / CPU usage
                if self.update_interval_ms > 0:
                    time.sleep(self.update_interval_ms)

            if iteration >= self.max_iterations:
                 print(f"Reached target iteration: {self.max_iterations}.")
            elif not self._running:
                 print(f"Training stopped externally at iteration {iteration}.")

        except Exception as e:
            # Propagate errors back to the main thread
            import traceback
            err_msg = f"Error in optimization thread: {e}\n{traceback.format_exc()}"
            self.error_occurred.emit(err_msg)
        finally:
            self._running = False
            self.finished.emit() # Signal that the thread's work is done

    def stop(self):
        """Requests the loop to stop."""
        self._running = False

    def pause(self):
        """Requests the loop to pause."""
        self._paused = True

    def resume(self):
        """Requests the loop to resume."""
        self._paused = False

    def is_running(self):
        return self._running


# == Action Selection Dialog ==
class ActionSelectionDialog(QDialog):
    """Dialog to select which actions to visualize when num_actions > 5."""
    def __init__(self, action_names: List[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Actions for Simplex Plot")
        self.action_names = action_names
        self.checkboxes: List[QCheckBox] = []
        self.selected_indices: List[int] = []

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select 2 to 5 actions to visualize:"))

        scroll_area = QScrollArea(self)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        for i, name in enumerate(action_names):
            cb = QCheckBox(f"{name} (Index {i})")
            cb.stateChanged.connect(self._validate_selection)
            self.checkboxes.append(cb)
            scroll_layout.addWidget(cb)

        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)

        # Buttons
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self._accept_selection)
        self.ok_button.setEnabled(False) # Disabled until valid selection
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        self.setMinimumWidth(300)

    def _validate_selection(self):
        """Enable OK button only if 2-5 actions are selected."""
        count = sum(1 for cb in self.checkboxes if cb.isChecked())
        self.ok_button.setEnabled(2 <= count <= 5)

    def _accept_selection(self):
        """Store selected indices and accept the dialog."""
        self.selected_indices = [i for i, cb in enumerate(self.checkboxes) if cb.isChecked()]
        self.accept()

    def get_selected_indices(self) -> List[int]:
        """Return the indices selected by the user."""
        return self.selected_indices


# == Main Application Window ==
class MainWindow(QMainWindow):
    """Main application window for setting up, running, and visualizing MSNE finding."""

    # Define default names and actions
    DEFAULT_PLAYER_NAMES = ["Alice", "Bob", "Charlie", "Dennis", "Eve", "Frank", "Grace", "Heidi", "Ivan", "Judy"]
    DEFAULT_ACTION_VERBS = ["Attack", "Block", "Concede", "Defend", "Evade", "Feint", "Grab", "Hold", "Inspect", "Jump"] # Extended list
    SPINNER_CHARS = ['|', '/', '-', '\\'] # Characters for activity indicator
    CONTINUE_ITERATIONS = 1000 # Number of iterations to add when continuing

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MSNE Finder using Gradient Ascent")
        self.setGeometry(100, 100, 1200, 800) # x, y, width, height

        # --- Internal State ---
        self.game: Optional[Game] = None
        self.msne_finder: Optional[MSNEFinder] = None
        self.plot_generator: PlotGenerator = PlotGenerator() # Create instance
        self.strategy_history: Dict[str, List[np.ndarray]] = {}
        self.loss_history: List[TrainingStatus] = []
        self.training_thread: Optional[QThread] = None
        self.worker: Optional[TrainingWorker] = None
        self.is_training_running = False
        self.is_training_paused = False
        self.player_widgets: List[Tuple[QLineEdit, QLineEdit]] = [] # (name_edit, actions_edit)
        self.payoff_table: Optional[QTableWidget] = None
        self.payoff_scroll_area: Optional[QScrollArea] = None # Add scroll area attribute
        self.action_indices_to_plot: Optional[List[int]] = None # For >5 actions
        self.current_player_info: Optional[List[Tuple[str, List[str]]]] = None # Store player names and actions
        self.current_action_counts: Optional[List[int]] = None # Store action counts
        self.log_interval = 100 # Log full details every N iterations
        self.total_iterations_run = 0 # Track total iterations completed across runs

        # --- Backend Settings ---
        self.learning_rate = 0.01
        self.optimizer_class = DefaultOptimizer # Now DefaultOptimizer is guaranteed to be defined
        self.initial_max_iterations = 500 # Max iterations for the *first* run
        self.update_interval_ms = 50 # Update plot roughly 20 times/sec

        # --- UI Elements ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget) # Main layout: Setup | Plots

        # -- Left Panel: Setup & Control --
        setup_panel = QWidget()
        self.setup_layout = QVBoxLayout(setup_panel) # Make setup_layout accessible
        setup_panel.setMaximumWidth(450) # Limit width of setup panel

        # GroupBox for Game Definition
        self.game_definition_groupbox = QGroupBox("Game Definition")
        definition_layout = QGridLayout(self.game_definition_groupbox) # Layout for the groupbox

        definition_layout.addWidget(QLabel("Number of Players:"), 0, 0)
        self.num_players_spinbox = QSpinBox()
        self.num_players_spinbox.setRange(2, 10) # Min 2 players, Max 10
        self.num_players_spinbox.valueChanged.connect(self._update_player_fields)
        definition_layout.addWidget(self.num_players_spinbox, 0, 1)

        definition_layout.addWidget(QLabel("Number of Actions:"), 1, 0)
        self.num_actions_spinbox = QSpinBox()
        self.num_actions_spinbox.setRange(2, len(self.DEFAULT_ACTION_VERBS)) # Min 2 actions, Max based on defaults
        self.num_actions_spinbox.setValue(2) # Default to 2 actions
        self.num_actions_spinbox.valueChanged.connect(self._update_action_fields) # Connect signal
        definition_layout.addWidget(self.num_actions_spinbox, 1, 1)

        # Dynamic Player Name/Action Fields within the GroupBox
        self.player_fields_layout = QVBoxLayout()
        definition_layout.addLayout(self.player_fields_layout, 2, 0, 1, 2) # Span 2 columns

        self.setup_layout.addWidget(self.game_definition_groupbox) # Add groupbox to main setup layout

        # Payoff Table Placeholder (will hold the scroll area)
        self.payoff_group_layout = QVBoxLayout()
        self.setup_layout.addLayout(self.payoff_group_layout)

        # Buttons
        button_layout = QGridLayout() # Use a grid for buttons too
        self.setup_game_button = QPushButton("1. Setup Game Structure")
        self.setup_game_button.clicked.connect(self.setup_game_structure)
        button_layout.addWidget(self.setup_game_button, 0, 0, 1, 2)

        self.initialize_button = QPushButton("2. Initialize Training")
        self.initialize_button.clicked.connect(self.initialize_training)
        self.initialize_button.setEnabled(False) # Disabled initially
        button_layout.addWidget(self.initialize_button, 1, 0, 1, 2)

        # Training Controls
        control_layout = QHBoxLayout()
        self.start_button = QPushButton("▶ Start") # Initial text
        self.start_button.clicked.connect(self.start_training)
        self.start_button.setEnabled(False)
        control_layout.addWidget(self.start_button)

        self.pause_button = QPushButton("❚❚ Pause")
        self.pause_button.clicked.connect(self.pause_resume_training)
        self.pause_button.setEnabled(False)
        control_layout.addWidget(self.pause_button)

        self.reset_button = QPushButton("■ Reset")
        self.reset_button.clicked.connect(self.reset_training)
        self.reset_button.setEnabled(False)
        control_layout.addWidget(self.reset_button)
        button_layout.addLayout(control_layout, 2, 0, 1, 2)

        # Action Selection Button (for >5 actions)
        self.select_actions_button = QPushButton("Select Actions for Plot")
        self.select_actions_button.clicked.connect(self._show_action_selection_dialog)
        self.select_actions_button.setVisible(False) # Hidden initially
        button_layout.addWidget(self.select_actions_button, 3, 0, 1, 2)

        self.setup_layout.addLayout(button_layout) # Add button grid layout

        # Console Output
        self.setup_layout.addWidget(QLabel("Training Log (Updates every {} iterations):".format(self.log_interval)))
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setFixedHeight(150) # Fixed height for console
        self.setup_layout.addWidget(self.console_output)

        # Status Label for live iteration/spinner
        self.status_label = QLabel("Status: Idle")
        self.status_label.setStyleSheet("QLabel { color : gray; }") # Style it subtly
        self.setup_layout.addWidget(self.status_label)

        self.setup_layout.addStretch() # Push controls towards top

        main_layout.addWidget(setup_panel)

        # -- Right Panel: Plots --
        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)

        # Simplex Plot
        self.simplex_fig, self.simplex_ax = self.plot_generator.create_simplex_plot(num_actions=2) # Default 2 actions
        if self.simplex_fig: # Only add canvas/toolbar if fig exists
             self.simplex_canvas = FigureCanvas(self.simplex_fig)
             self.simplex_toolbar = NavigationToolbar(self.simplex_canvas, self)
             plot_layout.addWidget(self.simplex_toolbar)
             plot_layout.addWidget(self.simplex_canvas)
        else:
             self.simplex_canvas, self.simplex_toolbar, self.simplex_ax = None, None, None

        # Loss Plot
        self.loss_fig, self.loss_ax = self.plot_generator.create_loss_plot()
        self.loss_canvas = FigureCanvas(self.loss_fig)
        self.loss_toolbar = NavigationToolbar(self.loss_canvas, self)
        plot_layout.addWidget(self.loss_toolbar)
        plot_layout.addWidget(self.loss_canvas)

        main_layout.addWidget(plot_panel)

        # Initial UI state
        self._update_player_fields() # Create fields for initial player count

    # --- GUI Update Slots ---

    def _generate_default_action_string(self, num_actions: int) -> str:
        """Generates a comma-separated string of default actions."""
        if num_actions <= len(self.DEFAULT_ACTION_VERBS):
            return ", ".join(self.DEFAULT_ACTION_VERBS[:num_actions])
        else:
            # Fallback if more actions requested than verbs defined
            verbs = self.DEFAULT_ACTION_VERBS[:]
            verbs.extend([f"Action_{i+1}" for i in range(num_actions - len(verbs))])
            return ", ".join(verbs)

    @pyqtSlot(int)
    def _update_player_fields(self, num_players=None):
        """Dynamically create input fields for player names and actions using defaults."""
        if num_players is None:
            num_players = self.num_players_spinbox.value()

        # Clear existing widgets safely
        while self.player_fields_layout.count():
            item = self.player_fields_layout.takeAt(0)
            # Handle nested layouts (QHBoxLayout for name/action)
            layout_item = item.layout()
            if layout_item:
                self._clear_layout(layout_item)
            # Handle widgets directly within player_fields_layout (if any)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        self.player_widgets = []
        # Get current number of actions from the spinbox
        num_actions = self.num_actions_spinbox.value()
        default_actions_str = self._generate_default_action_string(num_actions)

        for i in range(num_players):
            hbox = QHBoxLayout()
            default_name = self.DEFAULT_PLAYER_NAMES[i] if i < len(self.DEFAULT_PLAYER_NAMES) else f"Player {i+1}"
            name_edit = QLineEdit(default_name)
            actions_edit = QLineEdit(default_actions_str)
            actions_edit.setToolTip("Enter comma-separated action names")

            hbox.addWidget(QLabel(f"P{i+1} Name:"))
            hbox.addWidget(name_edit)
            hbox.addWidget(QLabel(" Actions:"))
            hbox.addWidget(actions_edit)
            self.player_fields_layout.addLayout(hbox)
            self.player_widgets.append((name_edit, actions_edit))

        # Ensure game definition section is visible and clear payoff table
        self.game_definition_groupbox.setVisible(True)
        self._clear_payoff_display()
        self.initialize_button.setEnabled(False) # Need to setup structure first


    @pyqtSlot(int)
    def _update_action_fields(self, num_actions: int):
        """Update the default action string in all existing player action fields."""
        default_actions_str = self._generate_default_action_string(num_actions)
        for _, actions_edit in self.player_widgets:
            actions_edit.setText(default_actions_str)

        # Ensure game definition section is visible and clear payoff table
        self.game_definition_groupbox.setVisible(True)
        self._clear_payoff_display()
        self.initialize_button.setEnabled(False) # Need to setup structure first


    def _clear_layout(self, layout):
        """Removes all widgets and sub-layouts from a layout."""
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    nested_layout = item.layout()
                    if nested_layout is not None:
                        # Recursively clear nested layouts AND remove the layout item itself
                        self._clear_layout(nested_layout)
                        # layout.removeItem(item) # Optional: remove layout item too

    def _clear_payoff_display(self):
        """Clears the payoff table and scroll area."""
        if self.payoff_scroll_area:
            self.payoff_group_layout.removeWidget(self.payoff_scroll_area)
            self.payoff_scroll_area.deleteLater()
            self.payoff_scroll_area = None
        self.payoff_table = None # Table widget is owned by scroll area, deleted with it

    @pyqtSlot()
    def setup_game_structure(self):
        """Reads player/action info, creates the payoff table within a scroll area, and hides setup."""
        try:
            self._clear_payoff_display() # Clear previous table/scroll area
            self.action_indices_to_plot = None # Reset action selection

            player_info = []
            action_counts = []
            num_actions_from_spinbox = self.num_actions_spinbox.value()

            for i, (name_edit, actions_edit) in enumerate(self.player_widgets):
                name = name_edit.text().strip()
                actions_str = actions_edit.text().strip()
                if not name or not actions_str:
                    raise ValueError(f"Player {i+1} name and actions cannot be empty.")
                actions = [a.strip() for a in actions_str.split(',') if a.strip()]
                if not actions:
                    raise ValueError(f"Player {i+1} must have at least one action.")
                if len(actions) != num_actions_from_spinbox:
                    raise ValueError(f"Player {i+1} has {len(actions)} actions, but {num_actions_from_spinbox} are expected. Adjust actions or selector.")
                if len(actions) != len(set(actions)):
                    raise ValueError(f"Player {i+1} has duplicate action names.")
                player_info.append((name, actions))
                action_counts.append(len(actions))

            if len(set(action_counts)) > 1: raise ValueError("All players must have the same number of actions.")
            if len(set(p[0] for p in player_info)) != len(player_info): raise ValueError("Player names must be unique.")

            self.current_player_info = player_info
            self.current_action_counts = action_counts

            # Create Payoff Table
            num_players = len(action_counts)
            total_combinations = int(np.prod(action_counts))
            self.payoff_table = QTableWidget(total_combinations, num_players)
            self.payoff_table.setToolTip("Enter payoff for each player (columns) for each action profile (rows). Payoffs initialized randomly.")

            # Set column headers (Player Names)
            self.payoff_table.setHorizontalHeaderLabels([p[0] for p in player_info])

            # Set row headers (Action Profiles) - Use simplified index notation
            # Pass action_counts instead of player_info
            row_labels = self._generate_action_profile_labels_short(action_counts)
            self.payoff_table.setVerticalHeaderLabels(row_labels)

            # Set validator for payoff entries (allow floats) and initialize with random ints
            validator = QDoubleValidator()
            for r in range(total_combinations):
                for c in range(num_players):
                    random_payoff = str(random.randint(-10, 10))
                    item = QTableWidgetItem(random_payoff)
                    self.payoff_table.setItem(r, c, item)
                    # TODO: Setting validator on item requires delegation later

            self.payoff_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            # Adjust vertical header width to fit the shorter labels
            self.payoff_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)


            # Create Scroll Area for the Table
            self.payoff_scroll_area = QScrollArea()
            self.payoff_scroll_area.setWidget(self.payoff_table)
            self.payoff_scroll_area.setWidgetResizable(True) # Important! Allows table to resize within scroll area
            self.payoff_scroll_area.setFixedHeight(400) # Set a fixed height for the scrollable area

            self.payoff_group_layout.addWidget(QLabel("Enter/Verify Payoffs:"))
            self.payoff_group_layout.addWidget(self.payoff_scroll_area) # Add scroll area instead of table

            # Hide the definition section, enable next step
            self.game_definition_groupbox.setVisible(False)
            self.initialize_button.setEnabled(True)
            self.start_button.setEnabled(False)
            self.pause_button.setEnabled(False)
            self.reset_button.setEnabled(False) # Should be enabled AFTER init typically
            self.log_message("Payoff table created and initialized. Game definition hidden.")

        except ValueError as e:
            QMessageBox.warning(self, "Setup Error", str(e))
            self.initialize_button.setEnabled(False)
            # Ensure definition is visible if setup failed
            self.game_definition_groupbox.setVisible(True)

    def _generate_action_profile_labels_short(self, action_counts: List[int]) -> List[str]:
        """
        Helper to create short row labels like '(0,0)', '(0,1)', '(1,0)', '(1,1)', ...
        representing the indices of actions chosen by each player in order.
        """
        # Create ranges for each player's action indices
        action_index_ranges = [range(count) for count in action_counts]

        # Get the Cartesian product of action indices
        action_index_profiles = list(product(*action_index_ranges))

        # Format each tuple as a string '(idx1, idx2, ...)'
        labels = [str(profile) for profile in action_index_profiles]
        return labels

    def _get_payoff_tensor_from_table(self, player_index):
        """Reads the payoff table and constructs the tensor for a specific player."""
        if not self.payoff_table: return None

        num_rows = self.payoff_table.rowCount()
        # num_players = self.payoff_table.columnCount() # Not needed here
        action_counts = self.current_action_counts # Use stored action counts
        if not action_counts:
             raise ValueError("Cannot get payoff tensor, action counts not determined yet.")

        payoffs_flat = []
        for r in range(num_rows):
            item = self.payoff_table.item(r, player_index)
            if item is None: raise ValueError(f"Missing payoff value at row {r}, player {player_index+1}")
            try:
                payoffs_flat.append(float(item.text()))
            except ValueError:
                raise ValueError(f"Invalid numeric value '{item.text()}' at row {r}, player {player_index+1}")

        payoff_tensor = np.array(payoffs_flat).reshape(action_counts)
        return payoff_tensor

    @pyqtSlot()
    def initialize_training(self):
        """Reads all inputs, creates backend objects, and prepares for first training run."""
        if not self.payoff_table:
            self.log_message("Error: Payoff table not set up.")
            return

        try:
            # Reset iteration count for a completely new initialization
            self.total_iterations_run = 0

            # --- (Create players, payoff tensors, game as before) ---
            players = []
            # Check if current_player_info is set
            if not self.current_player_info:
                 raise ValueError("Player info not available. Please set up game structure first.")
            for name, actions in self.current_player_info:
                 players.append(Player(name=name, actions=actions))

            payoff_tensors = {}
            for i, player in enumerate(players):
                 payoff_tensors[player.name] = self._get_payoff_tensor_from_table(player_index=i)

            self.game = Game(players=players, payoff_tensors=payoff_tensors)
            # --- (End object creation) ---


            # Create a *new* MSNEFinder instance for a fresh start
            self.msne_finder = MSNEFinder(
                game=self.game,
                learning_rate=self.learning_rate,
                optimizer_cls=self.optimizer_class
            )
            self.log_message(f"Created new MSNEFinder. Iteration count reset to {self.msne_finder._iteration_count}.")


            # Clear Histories and Reset Plots
            self.strategy_history = {p.name: [] for p in self.game.players}
            self.loss_history = []
            self._update_plots(force_simplex_recreate=True) # Force full plot reset

            # Handle Action Selection for >5 Actions
            # ... (action selection logic as before) ...
            num_actions_max = max(p.num_actions for p in self.game.players) if self.game.players else 0
            if num_actions_max > 5:
                 self.select_actions_button.setVisible(True)
                 self.log_message(f"Game has {num_actions_max} actions. Select actions to plot if desired.")
            else:
                 self.select_actions_button.setVisible(False)
                 self.action_indices_to_plot = None # Not needed


            # Update Button States for starting
            self.start_button.setEnabled(True)
            self.start_button.setText("▶ Start") # Ensure text is "Start"
            self.pause_button.setEnabled(False)
            self.reset_button.setEnabled(True) # Can reset after init
            self.initialize_button.setEnabled(False) # Done initializing
            self.setup_game_button.setEnabled(False) # Lock setup during training run
            self.num_players_spinbox.setEnabled(False)
            for name_edit, actions_edit in self.player_widgets:
                 name_edit.setEnabled(False)
                 actions_edit.setEnabled(False)
            if self.payoff_table: self.payoff_table.setEnabled(False)


            self.log_message("Training initialized successfully.")
            self.status_label.setText("Status: Ready to start")

        except (ValueError, TypeError, IndexError, AttributeError) as e:
             import traceback
             QMessageBox.critical(self, "Initialization Error", f"Failed to initialize: {e}\n{traceback.format_exc()}")
             self.start_button.setEnabled(False)
             self.reset_button.setEnabled(False)
             self.game_definition_groupbox.setVisible(False) # Keep definition hidden


    @pyqtSlot()
    def _show_action_selection_dialog(self):
        """Shows the dialog to select actions for plotting."""
        if not self.game or not self.game.players:
            return
        # Assuming same actions for all players for simplicity
        action_names = self.game.players[0].actions
        num_actions = len(action_names)

        if num_actions <= 5:
            self.log_message("Action selection only needed for games with more than 5 actions.")
            return

        dialog = ActionSelectionDialog(action_names, self)
        if dialog.exec(): # exec() shows the dialog modally
            self.action_indices_to_plot = dialog.get_selected_indices()
            self.log_message(f"Selected action indices for plotting: {self.action_indices_to_plot}")
            # Re-create/update the simplex plot with the selection
            self._update_plots(force_simplex_recreate=True)
        else:
            self.log_message("Action selection cancelled.")


    # --- Training Control Slots ---

    @pyqtSlot()
    def start_training(self):
        """Starts or continues the background training thread."""
        if self.is_training_running: # Prevent starting if already running
            self.log_message("Warning: Training is already running.")
            return

        if not self.msne_finder:
            self.log_message("Error: Training not initialized. Please Initialize first.")
            QMessageBox.warning(self, "Not Initialized", "Please initialize training before starting.")
            return

        # Determine target iterations
        current_iter = self.msne_finder._iteration_count
        is_continuing = current_iter > 0 # Check if we are continuing an existing run
        if is_continuing:
            target_iterations = current_iter + self.CONTINUE_ITERATIONS
            self.log_message(f"Continuing training from iteration {current_iter} up to {target_iterations}...")
        else:
            target_iterations = self.initial_max_iterations # First run
            self.log_message(f"Starting training up to iteration {target_iterations}...")


        self.worker = TrainingWorker(self.msne_finder, target_iterations, self.update_interval_ms)
        self.training_thread = QThread()
        self.worker.moveToThread(self.training_thread)

        # Connect signals from worker to slots in main thread
        self.worker.status_updated.connect(self.update_ui)
        self.worker.error_occurred.connect(self._handle_worker_error)
        self.worker.finished.connect(self._handle_worker_finished)
        self.training_thread.started.connect(self.worker.run) # Start worker's run method

        self.training_thread.start()
        self.is_training_running = True
        self.is_training_paused = False

        # Update UI state for running
        self.start_button.setEnabled(False) # Disable start/continue while running
        self.start_button.setText("▶ Start") # Reset text while running
        self.pause_button.setEnabled(True)
        self.pause_button.setText("❚❚ Pause")
        self.reset_button.setEnabled(True) # Allow reset while running
        self.status_label.setText("Status: Running...")


    @pyqtSlot()
    def pause_resume_training(self):
        """Toggles between pausing and resuming the training worker."""
        if not self.worker or not self.is_training_running:
            return

        if self.is_training_paused:
            self.worker.resume()
            self.is_training_paused = False
            self.pause_button.setText("❚❚ Pause")
            self.log_message("Training resumed.")
            self.status_label.setText("Status: Running...")
        else:
            self.worker.pause()
            self.is_training_paused = True
            self.pause_button.setText("▶ Resume")
            self.log_message("Training paused.")
            self.status_label.setText("Status: Paused")


    @pyqtSlot()
    def reset_training(self):
        """Stops training, clears state, re-shows game definition, and resets counts."""
        if self.worker:
            self.worker.stop()

        if self.training_thread and self.training_thread.isRunning():
             self.training_thread.quit()
             self.training_thread.wait(1000)
             if self.training_thread.isRunning():
                 print("Warning: Training thread termination needed.")
                 self.training_thread.terminate()
                 self.training_thread.wait()


        self.training_thread = None
        self.worker = None
        self.is_training_running = False
        self.is_training_paused = False

        # Reset core state for a true reset
        self.msne_finder = None # Clear the finder instance
        self.game = None
        self.total_iterations_run = 0 # Reset total count

        # Clear history and plots
        self.strategy_history = {}
        self.loss_history = []
        if self.loss_ax or self.simplex_ax: # Check if any plot exists
             self._update_plots(force_simplex_recreate=True)


        # Reset UI state
        self.start_button.setEnabled(False) # Disabled until next successful init
        self.start_button.setText("▶ Start") # Reset text
        self.pause_button.setEnabled(False)
        self.pause_button.setText("❚❚ Pause")
        self.reset_button.setEnabled(False) # Disabled until next successful init
        self.initialize_button.setEnabled(self.payoff_table is not None) # Re-enable if structure exists
        self.setup_game_button.setEnabled(True) # Allow re-running setup

        # Re-show the game definition section and enable its widgets
        self.game_definition_groupbox.setVisible(True)
        self.num_players_spinbox.setEnabled(True)
        self.num_actions_spinbox.setEnabled(True)
        for name_edit, actions_edit in self.player_widgets:
             name_edit.setEnabled(True)
             actions_edit.setEnabled(True)
        if self.payoff_table: self.payoff_table.setEnabled(True) # Re-enable edits


        # Clear status label
        self.status_label.setText("Status: Reset")
        self.log_message("Training reset. Ready for setup or initialization.")


    # --- Signal Handling Slots ---

    @pyqtSlot(TrainingStatus)
    def update_ui(self, status: TrainingStatus):
        """Receives status from worker and updates history, plots, console (sparsely), and status label."""
        if not self.game: return # Should not happen if training is running

        # 1. Update History (always update history for plots)
        self.loss_history.append(status)
        for name, strategy in status.current_strategies.items():
            if name in self.strategy_history:
                 self.strategy_history[name].append(strategy.probabilities)
            else:
                 self.strategy_history[name] = [strategy.probabilities]

        # 2. Update Console Sparsely and Status Label
        iteration = status.iteration
        if iteration == 1 or iteration % self.log_interval == 0:
            # Log full details to console
            loss_str = ", ".join([f"{name}: {loss:.4f}" for name, loss in status.player_losses.items()])
            strat_str = ", ".join([f"{name}: {np.round(s.probabilities, 3)}"
                                   for name, s in status.current_strategies.items()])
            log_line = f"Iter {iteration}: Losses=[{loss_str}] | Strats=[{strat_str}]"
            self.console_output.append(log_line)
            self.console_output.verticalScrollBar().setValue(self.console_output.verticalScrollBar().maximum()) # Auto-scroll
            # Update status label concisely
            self.status_label.setText(f"Status: Iteration {iteration} (Logged)")
        else:
            # Update status label with spinner
            spinner_char = self.SPINNER_CHARS[iteration % len(self.SPINNER_CHARS)]
            self.status_label.setText(f"Status: Iteration {iteration} {spinner_char}")

        # 3. Update Plots (can still update frequently)
        # Check if plots exist before updating
        if self.loss_ax:
             self._update_plots()


    def _update_plots(self, force_simplex_recreate=False):
         """Helper to redraw plots based on current history and settings."""
         # Check if game and plot objects exist before proceeding
         if not self.game or not self.loss_ax:
             # Attempt to clear if axes exist but game doesn't
             if self.simplex_ax and self.simplex_canvas: self.simplex_ax.clear(); self.simplex_canvas.draw()
             if self.loss_ax and self.loss_canvas: self.loss_ax.clear(); self.loss_canvas.draw()
             return

         player_names = [p.name for p in self.game.players]

         # Update Loss Plot (assuming loss_ax exists based on check above)
         self.plot_generator.update_loss_plot(self.loss_ax, self.loss_history, player_names)
         if self.loss_canvas: self.loss_canvas.draw() # Check canvas too

         # --- Simplex Plot Update Logic ---
         num_actions = self.game.players[0].num_actions # Assume same for all
         action_names = self.game.players[0].actions

         # --- Recreate Simplex Plot If Needed ---
         # Condition: Force recreate OR axes don't exist OR dimension changed implicitly
         needs_recreate = force_simplex_recreate or self.simplex_ax is None
         # Add check if plot type needed changes (e.g., 2 actions -> 3 actions) although UI prevents this currently
         current_plot_dims = getattr(self.simplex_ax, '_plot_effective_dims', None) # Store dims on ax if needed
         required_dims = len(self.action_indices_to_plot) if self.action_indices_to_plot else num_actions
         # Refine condition based on required dims vs current dims or type
         # ... (more sophisticated dimension change detection could go here) ...

         if needs_recreate:
             # Clear old plot widgets if they exist
             if self.simplex_toolbar:
                 self.centralWidget().layout().itemAt(1).widget().layout().removeWidget(self.simplex_toolbar)
                 self.simplex_toolbar.deleteLater()
                 self.simplex_toolbar = None
             if self.simplex_canvas:
                 self.centralWidget().layout().itemAt(1).widget().layout().removeWidget(self.simplex_canvas)
                 self.simplex_canvas.deleteLater()
                 self.simplex_canvas = None
             if self.simplex_fig:
                 plt.close(self.simplex_fig) # Close the figure resource

             # Try creating the new plot
             self.simplex_fig, self.simplex_ax = self.plot_generator.create_simplex_plot(
                 num_actions, action_names, self.action_indices_to_plot
             )

             # Add new plot widgets if creation was successful
             if self.simplex_fig and self.simplex_ax:
                 self.simplex_canvas = FigureCanvas(self.simplex_fig)
                 self.simplex_toolbar = NavigationToolbar(self.simplex_canvas, self)
                 # Find plot layout and insert widgets at the top
                 plot_layout = self.centralWidget().layout().itemAt(1).widget().layout() # HBox -> VBox for plots
                 plot_layout.insertWidget(0, self.simplex_toolbar) # Add toolbar first
                 plot_layout.insertWidget(1, self.simplex_canvas)
                 # Store effective dimensions if needed for later checks:
                 # self.simplex_ax._plot_effective_dims = required_dims
             else:
                 # Plot couldn't be created (e.g., >5 actions, no selection)
                 self.simplex_fig, self.simplex_ax, self.simplex_canvas, self.simplex_toolbar = None, None, None, None
                 # Optionally display a placeholder message in the plot area
                 # ... (code to add placeholder label if desired) ...

         # --- Update Simplex Plot Content ---
         # Only update if axes and canvas exist
         if self.simplex_ax and self.simplex_canvas:
             try:
                 self.plot_generator.update_simplex_plot(
                     self.simplex_ax, self.strategy_history, player_names, self.action_indices_to_plot
                 )
                 self.simplex_canvas.draw()
             except Exception as e:
                 print(f"Error updating simplex plot: {e}") # Log error, don't crash UI
                 # Optionally disable simplex plot updates on error?


    @pyqtSlot(str)
    def _handle_worker_error(self, error_message):
        """Shows error message from the worker thread and updates status."""
        self.log_message(f"ERROR: {error_message}")
        QMessageBox.critical(self, "Training Error", error_message)
        self.status_label.setText("Status: Error!")
        # Consider resetting automatically on error?
        self.reset_training() # Reset state on critical error

    @pyqtSlot()
    def _handle_worker_finished(self):
        """Cleans up after the worker thread finishes, updates status and 'Continue' button."""
        if not self.msne_finder: # Should not happen if worker was running
             self.log_message("Worker finished, but finder is missing.")
             # Reset UI to safe state
             self.start_button.setEnabled(False)
             self.pause_button.setEnabled(False)
             self.reset_button.setEnabled(False) # Needs re-init
             self.status_label.setText("Status: Error (Internal state missing)")
             return

        self.total_iterations_run = self.msne_finder._iteration_count # Update total count
        self.log_message(f"Training thread finished at iteration {self.total_iterations_run}.")
        self.status_label.setText(f"Status: Finished at iteration {self.total_iterations_run}")
        self.is_training_running = False
        self.is_training_paused = False

        # Update UI to allow continuing
        self.start_button.setEnabled(True) # Allow continuing
        self.start_button.setText(f"▶ Continue (+{self.CONTINUE_ITERATIONS} iters)")
        self.pause_button.setEnabled(False)
        self.pause_button.setText("❚❚ Pause")
        self.reset_button.setEnabled(True) # Can reset from finished state

        # Clean up thread object
        if self.training_thread:
             self.training_thread.quit()
             self.training_thread.wait(500)
             self.training_thread = None
        self.worker = None


    # --- Utility ---

    def log_message(self, message):
        """Appends a message to the main console output."""
        print(message) # Also print to standard console
        self.console_output.append(message)
        self.console_output.verticalScrollBar().setValue(self.console_output.verticalScrollBar().maximum())

    def closeEvent(self, event):
        """Ensure worker thread is stopped when closing the window."""
        self.log_message("Closing application...")
        self.reset_training() # Attempt clean shutdown of thread
        event.accept()


# --- Application Entry Point ---
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())