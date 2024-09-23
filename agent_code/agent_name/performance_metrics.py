"""
This function provides tools to monitor the performance of the agent.
"""
from abc import ABC, abstractmethod
import multiprocessing as mp
import time
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import events as e
import settings as s

# Plotting rules
N_ROWS_MAX = 4

# Default path
DEFAULT_PATH = "../../results/"


class StatTracker(ABC):
    """
    Abstract base class for stat trackers.
    A stat tracker decides what to track and how to track and monitor it.
    """

    def __init__(self, ax, path):
        """
        Initialize the stat tracker.
        
        :param ax: The axis to plot on.
        :param path: The path to save the results.
        """
        self._ax = ax
        self._path = path

    @abstractmethod
    def new_step(self, state:dict, events: list[str], next_state: dict|None):
        """
        Update the tracker about what happened in the game in the current step.
        """
        pass

    @abstractmethod
    def end_of_episode(self):
        """
        Inform the tracker that the episode has ended.
        """
        pass

    @abstractmethod
    def finish(self):
        """
        Tell the tracker the training is over to e.g. save the results.
        """
        pass


def _get_N_average(data, N_AVERAGE=50):
        """
        Get the N average of the data.
        """
        arr = np.array(data)
        n_to_be_divided = len(arr) // N_AVERAGE * N_AVERAGE
        arr = arr[:n_to_be_divided]
        return arr.reshape(-1, N_AVERAGE).mean(axis=1)


class CoinsCratesRound(StatTracker):
    """
    Tracker for the number of coins collected and crates destroyed per round.

    Tracks (n round averages):
    - Coins collected per round
    - Crates destroyed per round

    Creates:
    - Plot of the above
    - Data, averaged and not
    """

    def __init__(self, ax, path):
        super().__init__(ax, path)
        self._coins = [0]
        self._crates = [0]
        self._round = 1
        self._N_AVERAGE = 50
        self._was_plotted = False

        self._setup_plot()

    def new_step(self, state: dict, events: list[str], next_state: dict|None):
        """
        Update the tracker about what happened in the game in the current step.
        """
        for event in events:
            if event == e.COIN_COLLECTED:
                self._coins[-1] += 1
            elif event == e.CRATE_DESTROYED:
                self._crates[-1] += 1

    def end_of_episode(self):
        """
        Inform the tracker that the episode has ended.
        """

        if self._round % self._N_AVERAGE == 0:
            self._plot()
        
        # Start new round
        self._coins.append(0)
        self._crates.append(0)
        self._round += 1

    def finish(self):
        """
        Tell the tracker the training is over to e.g. save the results.
        """
        # End of episode does only mean a new episode if training is not over
        # So we have to pop the last element
        self._coins.pop()
        self._crates.pop()

        np.savetxt(self._path + "coins.txt", self._coins)
        np.savetxt(self._path + "crates.txt", self._crates)

        # Averages
        np.savetxt(self._path + "coins_avg.txt", _get_N_average(self._coins, self._N_AVERAGE))
        np.savetxt(self._path + "crates_avg.txt", _get_N_average(self._crates, self._N_AVERAGE))

    def _plot(self):
        """
        Plot the data.
        """
        ax: plt.Axes = self._ax

        last_N_average = np.mean(self._coins[-self._N_AVERAGE:])
        ax.plot(self._round, last_N_average, marker='.', color='gold', label='Coins')

        last_N_average = np.mean(self._crates[-self._N_AVERAGE:])
        ax.plot(self._round, last_N_average, marker='.', color='peru', label='Crates')

        if not self._was_plotted:
            self._was_plotted = True
            ax.legend()
    
    def _setup_plot(self):
        """
        Setup the plot.
        """
        ax: plt.Axes = self._ax

        ax.set_xlabel("Round")
        ax.set_ylabel("Average per round")
        ax.set_title(f"Coins and crates per round ({self._N_AVERAGE} round average)")
        # ax.grid()
    

class CoinsDeathsCratesKillsSteps(StatTracker):
    """
    Tracker for the number steps per coins collected, deaths, crates destroyed and kills.

    Tracks in steps of n rounds:
    - #Steps/#Coins
    - #Steps/#Deaths
    - #Steps/#Crates
    - #Steps/#Kills

    Creates:
    - Plot of the above
    - Data, averaged
    """

    def __init__(self, ax, path):
        super().__init__(ax, path)
        self._steps_coins = []
        self._steps_deaths = []
        self._steps_crates = []
        self._steps_kills = []

        self._n_coins = 0
        self._n_deaths = 0
        self._n_crates = 0
        self._n_kills = 0

        self._round = 1
        self._n_steps_since_average = 0

        self._N_AVERAGE = 20
        self._was_plotted = False

        self._setup_plot()

    def new_step(self, state: dict, events: list[str], next_state: dict|None):
        """
        Update the tracker about what happened in the game in the current step.
        """
        self._n_steps_since_average += 1

        for event in events:
            if event == e.COIN_COLLECTED:
                self._n_coins += 1
            elif event == e.GOT_KILLED:
                self._n_deaths += 1
            elif event == e.CRATE_DESTROYED:
                self._n_crates += 1
            elif event == e.KILLED_OPPONENT:
                self._n_kills += 1

    def end_of_episode(self):
        """
        Inform the tracker that the episode has ended.
        """
        if self._round % self._N_AVERAGE == 0:
            self._steps_coins.append(self._safe_div(self._n_steps_since_average, self._n_coins))
            self._steps_deaths.append(self._safe_div(self._n_steps_since_average, self._n_deaths))
            self._steps_crates.append(self._safe_div(self._n_steps_since_average, self._n_crates))
            self._steps_kills.append(self._safe_div(self._n_steps_since_average, self._n_kills))

            self._plot()

            self._n_steps_since_average = 0
            self._n_coins = 0
            self._n_deaths = 0
            self._n_crates = 0
            self._n_kills = 0

        self._round += 1

    def finish(self):
        """
        Tell the tracker the training is over to e.g. save the results.
        """
        np.savetxt(self._path + "steps_coins.txt", self._steps_coins)
        np.savetxt(self._path + "steps_deaths.txt", self._steps_deaths)
        np.savetxt(self._path + "steps_crates.txt", self._steps_crates)
        np.savetxt(self._path + "steps_kills.txt", self._steps_kills)

    def _plot(self):
        """
        Plot the data.
        """
        ax: plt.Axes = self._ax

        ax.plot(self._round, self._steps_coins[-1], marker='.', color='gold', label='Coins')
        ax.plot(self._round, self._steps_deaths[-1], marker='.', color='black', label='Deaths')
        ax.plot(self._round, self._steps_crates[-1], marker='.', color='peru', label='Crates')
        ax.plot(self._round, self._steps_kills[-1], marker='.', color='springgreen', label='Kills')

        if not self._was_plotted:
            self._was_plotted = True
            ax.legend()

    def _setup_plot(self):
        """
        Setup the plot.
        """
        ax: plt.Axes = self._ax

        ax.set_xlabel("Round")
        ax.set_ylabel("Steps per event")
        ax.set_title(f"Steps per event ({self._N_AVERAGE} round average)")
        ax.set_yscale('log')

    @staticmethod
    def _safe_div(num1, num2):
        """
        Safe division of two numbers.
        """
        if num2 == 0:
            return np.nan
        return num1 / num2


class Score(StatTracker):
    """
    Tracker for the final score of the agent aswell as the opponents.

    Tracks (n round averages):
    - Score
    - Sore of best opponent
    - Score of second best opponent
    - Score of third best opponent

    Creates:
    - Plot of the above
    - Data, averaged and not
    """

    def __init__(self, ax, path):
        super().__init__(ax, path)
        self._own_score = 0
        self._own_scores = []
        self._opponent_score_map = {}

        self._round = 1
        self._N_AVERAGE = 50
        self._was_plotted = False
        self._first_episode = True

        self._namings = ["1st", "2nd", "3rd"] + [f"{i}th" for i in range(4, s.MAX_AGENTS)] # Note that this is not fully correct e.g. it would be 21st

        self._setup_plot()

    def new_step(self, state: dict, events: list[str], next_state: dict|None):
        """
        Update the tracker about what happened in the game in the current step.
        """
        # Keep track of who has what score until the end of the episode
        self._own_score = state["self"][1]
        for name, score, *_ in state["others"]:
            self._opponent_score_map[name] = score 

    def end_of_episode(self):
        """
        Inform the tracker that the episode has ended.
        """
        if self._first_episode:
            n_opponents = len(self._opponent_score_map)
            self._opponent_scores = [[] for _ in range(n_opponents)] # Best, 2nd best, 3rd best...
            self._first_episode = False

        # Own score
        self._own_scores.append(self._own_score)

        # Get the scores of the opponents
        scores = list(self._opponent_score_map.values())
        # Now sort them in best, 2nd best, 3rd best...
        scores.sort(reverse=True)

        for i, score in enumerate(scores):
            self._opponent_scores[i].append(score)

        if self._round % self._N_AVERAGE == 0:
            self._plot()

        self._round += 1

    def finish(self):
        """
        Tell the tracker the training is over to e.g. save the results.
        """
        np.savetxt(self._path + "own_score.txt", self._own_scores)
        np.savetxt(self._path + "own_score_avg.txt", _get_N_average(self._own_scores, self._N_AVERAGE))
        for score, name in zip(self._opponent_scores, self._namings):
            np.savetxt(self._path + f"{name}_opponent_score.txt", score)
            np.savetxt(self._path + f"{name}_opponent_score_avg.txt", _get_N_average(score, self._N_AVERAGE))

    def _plot(self):
        """
        Plot the data.
        """
        ax: plt.Axes = self._ax

        last_N_average = np.mean(self._own_scores[-self._N_AVERAGE:])
        ax.plot(self._round, last_N_average, marker='.', color='green', label='Own')

        noramlizer = mpl.colors.Normalize(0, len(self._opponent_scores)-1)
        cmap = mpl.colormaps["viridis"]
        for i, (score, name) in enumerate(zip(self._opponent_scores, self._namings)):
            last_N_average = np.mean(score[-self._N_AVERAGE:])
            ax.plot(self._round, last_N_average, marker='.', color=cmap(noramlizer(i)), label=name)

        if not self._was_plotted:
            self._was_plotted = True
            ax.axhline(7, color='grey', linestyle='--')#, label='3rd place')
            ax.axhline(9, color='grey', linestyle='-.')#, label='2nd place')
            ax.axhline(12, color='grey', linestyle=':')#, label='1st place')
            ax.legend()

    def _setup_plot(self):
        """
        Setup the plot.
        """
        ax: plt.Axes = self._ax

        ax.set_xlabel("Round")
        ax.set_ylabel("Score")
        ax.set_title(f"Score ({self._N_AVERAGE} round average)")
    

class KillsDeathsSuicides(StatTracker):
    """
    Tracker for the number of kills, deaths and suicides per round.

    Tracks (n round averages):
    - Kills
    - Deaths
    - Suicides

    Creates:
    - Plot of the above
    - Data, averaged and not
    """

    def __init__(self, ax, path):
        super().__init__(ax, path)
        self._kills = [0]
        self._deaths = [0]
        self._suicides = [0]

        self._round = 1
        self._N_AVERAGE = 50
        self._was_plotted = False

        self._setup_plot()

    def new_step(self, state: dict, events: list[str], next_state: dict|None):
        """
        Update the tracker about what happened in the game in the current step.
        """
        for event in events:
            if event == e.KILLED_OPPONENT:
                self._kills[-1] += 1
            elif event == e.GOT_KILLED:
                self._deaths[-1] += 1
            elif event == e.KILLED_SELF:
                self._suicides[-1] += 1

    def end_of_episode(self):
        """
        Inform the tracker that the episode has ended.
        """
        if self._round % self._N_AVERAGE == 0:
            self._plot()
        
        # Start new round
        self._kills.append(0)
        self._deaths.append(0)
        self._suicides.append(0)
        self._round += 1

    def finish(self):
        """
        Tell the tracker the training is over to e.g. save the results.
        """
        # End of episode does only mean a new episode if training is not over
        # So we have to pop the last element
        self._kills.pop()
        self._deaths.pop()
        self._suicides.pop()

        np.savetxt(self._path + "kills.txt", self._kills)
        np.savetxt(self._path + "deaths.txt", self._deaths)
        np.savetxt(self._path + "suicides.txt", self._suicides)

        # Averages
        np.savetxt(self._path + "kills_avg.txt", _get_N_average(self._kills, self._N_AVERAGE))
        np.savetxt(self._path + "deaths_avg.txt", _get_N_average(self._deaths, self._N_AVERAGE))
        np.savetxt(self._path + "suicides_avg.txt", _get_N_average(self._suicides, self._N_AVERAGE))

    def _plot(self):
        """
        Plot the data.
        """
        ax: plt.Axes = self._ax

        last_N_average = np.mean(self._kills[-self._N_AVERAGE:])
        ax.plot(self._round, last_N_average, marker='.', color='springgreen', label='Kills')

        last_N_average = np.mean(self._deaths[-self._N_AVERAGE:])
        ax.plot(self._round, last_N_average, marker='.', color='black', label='Deaths')

        last_N_average = np.mean(self._suicides[-self._N_AVERAGE:])
        ax.plot(self._round, last_N_average, marker='.', color='red', label='Suicides')

        if not self._was_plotted:
            self._was_plotted = True
            ax.axhline(1, color='grey', linestyle='--')
            ax.legend()

    def _setup_plot(self):
        """
        Setup the plot.
        """
        ax: plt.Axes = self._ax

        ax.set_xlabel("Round")
        ax.set_ylabel("Average per round")
        ax.set_title(f"Kills, deaths and suicides per round ({self._N_AVERAGE} round average)")
        # ax.grid()




# Trackers to use

USE_TRACKERS = [
    CoinsCratesRound,
    CoinsDeathsCratesKillsSteps,
    KillsDeathsSuicides,
    Score
]


def setup_trackers(trackers, name):
    """
    Setup the trackers with figure and paths.
    """

    n_trackers = len(trackers)
    n_rows = min(n_trackers, N_ROWS_MAX)
    n_cols = -(n_trackers // -n_rows)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), layout="constrained")

    if n_rows == 1 and n_cols == 1:
        axs = np.array([axs])

    path = DEFAULT_PATH+time.strftime("%Y_%m.%d-%H.%M")+"_"+name+"/"

    setup_trackers = []
    for tracker_class, ax in zip(trackers, axs.flat):
        setup_trackers.append(tracker_class(ax, path))

    if not os.path.exists(path):
        os.makedirs(path)
    
    return setup_trackers, fig, path


def monitor_performance(queue: mp.Queue, trackers, name):
    """
    Monitor the performance of the agent.
    """
    setup_trackers_, fig, path = setup_trackers(trackers, name)
    new_round = True
    round_counter = 0

    while True:
        value = queue.get(block=True, timeout=3)
        # None signals to stop
        if value is None:
            for tracker in setup_trackers_:
                tracker.finish()
            fig.savefig(path + "plot.png")
            break

        # Else update the trackers
        state, events, next_state = value
        for tracker in setup_trackers_:
            tracker.new_step(state, events, next_state)
            if next_state is None:
                tracker.end_of_episode()
                new_round = True

        if round_counter % 10 == 0 and new_round:
            plt.pause(0.01)

        if new_round:
            new_round = False
            round_counter += 1    


class MPPerformanceMonitor:
    """
    Monitor the performance of the agent.
    """

    def __init__(self, name: str):
        self._queue = mp.Queue()
        self._process = mp.Process(target=monitor_performance, args=(self._queue, USE_TRACKERS, name), daemon=True)

    def start(self):
        """
        Start the monitor.
        """
        self._process.start()

    def stop(self):
        """
        Stop the monitor.
        """
        self._queue.put(None)
        self._process.join()

    def new_step(self, state: dict, events: list[str], next_state: dict|None):
        """
        Update the monitor about what happened in the game in the current step.
        """
        self._queue.put((state, events, next_state))


class PerformanceMonitor:
    """
    Tracking in the same process.
    """

    def __init__(self, name: str):
        self._trackers, self._fig, self._path = setup_trackers(USE_TRACKERS, name)
        self._round_counter = 0
        self._new_round = True

    def start(self):
        """
        Start the monitor.
        """
        pass

    def new_step(self, state: dict, events: list[str], next_state: dict|None):
        """
        Update the monitor about what happened in the game in the current step.
        """
        for tracker in self._trackers:
            tracker.new_step(state, events, next_state)
            if next_state is None:
                tracker.end_of_episode()
                self._new_round = True

        if self._round_counter % 10 == 0 and self._new_round:
            plt.pause(0.01)

        if self._new_round:
            self._new_round = False
            self._round_counter += 1

    def stop(self):
        """
        Stop the monitor.
        """
        for tracker in self._trackers:
            tracker.finish()
        self._fig.savefig(self._path + "plot.png")
        plt.close(self._fig)




