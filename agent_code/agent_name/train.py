from typing import List # list works as type annotation for new enough python versions, but since the final version is unknown, we use List

from .bombermans import BombermanBundle
from .performance_metrics import PerformanceMonitor

def end_run(self):
    model: BombermanBundle = self.model
    perfomance_monitor: PerformanceMonitor = self.perfomance_monitor

    model.save(self.file_name)
    perfomance_monitor.stop()


# Callbacks


def setup_training(self):
    self.perfomance_monitor = PerformanceMonitor(self.name)
    self.perfomance_monitor.start()

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    model: BombermanBundle = self.model
    perfomance_monitor: PerformanceMonitor = self.perfomance_monitor

    perfomance_monitor.new_step(old_game_state, events, new_game_state)
    model.game_events_occurred(old_game_state, self_action, new_game_state, events)

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    model: BombermanBundle = self.model
    perfomance_monitor: PerformanceMonitor = self.perfomance_monitor

    perfomance_monitor.new_step(last_game_state, events, None)
    model.game_events_occurred(last_game_state, last_action, None, events)

    current_round = last_game_state["round"]
    if current_round % 100 == 0:
        model.save(self.file_name)

    if current_round == self.n_rounds:
        end_run(self)