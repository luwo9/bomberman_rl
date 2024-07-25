from typing import List # list works as type annotation for new enough python versions, but since the final version is unknown, we use List

def setup_training(self):
    ...

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    ...

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    ...