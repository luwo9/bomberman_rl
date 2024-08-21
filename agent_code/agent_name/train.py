from typing import List # list works as type annotation for new enough python versions, but since the final version is unknown, we use List

from .bombermans import BombermanBundle
from .event_maker import EventMaker

def setup_training(self):
    self.event_maker = EventMaker()

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    model: BombermanBundle = self.model
    event_maker: EventMaker = self.event_maker

    events = event_maker.make_events(old_game_state, self_action, new_game_state, events)

    model.game_events_occurred(old_game_state, self_action, new_game_state, events)

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    model: BombermanBundle = self.model
    event_maker: EventMaker = self.event_maker

    events = event_maker.make_events(last_game_state, last_action, None, events)
    
    model.game_events_occurred(last_game_state, last_action, None, events)
    if last_game_state["round"] % 100 == 0:
        model.save(self.file_name)