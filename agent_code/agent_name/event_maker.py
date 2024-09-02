"""
Used to create custom events for the agent from other events or game states.
They can be used to combine events or create new events based on the game states.
"""
import events as e
import numpy as np
from .bomberman_base import get_blast_coords

place_map = {1: e.WON_GAME, 2: e.SECOND_PLACE, 3: e.THIRD_PLACE, 4: e.LOST_GAME}

score_map = {5: e.SCORE_5, 10: e.SCORE_10, 15: e.SCORE_15, 20: e.SCORE_20, 25: e.SCORE_25, 30: e.SCORE_30, 7: e.SCORE_7, 9: e.SCORE_9, 12: e.SCORE_12}

bomb_cooldown_map = {3: e.IN_BOMB_RANGE_3, 2: e.IN_BOMB_RANGE_2, 1: e.IN_BOMB_RANGE_1, 0: e.IN_BOMB_RANGE_0}

crates_in_bomb_range_map = {1: e.BOMB_DROPPED_NEXT_TO_CRATE_1, 2: e.BOMB_DROPPED_NEXT_TO_CRATE_2, 4: e.BOMB_DROPPED_NEXT_TO_CRATE_4, 8: e.BOMB_DROPPED_NEXT_TO_CRATE_8}

opponents_in_bomb_range_map = {1: e.BOMB_DROPPED_NEXT_TO_OPPONENTS_1, 2: e.BOMB_DROPPED_NEXT_TO_OPPONENTS_2, 3: e.BOMB_DROPPED_NEXT_TO_OPPONENTS_3}

# Implement as a class to have a persistent state if e.g. you were to reward total number of crates destroyed
class EventMaker:
    def __init__(self):
        pass
    
    # This method may be long and contain alot of code, which may not seem like the best idea
    # But it has exactly one responsibility of computing a whole list of events and for this purpose this is fine
    def make_events(self, old_game_state: dict, self_action: str, new_game_state: dict|None, events: list[str]):
        """
        Adds custom events to the event list.

        :param old_game_state: dict
        :param self_action: str
        :param new_game_state: dict
        :param events: list[str]
        :return: list[str]
        """

        events = events.copy() # To avoid any side effects

        # General calculations


        # End of game rewards
        game_ended = new_game_state is None
        if game_ended:
            # Placement rewards
            own_score = old_game_state["self"][1]
            opponents_score = [opponent[1] for opponent in old_game_state["others"]]
            final_place = sum([1 for score in opponents_score if score > own_score]) + 1
            events.append(place_map[final_place])

            return events

        # Below new_game_state can be assumed to be a dict

        # Score based rewards
        own_score = new_game_state["self"][1]
        if own_score in score_map:
            events.append(score_map[own_score])

        # Bomb related rewards
        bombs = new_game_state["bombs"]
        own_position = new_game_state["self"][3]

        bomb_range_events = []
        for (bomb_x, bomb_y), bomb_timer in bombs:
            bomb_range = get_blast_coords(bomb_x, bomb_y)
            if own_position in bomb_range:
                bomb_range_events.append(bomb_cooldown_map[bomb_timer])
            
            own_x, own_y = own_position
            if own_x == bomb_x and own_y == bomb_y:
                events.append(e.BOMB_DISTANCE_0)
            elif abs(own_x - bomb_x) + abs(own_y - bomb_y) == 1:
                events.append(e.BOMB_DISTANCE_1)

        events.extend(bomb_range_events)

        if len(bomb_range_events) > 0:
            events.append(e.IN_BOMB_RANGE)
        else:
            events.append(e.NO_BOMB_RANGE)

        # Negations
        if not e.COIN_COLLECTED in events:
            events.append(e.NO_COIN)

        if not e.CRATE_DESTROYED in events:
            events.append(e.NO_CRATE)

        
        if (self_action == 'BOMB'):
            own_x, own_y = own_position
            coords = get_blast_coords(own_x, own_y)
            x,y = coords.T
            crate_count = np.sum(old_game_state["field"][x,y] == 1)
            
            filtered_keys = [key for key in crates_in_bomb_range_map.keys() if key <= crate_count]
            if filtered_keys:
                largest_key = max(filtered_keys)
                events.append(crates_in_bomb_range_map[largest_key])

            opponents_count = 0
            opponents_position = [opponent[3] for opponent in old_game_state["others"]]
            for position in opponents_position:
                if list(position) in coords.tolist():
                    opponents_count += 1
            if opponents_count > 0:
                events.append(opponents_in_bomb_range_map[opponents_count])


            

            


        return events