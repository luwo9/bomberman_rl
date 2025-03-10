MOVED_LEFT = 'MOVED_LEFT'
MOVED_RIGHT = 'MOVED_RIGHT'
MOVED_UP = 'MOVED_UP'
MOVED_DOWN = 'MOVED_DOWN'
WAITED = 'WAITED'
INVALID_ACTION = 'INVALID_ACTION'

BOMB_DROPPED = 'BOMB_DROPPED'
BOMB_EXPLODED = 'BOMB_EXPLODED'

CRATE_DESTROYED = 'CRATE_DESTROYED'
COIN_FOUND = 'COIN_FOUND'
COIN_COLLECTED = 'COIN_COLLECTED'

KILLED_OPPONENT = 'KILLED_OPPONENT'
KILLED_SELF = 'KILLED_SELF'

GOT_KILLED = 'GOT_KILLED'
OPPONENT_ELIMINATED = 'OPPONENT_ELIMINATED'
SURVIVED_ROUND = 'SURVIVED_ROUND'

# Custom
# Actual rewards
WON_GAME = 'WON_GAME'
SECOND_PLACE = 'SECOND_PLACE'
THIRD_PLACE = 'THIRD_PLACE'
LOST_GAME = 'LOST_GAME'

# (In coin heaven e.g.)
SCORE_5 = 'SCORE_5'
SCORE_10 = 'SCORE_10'
SCORE_15 = 'SCORE_15'
SCORE_20 = 'SCORE_20'
SCORE_25 = 'SCORE_25'
SCORE_30 = 'SCORE_30'
SCORE_7 = 'SCORE_7' # At least 3rd place
SCORE_9 = 'SCORE_9' # At least 2nd place
SCORE_12 = 'SCORE_12'# Already won

# Potential based shaped rewards

IN_BOMB_RANGE = 'IN_BOMB_RANGE'
IN_BOMB_RANGE_3 = 'IN_BOMB_RANGE_3' # Cooldown 3, placed in this turn
IN_BOMB_RANGE_2 = 'IN_BOMB_RANGE_2' # Cooldown 2
IN_BOMB_RANGE_1 = 'IN_BOMB_RANGE_1' # Cooldown 1
IN_BOMB_RANGE_0 = 'IN_BOMB_RANGE_0' # Cooldown 0, explode next turn

BOMB_DISTANCE_0 = 'BOMB_DISTANCE_0' # Is in the same tile as a bomb
BOMB_DISTANCE_1 = 'BOMB_DISTANCE_1' # Is one tile away from a bomb

BOMB_DROPPED_NEXT_TO_CRATE_1 = 'BOMB_DROPPED_NEXT_TO_CRATE_1' 
BOMB_DROPPED_NEXT_TO_CRATE_2 = 'BOMB_DROPPED_NEXT_TO_CRATE_2' 
BOMB_DROPPED_NEXT_TO_CRATE_4 = 'BOMB_DROPPED_NEXT_TO_CRATE_4' 
BOMB_DROPPED_NEXT_TO_CRATE_8 = 'BOMB_DROPPED_NEXT_TO_CRATE_8'
BOMB_DROPPED_NEXT_TO_CRATE_PER_CRATE = 'BOMB_DROPPED_NEXT_TO_CRATE_PER_CRATE'

BOMB_POSSIBLE_BUT_NO_CRATE_IN_RANGE = 'BOMB_POSSIBLE_BUT_NO_CRATE_IN_RANGE'

BOMB_DROPPED_NEXT_TO_OPPONENTS_1 = 'BOMB_DROPPED_NEXT_TO_OPPONENTS_1'
BOMB_DROPPED_NEXT_TO_OPPONENTS_2 = 'BOMB_DROPPED_NEXT_TO_OPPONENTS_2'
BOMB_DROPPED_NEXT_TO_OPPONENTS_3 = 'BOMB_DROPPED_NEXT_TO_OPPONENTS_3'

CLOSEST_ENEMY_16 = "CLOSEST_ENEMY_16"
CLOSEST_ENEMY_12 = "CLOSEST_ENEMY_12"
CLOSEST_ENEMY_8 = "CLOSEST_ENEMY_8"
CLOSEST_ENEMY_6 = "CLOSEST_ENEMY_6"
CLOSEST_ENEMY_4 = "CLOSEST_ENEMY_4"
CLOSEST_ENEMY_2 = "CLOSEST_ENEMY_2"
CLOSEST_ENEMY_1 = "CLOSEST_ENEMY_1"

CLOSEST_ENEMY_CLOSER = "CLOSEST_ENEMY_CLOSER"
CLOSEST_ENEMY_SAME = "CLOSEST_ENEMY_SAME"
CLOSEST_ENEMY_FURTHER = "CLOSEST_ENEMY_FURTHER"


# Other
NO_COIN = 'NO_COIN'
NO_CRATE = 'NO_CRATE'
NO_BOMB_RANGE = 'NOT_BOMB_RANGE'