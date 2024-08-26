"""
Contains general settings for Q-learning models.
"""

N_ACTIONS = 6

ACTIONS_MAP = {
    0: 'UP',
    1: 'RIGHT',
    2: 'DOWN',
    3: 'LEFT',
    4: 'WAIT',
    5: 'BOMB'
}

ACTIONS_INV_MAP = {v: k for k, v in ACTIONS_MAP.items()}