import os
from . import bombermans

def setup(self):
    model: bombermans.VectorMLPSimple
    if os.path.exists("msaved-model.pt"):
        model = bombermans.load("saved-model.pt")
    else:
        model = bombermans.VectorMLPSimple()

    self.model = model

def act(self, game_state: dict):
    return self.model.act(game_state)
    