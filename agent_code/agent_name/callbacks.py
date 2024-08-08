import os
from . import bombermans

def setup(self):
    model: bombermans.VectorMLPSimple
    if os.path.exists("saved-model.pt"):
        model = bombermans.VectorMLPSimple.load("saved-model.pt", self.train)
    else:
        model = bombermans.VectorMLPSimple(self.train)

    self.model = model

def act(self, game_state: dict):
    return self.model.act(game_state)
    