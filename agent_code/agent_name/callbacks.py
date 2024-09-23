import os

from .import bombermans

ASK_FOR_NAME = False
ASK_FOR_NEW_RUN = False
ASK_FOR_TYPE = False

def ask_questions():
    if ASK_FOR_NAME:
        name = input("Enter the name of the agent: ")
    else:
        name = "final"

    if ASK_FOR_NEW_RUN:
        new_run = input("Is this a new run? (y/n): ").lower()
        if new_run == "y":
            new_run = True
        else:
            new_run = False
    else:
        new_run = False

    if ASK_FOR_TYPE:
        type_ = input("Enter the type of the agent (BombermanBundle): ")
    else:
        type_ = "CNNext"

    type_ = getattr(bombermans, type_) # Could sanitize this input

    return name, new_run, type_

def make_filepath(name):
    return name + ".pt" + ".gz"


# Callbacks


def setup(self):
    name, new_run, class_ = ask_questions()
    file_name = make_filepath(name)
    model: bombermans.BombermanBundle = class_()
    
    if os.path.exists(file_name):
        model.load(file_name, self.train)
    
    if new_run:
        model.new_run()
    
    self.model = model
    self.file_name = file_name
    self.name = name


def act(self, game_state: dict):
    model: bombermans.BombermanBundle = self.model
    return model.act(game_state)
    