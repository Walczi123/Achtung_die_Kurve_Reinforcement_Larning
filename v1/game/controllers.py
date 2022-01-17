import random


class Controller():
    def __init__(self):
        self.name = 'Controller'

    def make_move(self, args):
        return args[0]

class Man_Controller(Controller):
    def __init__(self) -> None:
        self.name = 'Man'

    def make_move(self, args):
        return args[0]

class DQN_Controller(Controller):
    def __init__(self, model):
        self.name = 'DQN'
        self.model = model

    def make_move(self, args):
        if args[0]:
            print("nope")
        action = self.model.predict(args[1])
        return action[0]

class A2C_Controller(Controller):
    def __init__(self, model):
        self.name = 'A2C'
        self.model = model

    def make_move(self, args):
        if args[0]:
            print("nope")
        action = self.model.predict(args[1])
        return action[0]

class CNN_Controller(Controller):
    def __init__(self, model):
        self.name = 'CNN'
        self.model = model

    def make_move(self, args):
        if args[0]:
            print("nope")
        action = self.model.predict(args[1])
        return action[0]

class Random_Controller(Controller):
    def __init__(self):
        self.name = 'Random'

    def make_move(self, args):
        if args[0]:
            print("nope")
        action = random.choice([0,1,2])
        return action