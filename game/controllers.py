from game.common import prepro


class Controller():
    def __init__(self) -> None:
        pass

    def make_move(self, args):
        pass

class Man_Controller(Controller):
    def __init__(self) -> None:
        pass

    def make_move(self, args):
        return args[0]

class DQN_Controller(Controller):
    def __init__(self, model):
        self.model = model

    def make_move(self, args):
        if args[0]:
            print("hhehe")
        return self.model.predict(args[1])
