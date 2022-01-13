class Controller():
    def __init__(self) -> None:
        pass

    def make_move(self, args):
        pass

class Man_Controller(Controller):
    def __init__(self) -> None:
        pass

    def make_move(self, args):
        return args

class DQN_Controller(Controller):
    def __init__(self, model):
        pass

    def make_move(self, args):
        return args
