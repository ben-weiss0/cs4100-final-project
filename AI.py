import game
from PIL import Image
import numpy as np


class AI:

    def __init__(self, car):
        self.state = None
        self.moves = [0, 1, 2, 3]
        self.car = car

    def get_state(self):
        game.export_window()
        img = Image.open('state.png', 'r')
        self.state = np.array(img)

    def move_forward(self):
        self.car.move_forward()

    def move_backward(self):
        self.car.move_backward()

    def rotate_left(self):
        self.car.rotate_left()

    def rotate_right(self):
        self.car.rotate_right()

    def execute(self):
        action = np.random.choice(self.moves)
        if action == 0:
            self.move_forward()
        if action == 1:
            self.move_backward()
        if action == 2:
            self.rotate_left()
        if action == 3:
            self.rotate_right()
