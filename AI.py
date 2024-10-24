import main
from main import *
from PIL import Image
import numpy as np
class AI():

    def __init__(self):
        state = None

    def get_state(self):
        main.export_window()
        img = Image.open('state.png', 'r')
        self.state = np.array(img)

ai = AI()
ai.get_state()
print(ai.state)