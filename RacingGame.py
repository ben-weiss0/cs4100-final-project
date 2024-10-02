# File contains all elements of our racing game
# Almost anything non-AI related will be found here

# Notes:
# - I couldn't get this to actually display anything
# - but here is I think solid skeleton of things we will want
# - I didn't like that library that prof recommended
# - I didn't branch here but going forward we should

from enum import Enum
import pygame
import sys

# Represents the different racing maps
# For now, this is just going to be one
class Maps(Enum):
    SIMPLE = 1

# Main class to reprsent a racing game
class RacingGame:

    # Default constructor that takes in no arguments
    def __init__(self):
        # Attributes of racing game
        # All just set to default values for time being
        # Which map is being used
        self.map = Maps.SIMPLE
        # Number of opponents
        self.numOpponents = 0
        # Number of seconds before race terminates
        self.maxTime = 120
        # Width of game screen
        self.screenWidth = 400
        # Height of game screen
        self.screenHeight = 300
        # Timer that begins at initialization
        self.time = pygame.time.Clock()

    # Initializes the game
    def startGame(self):
        pygame.init()
        window_size = ((self.screenWidth, self.screenHeight))
        screen = pygame.display.set_mode(window_size)
        # Just to get it to do something rn
        pygame.display.set_caption('Simple Pygame Window')

        # This is totally from ChatGPT and nonsense but just to get us a good idea
        # to start as we start filling this out
        isRunning = True
        while self.time < self.maxTime:
            screen.fill((255, 255, 255))
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("dfjklsjsed")
                    self.endGame()
                    #running = False  # Exit the loop to close the window

            # Fill the screen with a color (optional)
            screen.fill((0, 128, 255))  # Fill with blue color
            pygame.display.update()  # Update the display

        self.endGame()  # Call endGame when the loop ends

    # Ends the pygame session
    def endGame(self):
        pygame.quit()
        sys.exit()

# Main function
def main(args):
    # Start a default racing game
    game = RacingGame()
    game.startGame()


        