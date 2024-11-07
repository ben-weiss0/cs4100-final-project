import pygame
import time
import math
from utils import *
from PIL import Image
import numpy as np
import pickle

pygame.font.init()

GRASS = scale_image(pygame.image.load("imgs/grass.jpg"), 2.5)
TRACK = scale_image(pygame.image.load("imgs/track.png"), 0.9)

TRACK_BORDER = scale_image(pygame.image.load("imgs/track-border.png"), 0.9)
TRACK_BORDER_MASK = pygame.mask.from_surface(TRACK_BORDER)

FINISH = pygame.image.load("imgs/finish.png")
FINISH_MASK = pygame.mask.from_surface(FINISH)
FINISH_POSITION = (130, 250)

RED_CAR = scale_image(pygame.image.load("imgs/red-car.png"), 0.55)
GREEN_CAR = scale_image(pygame.image.load("imgs/green-car.png"), 0.55)

WIDTH, HEIGHT = TRACK.get_width(), TRACK.get_height()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Racing Game!")

MAIN_FONT = pygame.font.SysFont("comicsans", 44)

FPS = 60
PATH = [(175, 119), (110, 70), (56, 133), (70, 481), (318, 731), (404, 680), (418, 521), (507, 475), (600, 551),
        (613, 715), (736, 713),
        (734, 399), (611, 357), (409, 343), (433, 257), (697, 258), (738, 123), (581, 71), (303, 78), (275, 377),
        (176, 388), (178, 260)]


class GameInfo:
    LEVELS = 10

    def __init__(self, level=1):
        self.level = level
        self.started = False
        self.level_start_time = 0

    def next_level(self):
        self.level += 1
        self.started = False

    def reset(self):
        self.level = 1
        self.started = False
        self.level_start_time = 0

    def game_finished(self):
        return self.level > self.LEVELS

    def start_level(self):
        self.started = True
        self.level_start_time = time.time()

    def get_level_time(self):
        if not self.started:
            return 0
        return round(time.time() - self.level_start_time)


class AbstractCar:
    def __init__(self, max_vel, rotation_vel):
        self.img = self.IMG
        self.max_vel = max_vel
        self.vel = 0
        self.rotation_vel = rotation_vel
        self.angle = 0
        self.x, self.y = self.START_POS
        self.acceleration = 0.1

    def rotate(self, left=False, right=False):
        if left:
            self.angle += self.rotation_vel
        elif right:
            self.angle -= self.rotation_vel

    def draw(self, win):
        rotated_image, rotated_rect = rotate_center(self.img, (self.x, self.y), self.angle)
        win.blit(rotated_image, rotated_rect.topleft)

    def move_forward(self):
        self.vel = min(self.vel + self.acceleration, self.max_vel)
        self.move()

    def move_backward(self):
        self.vel = max(self.vel - self.acceleration, -self.max_vel / 2)
        self.move()

    def move(self):
        radians = math.radians(self.angle)
        vertical = math.cos(radians) * self.vel
        horizontal = math.sin(radians) * self.vel

        self.y -= vertical
        self.x -= horizontal

    def collide(self, mask, x=0, y=0):
        rotated_image, rotated_rect = rotate_center(self.img, (self.x, self.y), self.angle)
        car_mask = pygame.mask.from_surface(rotated_image)
        offset = (int(rotated_rect.left - x), int(rotated_rect.top - y))
        poi = mask.overlap(car_mask, offset)
        return poi

    def reset(self):
        self.x, self.y = self.START_POS
        self.angle = 0
        self.vel = 0


class PlayerCar(AbstractCar):
    IMG = RED_CAR
    START_POS = (180, 200)

    def reduce_speed(self):
        self.vel = max(self.vel - self.acceleration / 2, 0)
        self.move()

    def bounce(self):
        self.vel = -self.vel
        self.move()


class ComputerCar(AbstractCar):
    IMG = GREEN_CAR
    START_POS = (150, 200)

    def __init__(self, max_vel, rotation_vel, path=[]):
        super().__init__(max_vel, rotation_vel)
        self.path = path
        self.current_point = 0
        self.vel = max_vel
        self.state = None
        self.moves = [0, 1, 2, 3]
        self.x, self.y = self.START_POS

    def update_position(self):
        """
        Update the x and y coordinates of the computer car based on its current velocity and angle.
        """
        radians = math.radians(self.angle)
        dx = math.cos(radians) * self.vel
        dy = math.sin(radians) * self.vel

        self.x += dx
        self.y -= dy

        # Ensure car stays within screen bounds
        self.x = max(0, min(self.x, WIDTH))
        self.y = max(0, min(self.y, HEIGHT))

    def draw_points(self, win):
        for point in self.path:
            pygame.draw.circle(win, (255, 0, 0), point, 5)

    def draw(self, win):
        super().draw(win)
        # self.draw_points(win)

    def get_state(self):
        pygame.image.save(WIN, 'state.png')
        img = Image.open('state.png', 'r')
        full_state = np.array(img)

        car_x, car_y = int(self.x), int(self.y)

        start_x = max(0, car_x - 15)
        end_x = min(full_state.shape[1], car_x + 15)
        start_y = max(0, car_y - 15)
        end_y = min(full_state.shape[0], car_y + 15)

        cropped_state = full_state[start_y:end_y, start_x:end_x]

        quantized_state = np.round(cropped_state / 10).astype(int)

        self.state = tuple(quantized_state.flatten())

    def move_forward(self):
        super().move_forward()

    def move_backward(self):
        super().move_backward()

    def rotate_left(self):
        super().rotate(left=True)

    def rotate_right(self):
        super().rotate(left=True)

    def next_level(self, level):
        self.reset()
        self.vel = self.max_vel + (level - 1) * 0.2
        self.current_point = 0

    def execute(self, action):
        reward = 0
        done = False
        if action == 0:
            self.move_forward()
        if action == 1:
            self.move_backward()
        if action == 2:
            self.rotate_left()
        if action == 3:
            self.rotate_right()
        if self.collide(TRACK_BORDER_MASK) is not None:
            reward = -5000
        elif self.collide(
                FINISH_MASK, *FINISH_POSITION) is not None:
            reward = 10000
            done = True
        self.get_state()
        return self.state, reward, done


def draw(win, images, player_car, computer_car, game_info):
    for img, pos in images:
        win.blit(img, pos)

    level_text = MAIN_FONT.render(
        f"Level {game_info.level}", 1, (255, 255, 255))
    win.blit(level_text, (10, HEIGHT - level_text.get_height() - 70))

    time_text = MAIN_FONT.render(
        f"Time: {game_info.get_level_time()}s", 1, (255, 255, 255))
    win.blit(time_text, (10, HEIGHT - time_text.get_height() - 40))

    vel_text = MAIN_FONT.render(
        f"Vel: {round(player_car.vel, 1)}px/s", 1, (255, 255, 255))
    win.blit(vel_text, (10, HEIGHT - vel_text.get_height() - 10))

    player_car.draw(win)
    computer_car.draw(win)
    pygame.display.update()


def move_player(player_car):
    keys = pygame.key.get_pressed()
    moved = False

    if keys[pygame.K_a]:
        player_car.rotate(left=True)
    if keys[pygame.K_d]:
        player_car.rotate(right=True)
    if keys[pygame.K_w]:
        moved = True
        player_car.move_forward()
    if keys[pygame.K_s]:
        moved = True
        player_car.move_backward()

    if not moved:
        player_car.reduce_speed()


def handle_collision(player_car, computer_car, game_info):
    if player_car.collide(TRACK_BORDER_MASK) != None:
        player_car.bounce()

    computer_finish_poi_collide = computer_car.collide(
        FINISH_MASK, *FINISH_POSITION)
    if computer_finish_poi_collide != None:
        blit_text_center(WIN, MAIN_FONT, "You lost!")
        pygame.display.update()
        pygame.time.wait(5000)
        game_info.reset()
        player_car.reset()
        computer_car.reset()

    player_finish_poi_collide = player_car.collide(
        FINISH_MASK, *FINISH_POSITION)
    if player_finish_poi_collide != None:
        if player_finish_poi_collide[1] == 0:
            player_car.bounce()
        else:
            game_info.next_level()
            player_car.reset()
            computer_car.next_level(game_info.level)


class Game:

    def __init__(self):
        self.run = True
        self.clock = pygame.time.Clock()
        self.images = [(GRASS, (0, 0)), (TRACK, (0, 0)),
                       (FINISH, FINISH_POSITION), (TRACK_BORDER, (0, 0))]
        self.player_car = PlayerCar(4, 4)
        self.computer_car = ComputerCar(4, 4)
        self.game_info = GameInfo()

    # def run(self):
    #     while self.run:
    #         self.clock.tick(FPS)
    #
    #         draw(WIN, self.images, self.player_car, self.computer_car, self.game_info)
    #
    #         while not self.game_info.started:
    #             blit_text_center(
    #                 WIN, MAIN_FONT, f"Press any key to start level {self.game_info.level}!")
    #             pygame.display.update()
    #             for event in pygame.event.get():
    #                 if event.type == pygame.QUIT:
    #                     pygame.quit()
    #                     break
    #
    #                 if event.type == pygame.KEYDOWN:
    #                     self.game_info.start_level()
    #
    #         for event in pygame.event.get():
    #             if event.type == pygame.QUIT:
    #                 run = False
    #                 break
    #
    #         handle_collision(self.player_car, self.computer_car, self.game_info)
    #
    #         if game_info.game_finished():
    #             blit_text_center(WIN, MAIN_FONT, "You won the game!")
    #             pygame.time.wait(5000)
    #             game_info.reset()
    #             player_car.reset()
    #             computer_car.reset()
    #
    #     pygame.quit()

    def reset(self):
        self.game_info.reset()
        self.player_car.reset()
        self.computer_car.reset()
        draw(WIN, self.images, self.player_car, self.computer_car, self.game_info)
        self.computer_car.get_state()
        obs = self.computer_car.state
        return obs, 0, False

    def step(self, action):
        obs, reward, done = self.computer_car.execute(action)
        return obs, reward, done


def Q_learning(num_episodes=10000, gamma=0.9, epsilon=1, decay_rate=0.999):
    """
    Run Q-learning algorithm for a specified number of episodes.

    Parameters:
    - num_episodes (int): Number of episodes to run.
    - gamma (float): Discount factor.
    - epsilon (float): Exploration rate.
    - decay_rate (float): Rate at which epsilon decays. Epsilon is decayed as epsilon = epsilon * decay_rate after each episode.

    Returns:
    - Q_table (dict): Dictionary containing the Q-values for each state-action pair.
    """

    Q_table = {}
    game = Game()

    # of updates
    updates = np.zeros((100000000, 4))

    for episode in range(num_episodes):
        obs_prev, reward, done = game.reset()
        while not done:
            state = hash(obs_prev)
            if Q_table.get(state) is None:
                Q_table[state] = np.zeros(6)
            if np.random.random() < epsilon:
                action = np.random.choice(range(6))
            else:
                action = np.argmax(Q_table[pd.util.hash_array(obs_prev)])
            obs_next, reward, done = game.step(action)

            next_state = hash(obs_next)
            if Q_table.get(next_state) is None:
                Q_table[next_state] = np.zeros(6)
            n = 1 / (1 + updates[state][action])
            qminus1 = Q_table[state][action]
            if Q_table.get(next_state) is None or done:
                V_next = 0
            else:
                V_next = max(Q_table.get(next_state))
            Q_table[state][action] = (1 - n) * qminus1 + n * (
                    reward + gamma * V_next)
            updates[state][action] += 1
            obs_prev = obs_next
        epsilon = epsilon * decay_rate
    return Q_table


decay_rate = 0.999997

Q_table = Q_learning(num_episodes=1000000, gamma=0.9, epsilon=1, decay_rate=decay_rate)  # Run Q-learning

# Save the Q-table dict to a file
with open('Q_table.pickle', 'wb') as handle:
    pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)
