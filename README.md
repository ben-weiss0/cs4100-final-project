Ben Weiss, Marco Gracie, Nicole Li, Tianyu Chen
CS4100 Final Project
Fall 2024

**Environment**

The Gym environment is a modified version of OpenAI's default car racing environment that was featured in lecture. This modification that we found on GitHub allows for multiple cars, which better suited our project goal. Here is the link to the GitHub repo that we cloned and added onto, which also includes their own instructions for installation:

https://github.com/igilitschenski/multi_car_racing

The majority of the code in our project is untouched code from their public repository. Line 679 (creation of DQN class) and below is the code that our group added.

**Variables**

There are several key variables that vary the output of the program being run: 

EPSILON_START - starting point of epsilon
EPSILON_END - end point of epsilon
EPSILON_DECAY - the rate of decay of epsilon

LOW_REWARD_THRESHOLD - the value for Car 0 that causes episode to reset or end

train_model(number) - function called on very last line of the code that decides
                      how many episodes will be run before program quits

**Package Installation**

Note: First, follow the instructions from the provided repository to meet the majority of the package requirements. All 4 members of our group have M1 Macs, and we all had different issues with getting this to run locally. Several of us had to go to TAs for help, and we all seemed to have different solutions to ultimately resolve the issues.

Unfortunately, this makes it hard to provide robust instructions, however, we can tell you what versions one of our laptops had to aid troubleshooting as best possible.

Here are the results of running 'pip freeze' on Ben's laptop:

box2d-py==2.3.8
cloudpickle==1.6.0
e==1.4.5
filelock==3.16.1
fsspec==2024.10.0
future==1.0.0
gym==0.17.3
gym-notices==0.0.8
gym_multi_car_racing @ file:///Users/benweiss/Desktop/CS4100/cs4100-final-project
importlib_metadata==8.5.0
Jinja2==3.1.4
MarkupSafe==3.0.2
mpmath==1.3.0
networkx==3.2.1
numpy==2.0.2
pygame==2.6.1
pyglet==1.5.11
PyOpenGL==3.1.7
scipy==1.13.1
Shapely==1.7.1
sympy==1.13.3
torch==2.2.2
typing_extensions==4.12.2
zipp==3.20.2

**Gameplay**

With the default settings, two windows with seperate cars will open when hitting play. To run the game, you can simply hit the play button in VSCode from the multi_car_racing.py file. The car on the left, Car 0, is controlled by our convolutional neural network. The car on the right, Car 1, is controlled by the user. 

To control the car:

- W to accelerate
- S to brake
- A to turn left
- D to turn right

The episode ends when Car 0 reaches LOW_REWARD_THRESHOLD or every item on the track has been claimed. There is no distinct finish line, but if a car does a complete lap and stays on the track the entire time, the episode will come to an end.