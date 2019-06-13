"""import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

style.use("ggplot")

ENV_SIZE = (120, 80)

class Ball:
    def __init__(self):
        self.x = ENV_SIZE[0]/2
        self.y = ENV_SIZE[1]/2
        self.dx = np.random.randint(-3,4)
        self.dy = np.random.randint(-3,4)
        self.height = 4
        self.width = 4
    def move(self):
        self.x += self.dx
        self.y += self.dy
        if self.y <=

class Player:
    def __init__"""
