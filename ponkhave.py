import pygame as pg
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

def moveplayer(act):
    global hracY
    global res
    if act == 0:
        pass
    if act == 1:
        hracY += 5
        if hracY < 0:
            hracY = 0
    if act == 2:
        hracY -= 5
        if hracY > res[1]-60:
            hracY = res[1]-60
makingQtableDir = True
RUN = 1
while makingQtableDir:
    try:
        os.makedirs(f"qtables/pong2.0/run{RUN}")
        makingQtableDir = False
    except:
        RUN += 1

EPISODES = 15000
ENEMY_POINT_PENALTY = 300
POINT_REWARD = 25
DOWN_SIZE = 30
START_EPSILON = 0.9
EPS_DECAY = 0.9998
SHOW_EVERY = 500
start_q_table = None
LEARNING_RATE = 0.1
DISCOUNT = 0.95
epsilonReIn = 500
stepsPerEpisode = 2000

with open(f"qtables/pong2.0/run{RUN}/opts.txt", "w") as f:
    f.write(f"EPISODES = {EPISODES}\nENEMY_POINT_PENALTY = {ENEMY_POINT_PENALTY}\nPOINT_REWARD = {POINT_REWARD}\nDOWN_SIZE = {DOWN_SIZE}\nSTART_EPSILON = {START_EPSILON}\nEPS_DECAY = {EPS_DECAY}\nSHOW_EVERY = {SHOW_EVERY}\nstart_q_table = {start_q_table}\nLEARNING_RATE = {LEARNING_RATE}\nDISCOUNT = {DISCOUNT}\nepsilonReIn = {epsilonReIn}\nstepsPerEpisode = {stepsPerEpisode}")

CERNA = (0,0,0)
BILA = (250,250,250)
MODRA = (0,40,250)
ZELENA = (0,255,0)
CERVENA = (250,0,0)
DIVNA = (154, 200, 245)

res = (800,600)
pg.init()
hodiny = pg.time.Clock()
okno = pg.display.set_mode(res)

start_q_table = None

tillEpsilon = 0
epsilon = START_EPSILON

q_table = {}
if start_q_table is None:
    for y in range(-res[0],res[0]+1):
        q_table[(y)] = [np.random.uniform(-5,0) for i in range(3)]
    q_table[(-12345)] = [np.random.uniform(-5,0) for i in range(3)]
else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

episode_rewards = []

#print(q_table)
for episode in range(EPISODES):
    episode_reward = 0
    hracY = 270
    nahoru = False
    dolu = False

    micX = 395
    micY = 295
    micSmerX = np.random.randint(-5,6)
    micSmerY = np.random.randint(-5,6)

    pocY = 270
    pocScore = 0
    hracScore = 0
    for step in range(stepsPerEpisode):
        for u in pg.event.get():
            if u.type == pg.QUIT:
                hraj = False
        if micSmerX == 0:
            micSmerX += 1

        reward = 0
        ballYonX = ((micX-20)*micSmerY)/micSmerX + micSmerY
        #print(ballYonX)
        while ballYonX < 0 or ballYonX > res[0]:
            if ballYonX < 0:
                ballYonX = -ballYonX
            if ballYonX > res[0]:
                ballYonX = ballYonX-res[0]
        if micSmerX < 0:
            obs = (int(ballYonX - hracY))
        else:
            obs = (-12345)
        #print(ballYonX)
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0,3)
        #print(action)
        ballPlayerYdiff = abs(ballYonX-hracY)
        moveplayer(action)

        #POHYB MICKU
        micX += micSmerX
        micY += micSmerY

        if micX<0:
            micSmerX*=-1
            micX = 395
            micY = 295
            pocScore+=1
            reward -= ENEMY_POINT_PENALTY

        if micY>590:
            micSmerY*=-1
        if micY<0:
            micSmerY*=-1
        if micX>res[0]:
            micSmerX*=-1
            micX = 395
            micY = 295
            hracScore+=1
            reward += POINT_REWARD

        #POHYB POCITACE
        if micSmerX>0:
            if pocY+25>micY:
                pocY-=5
            if pocY+25<micY:
                pocY+=5

        if hracY < 0:
            hracY = 0
        if hracY > res[1]-60:
            hracY = res[1]-60

        #ODRAZI HRAC MICEK?
        if micX>=20 and micX<=30 and micY>=hracY-10 and micY<=hracY+60:
            micSmerX*=-1
            micSmerY = ((micY - hracY - 25)//5 )
            micX=30

        #ODRAZI POCITAC MICEK?
        if micX>=770 and micX<=780 and micY>=pocY-10 and micY<=pocY+60:
            micSmerX*=-1
            micSmerY = ((micY - pocY - 25)//5 )
            micX=760

        if micSmerX < -10:
            micSmerX = -10
        if micSmerX > 10:
            micSmerX = 10
        if micSmerY < -10:
            micSmerY = -10
        if micSmerY > 10:
            micSmerY = 10

        ballYonX = ((micX-20)*micSmerY)/micSmerX + micY
        while ballYonX < 0 or ballYonX > res[0]:
            if ballYonX < 0:
                ballYonX = -ballYonX
            if ballYonX > res[0]:
                ballYonX = ballYonX-res[0]
        if micSmerX < 0:
            new_obs = (int(ballYonX - hracY))
        else:
            new_obs = (-12345)
        if abs(ballYonX-hracY)<ballPlayerYdiff:
            reward += 10
        if abs(ballYonX-hracY)>ballPlayerYdiff:
            reward += -10
        if abs(ballYonX-hracY) == 0:
            reward += 20
        #print(new_obs, ballYonX, hracY)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]

        if reward == POINT_REWARD:
            new_q = POINT_REWARD
        elif reward == -ENEMY_POINT_PENALTY:
            new_q = -ENEMY_POINT_PENALTY
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

        q_table[obs][action] = new_q
        episode_reward += reward
        epsilon *= EPS_DECAY
        #KRESLENI OBRAZU
        if episode % SHOW_EVERY == 0:
            okno.fill(CERNA)
            pg.draw.rect(okno, CERVENA, (20, hracY, 10, 60))
            pg.draw.rect(okno, MODRA, (770, pocY, 10, 60))
            #for i in range(30):
            #    pg.draw.rect(okno, BILA, (397, i*20+5, 6, 10))
            #pg.draw.circle(okno, BILA, (400,300), 100, 6)

            pg.draw.rect(okno, ZELENA, (micX, micY, 10, 10))
            pg.display.update()
            hodiny.tick(90)
    if epsilon < 0.1 and tillEpsilon == 0:
        tillEpsilon = epsilonReIn
        epsilon = 0
    if tillEpsilon > 1:
        tillEpsilon -= 1
    if tillEpsilon == 1:
        epsilon = START_EPSILON
        tillEpsilon = 0
    if episode % SHOW_EVERY == 0:
        with open(f"qtables/pong2.0/run{RUN}/q_table-{episode}-{int(time.time())}.pickle", "wb") as f:
            pickle.dump(q_table, f)
    if episode % 100 == 0:
        print(f"EP Reward on episode {episode}:{episode_reward}, Epsilon: {epsilon}, Past {SHOW_EVERY} episodes avg: {sum(episode_rewards[-SHOW_EVERY:])/SHOW_EVERY}")
    episode_rewards.append(episode_reward)
pg.quit()
with open(f"qtables/pong2.0/run{RUN}/q_table-final-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)

plt.plot([i for i in range(len(episode_rewards))], episode_rewards)
plt.ylabel("episode rewards")
plt.xlabel("episode number")
plt.show()
