import random
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.backend import reshape
from keras.utils.np_utils import to_categorical
import csv

from board import create_dataset, play_random_move, play_scoreV2_move
from board import (CELL_X, CELL_O, RESULT_X_WINS, RESULT_O_WINS, BOARD_SIZE)

# with open('score_v_score2_100k_games_dataset.csv', 'r') as file:
#     reader = csv.reader(file)
#     svs2 = list(reader)
    
# for i in range(len(svs2)):
#     for j in range(len(svs2[i])):
#         svs2[i][j] = eval(svs2[i][j])
        
# with open('score_v_score_100k_games_dataset.csv', 'r') as file:
#     reader = csv.reader(file)
#     svs = list(reader)
    
# for i in range(len(svs)):
#     for j in range(len(svs[i])):
#         svs[i][j] = eval(svs[i][j])

with open('score_v_score_100k_games_dataset.csv', 'r') as file:
    reader = csv.reader(file)
    rvr = list(reader)
    
for i in range(len(rvr)):
    for j in range(len(rvr[i])):
        rvr[i][j] = eval(rvr[i][j])

games = rvr



def movesToBoard(moves):
    board = np.zeros((BOARD_SIZE,BOARD_SIZE),int)
    for move in moves:
        player = move[0]
        coords = move[1]
        board[coords[0]][coords[1]] = player
    return board

def getModel():
    numCells = BOARD_SIZE**2 # How many cells in a 3x3 tic-tac-toe board?
    outcomes = 3 # How many outcomes are there in a game? (draw, X-wins, O-wins)
    model = Sequential()
    model.add(Dense(400, activation='relu', input_shape=(numCells, )))
    model.add(Dropout(0.2))
    model.add(Dense(250, activation='relu'))
    model.add(Dense(150, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(75, activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(outcomes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    return model
model = getModel()
print(model.summary())

def gamesToWinLossData(games):
    X = []
    y = []
    for game in games:
        winner = game.pop()
        if winner == -1:
            winner = 2
        for move in range(len(game)):
            X.append(movesToBoard(game[:(move + 1)]))
            y.append(winner)

    X = np.array(X).reshape((-1, BOARD_SIZE**2))
    y = to_categorical(y)
    
    # Return an appropriate train/test split
    trainNum = int(len(X) * 0.8)
    return (X[:trainNum], X[trainNum:], y[:trainNum], y[trainNum:])

# Split out train and validation data
X_train, X_test, y_train, y_test = gamesToWinLossData(games)

nEpochs = 100
batchSize = 1000
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=nEpochs, batch_size=batchSize)