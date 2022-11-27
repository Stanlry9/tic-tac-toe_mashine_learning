import random
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.backend import reshape
from keras.utils.np_utils import to_categorical
import csv
import itertools
import copy
import matplotlib.pyplot as plt

from board import (create_dataset, play_random_move, play_scoreV2_move, play_showcase,
                   play_games_more_stats)
from board import (CELL_X, CELL_O, RESULT_X_WINS, RESULT_O_WINS, BOARD_SIZE)


def load_games(file_name):
    # with open('score2_v_score2_100k.npy', 'rb') as f:
    with open(file_name, 'rb') as f:
        games = np.load(f)
        wins = np.load(f)
    return games, wins


def train_model(file_name, nEpochs, batchSize, make_plots = False):
    games, wins = load_games(file_name)
    model = getModel()
    X_train, X_test, y_train, y_test = gamesToWinLossData(games, wins)
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=nEpochs, batch_size=batchSize)
    if make_plots:
        plt.plot(history.history['acc'], label='accuracy')
        plt.plot(history.history['val_acc'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0., 1])
        plt.legend(loc='lower right')
    return model

    
def create_dcnn_player(model):
    def play(board):
        return play_dcnn_move_s(board, model)
    return play


def play_dcnn_move_s(board, model):
    max_move_index = select_valid_dcnn_move(board, model)
    return board.play_move(max_move_index)


def select_valid_dcnn_move(board, model, rnd=0):
    scores = []
    moves = board.get_valid_move_indexes()
    if board.lastmove == None:
        player = 1
    else: 
        player = -board.board_2d[board.lastmove]
        
    # Make predictions for each possible move
    for i in range(len(moves)):
        future = copy.copy(board.board_2d)
        future[int(moves[i]/BOARD_SIZE)][moves[i]% BOARD_SIZE] = player
        future = future.reshape([1, BOARD_SIZE,BOARD_SIZE,1])
        prediction = model(future, training=False)[0]
        # prediction = model.predict(future, verbose=0)[0] #extrtimely slow
        if player == 1:
            winPrediction = prediction[1]
            lossPrediction = prediction[2]
        else:
            winPrediction = prediction[2]
            lossPrediction = prediction[1]
        drawPrediction = prediction[0]
        if winPrediction - lossPrediction > 0:
            scores.append(winPrediction - lossPrediction)
        else:
            scores.append(drawPrediction - lossPrediction)

    # Choose the best move with a random factor
    bestMoves = np.flip(np.argsort(scores))
    for i in range(len(bestMoves)):
        if random.random() * rnd < 0.5:
            return moves[bestMoves[i]]

    # Choose a move completely at random
    return moves[random.randint(0, len(moves) - 1)]


def getModel():
    inputshape = (BOARD_SIZE, BOARD_SIZE, 1)
    outcomes = 3 # How many outcomes are there in a game? (draw, X-wins, O-wins)
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', input_shape = inputshape))
    model.add(MaxPooling2D(2,2))
    model.add(Flatten())
    model.add(Dense(150, activation='relu'))
    model.add(Dense(75, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(outcomes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model


def gamesToWinLossData(games, wins):
    X = games
    y = []
    for i in range(len(wins)):
        n = wins[i][0]
        k = 0
        while (k<n):
            y.append(wins[i][1])
            k += 1

    y = to_categorical(y)
    
    # Return an appropriate train/test split
    trainNum = int(len(X) * 0.8)
    return (X[:trainNum], X[trainNum:], y[:trainNum], y[trainNum:])

# model = getModel()
# print(model.summary())

# # Split out train and validation data
# X_train, X_test, y_train, y_test = gamesToWinLossData(games, wins)

# nEpochs = 10
# batchSize = 20
# history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=nEpochs, batch_size=batchSize)

# model = train_model('conv_score2_v_score2_100k.npy', 100, 2000, True)
# model.save('model_conv_scorev2')

# model = keras.models.load_model('model_conv_scorev2')


# play_dcnn_move = create_dcnn_player(model)

# play_games_more_stats(1000, play_scoreV2_move, play_dcnn_move, print_stats = True)