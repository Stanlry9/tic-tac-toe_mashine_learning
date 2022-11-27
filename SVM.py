import numpy as np
# import statistics as stats
import random
# import operator
from sklearn import svm
# from sklearn.model_selection import train_test_split
# from sklearn.decomposition import TruncatedSVD
import time 
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from matplotlib import pyplot as plt

from board import BoardCache, Board
from board import play_game, play_random_move, play_scoreV2_move, get_result, play_showcase
from board import (CELL_X, CELL_O, RESULT_X_WINS, RESULT_O_WINS, BOARD_SIZE)


TRAINING_GAMES = 1000
TESTING_GAMES = 100

class SVM:
    def __init__(self):
        self.clf = None

    def update_clf(self, dataset, results):
        self.clf = svm.SVC()
        self.clf.fit(dataset, results)
        
    def get_clf(self):
        return self.clf
    
new_model_SVM = SVM()    

def play_training_SVM_games(model, total_games=TRAINING_GAMES, strategy_x = play_random_move, strategy_o = play_random_move, plot = False):
    
    # if model == None:
    #     model = SVM()
    start_time = time.time()
    training_time = 0

    dataset = np.zeros([total_games,BOARD_SIZE**2],'b')
    results = np.zeros(total_games,'b')
    
    for i in range (total_games):
        if (i+1) % (total_games / 10) == 0:
            training_time = time.time()- start_time
            print("training games ",100*i/total_games,"% comlete. It took ",training_time,"s")
            # print("training games ",100*i/total_games,"% comlete\n")
        game = play_game(strategy_x, strategy_o)
        dataset[i] = game.board
        results[i] = get_result(game)
        
    model.update_clf(dataset, results)
    
    if plot:
        cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=0)
        plot_learning_curve(model.get_clf(), 'Learning curves', dataset, results, cv=cv, n_jobs=4)

def test_SVM_accuracy(model, total_games=TESTING_GAMES, strategy_x = play_random_move, strategy_o = play_random_move):
    
    hit = 0
    for i in range (total_games):
        # if (i) % (total_games / 10) == 0:
        #     print("testing games ",100*i/total_games,"% comlete\n")
        test_game = play_game(strategy_x, strategy_o)
        test_result = get_result(test_game)
        clf = model.get_clf()
        prediction = clf.predict([test_game.board])
        if test_result == prediction:
            hit += 1
    
    acc = 100*hit/total_games
    print ("accuracy = ", acc, "%")

def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    print("plotting learning curves - might take a while\n")
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid()
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")
    
    plt.show()

    return plt

def create_SVM_player(model):
    def play(board):
        return play_SVM_move(board, model)

    return play


def play_SVM_move(board, model):
    if board.lastmove == None:
        player = CELL_X
    elif board.last2move == None:
        player = CELL_O
    else:
        player = -board.board_2d[board.lastmove]
    valid_moves = board.get_valid_move_indexes()
    board_copy = np.copy(board.board)
    good_moves = []
    for i in range (len(valid_moves)):
        test_move_index = valid_moves[i]
        board_copy[test_move_index]=player
        clf = model.get_clf()
        if clf.predict([board_copy]) == player:
            good_moves.append(test_move_index)
    if len(good_moves) == 0:
        return board.play_move(random.choice(valid_moves))
    move_index = random.choice(good_moves)    
    return board.play_move(move_index)

# new_model1 = SVM()
# new_model2 = SVM()
# play_training_SVM_games(new_model1, 100)
# play_training_SVM_games(new_model2, 100, plot = True)
# test_SVM_accuracy(new_model1, 100)
# test_SVM_accuracy(new_model2, 100)

# play_SVMm_move = create_SVM_player(new_model1)
# play_showcase(play_SVMm_move, play_random_move)
