
import numpy as np
import ResNet
from keras.optimizers import SGD
from loss import softmax_cross_entropy_with_logits, softmax
from board import (create_dataset, play_random_move, play_scoreV2_move, play_showcase,
                   play_games_more_stats)

h,w,d = 6,6,2


def create_Alpha_Z_player(epoch = 4, print_summary = False):
    agent = ResNet.ResNet.build(h, w, d, 256, 36, num_res_blocks=16)
    agent.compile(loss=[softmax_cross_entropy_with_logits, 'mean_squared_error'],
              optimizer=SGD(lr=0.001, momentum=0.9))
    
    agent.load_weights(f'Models/TicTacToe_6x6/{epoch}.h5')
    def play(board):
        return play_Alpha_Z_move(board, agent)
    if print_summary:
        print(agent.summary())
    return play

def play_Alpha_Z_move(board, agent):
    max_move_index = select_valid_Alpha_Z_move(board, agent)
    return board.play_move(max_move_index)


def select_valid_Alpha_Z_move(board, agent):
    ABoard = make_ABooard(board.board_2d)
    moves = board.get_valid_move_indexes()
    pred = agent.predict(ABoard, verbose = 0)
    watchdog = 0
    while True:
        if watchdog > 50:
            return 0
        bestmove = np.argmax(pred[0][0])
        if bestmove in moves:
            return bestmove
        else:
            pred[0][0][bestmove] = -10
        watchdog +=1
            
def make_ABooard(board):
    ABoard = np.zeros([1,h,w,d])
    ABoard[0,:,:,0] = (board == 1).astype(int)
    ABoard[0,:,:,1] = (board == -1).astype(int)
    return ABoard

if __name__ == '__main__':
    
    
    play_Alpha_Z_movee = create_Alpha_Z_player()
    play_showcase(play_random_move, play_Alpha_Z_movee)