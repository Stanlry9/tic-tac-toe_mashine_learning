import random
import itertools
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import csv

from transform import Transform, Identity, Rotate90, Flip

TRANSFORMATIONS = [Identity(), Rotate90(1), Rotate90(2), Rotate90(3),
                   Flip(np.flipud), Flip(np.fliplr),
                   Transform(Rotate90(1), Flip(np.flipud)),
                   Transform(Rotate90(1), Flip(np.fliplr))]

BOARD_SIZE = 6
BOARD_DIMENSIONS = (BOARD_SIZE, BOARD_SIZE)
NEED_TO_WIN = 4

CELL_X = 1
CELL_O = -1
CELL_EMPTY = 0

RESULT_X_WINS = 1
RESULT_O_WINS = -1
RESULT_DRAW = 0
RESULT_NOT_OVER = 2

new_board = np.array([CELL_EMPTY] * BOARD_SIZE ** 2,'b')
  
  
def play_human_game(human_move, AI_strategy):
    board = Board()
    if human_move == CELL_X:
        player_strategies = itertools.cycle([play_human_move, AI_strategy])
    elif human_move == CELL_O: 
        player_strategies = itertools.cycle([AI_strategy, play_human_move])
    else:
        print("Wrong human_move indicator - plese enter 1 or -1")
        return 0
    
    while not board.is_gameover():
        board.print_board()
        play = next(player_strategies)
        board = play(board)
        
    if board.is_gameover():
        board.print_board()
        if(board.get_game_result() == 1):
            print("X wins!")
        if(board.get_game_result() == -1):
            print("O wins!")
        if(board.get_game_result() == 0):
            print("Draw!")   

    return board


def play_showcase(x_strategy, o_strategy):
    board = Board()
    player_strategies = itertools.cycle([x_strategy, o_strategy])

    while not board.is_gameover():
        board.print_board()
        play = next(player_strategies)
        board = play(board)
        
    if board.is_gameover():
        board.print_board()
        if(board.get_game_result() == 1):
            print("X wins!")
        if(board.get_game_result() == -1):
            print("O wins!")
        if(board.get_game_result() == 0):
            print("Draw!")   

    return board


def play_test_game(x_strategy, o_strategy):
    board = Board()
    player_strategies = itertools.cycle([x_strategy, o_strategy])

    while not board.is_gameover():
        play = next(player_strategies)
        board = play(board)
        
    if board.is_gameover():
        board.print_board()
        if(board.get_game_result() == 1):
            print("X wins!")
        if(board.get_game_result() == -1):
            print("O wins!")
        if(board.get_game_result() == 0):
            print("Draw!")   

    return board


def play_game(x_strategy, o_strategy):
    board = Board()
    player_strategies = itertools.cycle([x_strategy, o_strategy])

    while not board.is_gameover():
        play = next(player_strategies)
        board = play(board)

    return board


def simulate_game(x_strategy, o_strategy):
    history = []
    board = Board()
    player_strategies = itertools.cycle([x_strategy, o_strategy])
    
    while not board.is_gameover():
        play = next(player_strategies)
        board = play(board)
        history.append(board.get_last_player_and_move())
        
    if board.is_gameover():
        history.append(board.get_game_result())
        
    return history


def create_dataset(total_games, x_strategy, o_strategy):
    
    games = [simulate_game(x_strategy, o_strategy) for _ in range(total_games)]
    
    return games


def simulate_game_conv(x_strategy, o_strategy):
    history = np.ndarray([0,BOARD_SIZE,BOARD_SIZE,1], 'h')
    board = Board()
    player_strategies = itertools.cycle([x_strategy, o_strategy])
    
    while not board.is_gameover():
        play = next(player_strategies)
        board = play(board)
        board_4d = board.board_2d.reshape([1,BOARD_SIZE,BOARD_SIZE,1])
        history = np.append(history, board_4d, axis = 0 )
        
    if board.is_gameover():
        winner = board.get_game_result()
    if winner == -1:
        winner = 2
    
    return history, winner


def create_conv_dataset(total_games, x_strategy, o_strategy):
    
    games = np.ndarray([0,BOARD_SIZE,BOARD_SIZE,1], 'h')
    wins = np.ndarray([0,2], int)
    
    for i in range(total_games):
        game, win = simulate_game_conv(x_strategy, o_strategy) 
        games = np.append(games, game, axis = 0)
        win_stat = np.array([len(game), win])
        win_stat = win_stat.reshape([1,2])
        wins = np.append(wins, win_stat, axis = 0)
        if i % (total_games/100) == 0:
            print (i, ' na ', total_games , ' gier rozegranych')
    
    return games, wins


def play_games(total_games, x_strategy, o_strategy, play_single_game=play_game):
    results = {
        RESULT_X_WINS: 0,
        RESULT_O_WINS: 0,
        RESULT_DRAW: 0
    }
    
    for g in range(total_games):
        end_of_game = (play_single_game(x_strategy, o_strategy))
        result = end_of_game.get_game_result()
        results[result] += 1

    x_wins_percent = results[RESULT_X_WINS] / total_games * 100
    o_wins_percent = results[RESULT_O_WINS] / total_games * 100
    draw_percent = results[RESULT_DRAW] / total_games * 100
    
    print("\n-----------\n")
    print(f"X wins: {x_wins_percent:.2f}%")
    print(f"O wins: {o_wins_percent:.2f}%")
    print(f"draws  : {draw_percent:.2f}%")
    
    return  x_wins_percent, o_wins_percent, draw_percent


def play_games_more_stats(total_games, x_strategy, o_strategy, play_single_game=play_game, print_stats = False):
    results = {
        RESULT_X_WINS: 0,
        RESULT_O_WINS: 0,
        RESULT_DRAW: 0
    }
    
    heatmapx = np.zeros(BOARD_SIZE**2,int)
    heatmapo = np.zeros(BOARD_SIZE**2,int)
    start_time = time.time()

    for g in range(total_games):
        end_of_game = (play_single_game(x_strategy, o_strategy))
        heatmapx = end_of_game.update_heatmapx(heatmapx)
        heatmapo = end_of_game.update_heatmapo(heatmapo)
        result = end_of_game.get_game_result()
        results[result] += 1
        
    elapsed_time = time.time()-start_time
    x_wins_percent = results[RESULT_X_WINS] / total_games * 100
    o_wins_percent = results[RESULT_O_WINS] / total_games * 100
    draw_percent = results[RESULT_DRAW] / total_games * 100
    avrg_moves = round((heatmapx.sum()+heatmapo.sum())/total_games,2)
    avrg_game_time = 1000*elapsed_time/total_games
    avrg_move_time = avrg_game_time/avrg_moves
    
    if print_stats:
    
        print("\n-----------\n")
        print(f"X wins: {x_wins_percent:.2f}%")
        print(f"O wins: {o_wins_percent:.2f}%")
        print(f"draws:  {draw_percent:.2f}%")
        print("Average moves played in game: ",avrg_moves)
        print("Average time of game: ",round(avrg_game_time,2),"ms")
        print("Average time of move: ",round(avrg_move_time,3),"ms")
    
        heatmapx_shape = heatmapx.reshape(BOARD_DIMENSIONS)
        plt.imshow(heatmapx_shape, cmap='hot', interpolation='nearest')
        plt.title("Heatmap X")
        plt.colorbar()
        plt.show()
        
        heatmapo_shape = heatmapo.reshape(BOARD_DIMENSIONS)
        plt.imshow(heatmapo_shape, cmap='hot', interpolation='nearest')
        plt.title("Heatmap O")
        plt.colorbar()
        plt.show()
    
    return  x_wins_percent, o_wins_percent, draw_percent, avrg_moves, avrg_game_time, avrg_move_time, heatmapx, heatmapo

def play_random_move(board):
    move = board.get_random_valid_move_index()
    return board.play_move(move)

def play_middle_move(board):
    move = board.get_middle_valid_move_index()
    return board.play_move(move)
    
def play_score_move(board):
    move = board.get_score_valid_move_index()
    return board.play_move(move)

def play_scoreV2_move(board):
    move = board.get_scoreV2_valid_move_index()
    return board.play_move(move)

def play_human_move(board):
    move = board.get_human_valid_move_index()
    return board.play_move(move)

def is_even(value):
    return value % 2 == 0


def is_empty(values):
    return values is None or len(values) == 0

def get_result(board):
    return board.get_game_result()


class Board:
    def __init__(self, board=None, last2move = None, lastmove = None, illegal_move = None, X_score = 0, O_score = 0):
        if board is None:
            self.board = np.copy(new_board)
        else:
            self.board = board

        self.illegal_move = illegal_move
        self.board_2d = self.board.reshape(BOARD_DIMENSIONS)        
        self.lastmove = lastmove        
        self.last2move = last2move
        self.X_score = X_score
        self.O_score = O_score
        self.result = 2

    def check_line2(self, board_2d, player, ii, jj, lastmove):
        c_i,c_j = lastmove
        def helpme(sign,i,j,x):
            while(board_2d[i][j]==player):
                x +=1
                i += sign*ii
                j += sign*jj
                if(j>=len(board_2d) or i>=len(board_2d) or j<0 or i<0):
                    break
            return x
        x = 0
        i,j = c_i,c_j
        x = helpme(1,i,j,x)    
        i,j = c_i,c_j
        x = helpme(-1,i,j,x)
        return x-1
    
    def win(self, board, player, lastmove = None):
        if(lastmove!=None):
            cnt_hor = self.check_line2(board,player,0,1,lastmove)
            cnt_ver = self.check_line2(board,player,1,0,lastmove)
            cnt_diag = self.check_line2(board,player,1,1,lastmove)
            cnt_diagrev = self.check_line2(board,player,1,-1,lastmove)
            if player == 1:
                self.X_score = max(cnt_hor, cnt_ver, cnt_diag, cnt_diagrev, self.X_score)
            if player == -1:
                self.O_score = max(cnt_hor, cnt_ver, cnt_diag, cnt_diagrev, self.O_score)   
                
            if(cnt_hor>=NEED_TO_WIN or cnt_ver>=NEED_TO_WIN or cnt_diag>=NEED_TO_WIN or cnt_diagrev>=NEED_TO_WIN):
                return True
            return False
    
    def get_move_score(self, board, player, lastmove):
        if(lastmove!=None):
            cnt_hor = self.check_line2(board,player,0,1,lastmove)
            cnt_ver = self.check_line2(board,player,1,0,lastmove)
            cnt_diag = self.check_line2(board,player,1,1,lastmove)
            cnt_diagrev = self.check_line2(board,player,1,-1,lastmove)
            return max(cnt_hor, cnt_ver, cnt_diag, cnt_diagrev)
        
    def get_game_score(self, player):
        if player == 1:
            return self.X_score
        if player == -1:
            return self.O_score            
    
    def get_game_result(self):
        if self.illegal_move is not None:
            return RESULT_O_WINS if self.get_turn() == CELL_X else RESULT_X_WINS
        
        if(self.lastmove!=None):
            if self.win(self.board_2d, self.board_2d[self.lastmove], self.lastmove):
                if (self.board_2d[self.lastmove] == CELL_X):
                    # self.print_board()
                    return RESULT_X_WINS
                if (self.board_2d[self.lastmove] == CELL_O):
                    # self.print_board()
                    return RESULT_O_WINS
            
        if CELL_EMPTY not in self.board_2d:
            # self.print_board()
            return RESULT_DRAW

        return RESULT_NOT_OVER
    
    def update_heatmap(self, heatmap):
        for i in range(BOARD_SIZE**2):
            if self.board[i]: heatmap[i] +=1
        return heatmap
    
    def update_heatmapx(self, heatmap):
        for i in range(BOARD_SIZE**2):
            if self.board[i] == CELL_X: heatmap[i] +=1
        return heatmap
    
    def update_heatmapo(self, heatmap):
        for i in range(BOARD_SIZE**2):
            if self.board[i] == CELL_O: heatmap[i] +=1
        return heatmap
    
    def is_gameover(self):
        return self.get_game_result() != RESULT_NOT_OVER

    def is_in_illegal_state(self):
        return self.illegal_move is not None

    def play_move(self, move_index):
        board_copy = np.copy(self.board)

        if move_index not in self.get_valid_move_indexes():
            return Board(board_copy, illegal_move=move_index)

        board_copy[move_index] = self.get_turn()
        move_2d =  int(move_index/BOARD_SIZE), move_index % BOARD_SIZE
        if self.lastmove != None:
            return Board(board_copy, last2move = self.lastmove, lastmove = move_2d, X_score = self.X_score, O_score = self.O_score)
        return Board(board_copy, lastmove = move_2d, X_score = self.X_score, O_score = self.O_score)

    def get_turn(self):
        non_zero = np.count_nonzero(self.board)
        return CELL_X if is_even(non_zero) else CELL_O

    def get_valid_move_indexes(self):
        return ([i for i in range(self.board.size)
                 if self.board[i] == CELL_EMPTY])

    def get_illegal_move_indexes(self):
        return ([i for i in range(self.board.size)
                if self.board[i] != CELL_EMPTY])

    def get_random_valid_move_index(self):
        return random.choice(self.get_valid_move_indexes())
    
    def make_rad(self, table):
        mid = (BOARD_SIZE-1)/2
        x = len(table)
        radius = np.array([0]*x,float)
        for i in range (x):
            radius[i] = (math.sqrt(((int(table[i]/BOARD_SIZE))-mid)**2+((table[i]%BOARD_SIZE)-mid)**2))
        return radius
    
    def get_middle_valid_move_index(self):
        valid_moves = self.get_valid_move_indexes()
        rad = self.make_rad(np.array(valid_moves))
        return valid_moves[np.argmin(rad)]
    
    def get_score_valid_move_index(self):
        if self.last2move == None:
            return self.get_middle_valid_move_index()
        player = self.board_2d[self.last2move]
        score_old = self.get_game_score(player)
        enemy_score = self.get_game_score(-player)
        if enemy_score > score_old and random.uniform(0, (enemy_score+1)**2)>1.5:
            player = -player
        valid_moves = self.get_valid_move_indexes()
        board_copy = np.copy(self.board)
        for i in range (len(valid_moves)):
            rad = self.make_rad(np.array(valid_moves))
            test_move_index = valid_moves[np.argmin(rad)]
            board_copy[test_move_index]=player
            test_move = int(test_move_index/BOARD_SIZE), test_move_index % BOARD_SIZE
            score_new = self.get_move_score(np.reshape(board_copy,BOARD_DIMENSIONS), player, test_move)
            if score_new > score_old and random.uniform(0, score_new**2)>1.0:
                return test_move_index
            else:
                valid_moves.remove(test_move_index)
                board_copy[test_move_index]=0
        return self.get_middle_valid_move_index()
    
    def get_scoreV2_valid_move_index(self):
        if self.last2move == None:
            return self.get_middle_valid_move_index()
        player = self.board_2d[self.last2move]
        score_old = self.get_game_score(player)
        enemy_score = self.get_game_score(-player)
        if enemy_score > score_old and random.uniform(0, enemy_score+1)>1.5:
            player = -player
        valid_moves = self.get_valid_move_indexes()
        board_copy = np.copy(self.board)
        score_best = 0
        best_moves = []
        for i in range (len(valid_moves)):
            rad = self.make_rad(np.array(valid_moves))
            test_move_index = valid_moves[np.argmin(rad)]
            board_copy[test_move_index]=player
            test_move = int(test_move_index/BOARD_SIZE), test_move_index % BOARD_SIZE
            score_test = self.get_move_score(np.reshape(board_copy,BOARD_DIMENSIONS), player, test_move)
            if score_test >= score_best:
                if score_test > score_best:
                    score_best = score_test
                    best_moves.clear()
                best_moves.append(test_move_index)
            valid_moves.remove(test_move_index)
            board_copy[test_move_index]=0
        return random.choice(best_moves)
    
    def get_human_valid_move_index(self):
        valid_moves_indexes = self.get_valid_move_indexes()
        valid_moves = []
        for i in range (len(valid_moves_indexes)):
            valid_moves.append((int(valid_moves_indexes[i]/BOARD_SIZE), valid_moves_indexes[i] % BOARD_SIZE))
        nums = []
        for i in range(BOARD_SIZE):
            nums.append(str(i))
        while True:
            x = input("\nPlease type x,y , for example 0,3 : ").split(',')
            if len(x) == 2:
                if x[0] in nums and x[1] in nums:
                    z = (int(x[0]),int(x[1]))
                    if z in valid_moves:
                        print ("Get in \n")
                        return z[0]*BOARD_SIZE+z[1]
                        break
                    else: 
                        print("This cell is not empty. Try again \n")
                else:
                    print("Type numbers in range of the board. Try again \n")
            else:
                print("Type exactly 2 naumbers separated by ','. Try again \n")

    def print_board(self):
        print(self.get_board_as_string())
        print("Last move was:",self.lastmove)
        
    def get_last_player_and_move(self):
        player = self.board_2d[self.lastmove]
        return (player, self.lastmove)

    def get_board_as_string(self):
        rows, cols = self.board_2d.shape
        board_as_string = "\n---------------\n"
        for c in range(cols):
            if c == 0:
                board_as_string += f"  {c} "
            elif c < BOARD_SIZE-1:
                board_as_string += f"{c} "
            else:
                board_as_string += f"{c} \n"
        for r in range(rows):
            for c in range(cols):
                move = get_symbol(self.board_2d[r, c])
                if c == 0:
                    board_as_string += f"{r}|{move}|"
                elif c < BOARD_SIZE-1:
                    board_as_string += f"{move}|"
                else:
                    board_as_string += f"{move}|\n"
        board_as_string += "---------------"

        return board_as_string


class BoardCache:
    def __init__(self):
        self.cache = {}

    def set_for_position(self, board, o):
        self.cache[board.board_2d.tobytes()] = o

    def get_for_position(self, board):
        board_2d = board.board_2d

        orientations = get_symmetrical_board_orientations(board_2d)

        for b, t in orientations:
            result = self.cache.get(b.tobytes())
            if result is not None:
                return (result, t), True

        return None, False

    def reset(self):
        self.cache = {}


def get_symmetrical_board_orientations(board_2d):
    return [(t.transform(board_2d), t) for t in TRANSFORMATIONS]


def get_rows_cols_and_diagonals(board_2d):
    rows_and_diagonal = get_rows_and_diagonal(board_2d)
    cols_and_antidiagonal = get_rows_and_diagonal(np.rot90(board_2d))
    return rows_and_diagonal + cols_and_antidiagonal


def get_rows_and_diagonal(board_2d):
    num_rows = board_2d.shape[0]
    return ([row for row in board_2d[range(num_rows), :]]
            + [board_2d.diagonal()])


def get_symbol(cell):
    if cell == CELL_X:
        return 'X'
    if cell == CELL_O:
        return 'O'
    return '-'


def is_draw(board):
    return board.get_game_result() == RESULT_DRAW

# games , wins = create_conv_dataset(200, play_random_move, play_random_move)

# games, wins = create_conv_dataset(100000, play_scoreV2_move, play_scoreV2_move)

# with open('conv_score2_v_score2_100k.npy', 'wb') as f:

#     np.save(f, games)
#     np.save(f, wins)

# with open('test.npy', 'rb') as f:

#     games_copy = np.load(f)
#     wins_copy = np.load(f)

# with open('test.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(games)


# # testowanie i metoda odczytu
# with open('test.csv', 'r') as file:
#     reader = csv.reader(file)
#     games_copy = list(reader)
    
# for i in range(len(games_copy)):
#     for j in range(len(games_copy[i])):
#         games_copy[i][j] = eval(games_copy[i][j])
        
# if games.all() == games_copy.all() and wins.all() == wins_copy.all():
#     print('swieto lasu')