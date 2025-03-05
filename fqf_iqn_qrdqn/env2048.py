import fqf_iqn_qrdqn.myconst as myconst  # Assuming myconst is in the same directory
import numpy as np
from collections import deque
import random
import torch
from gym import spaces  # Import spaces

class Env2048:
    def __init__(self):
        actions = [0, 1, 2, 3]
        self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
        self.training = True  # Consistent with model training mode
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(myconst.MAX_TILE + 1, myconst.SIZE, myconst.SIZE), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(self.actions))
        self.reset()


    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # if you're using CUDA
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU, if applicable
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def get_state(self):
        # One-hot encode the board state
        state = np.zeros((myconst.MAX_TILE + 1, myconst.SIZE, myconst.SIZE))
        for i in range(myconst.SIZE):
            for j in range(myconst.SIZE):
                tile = int(self.board[i, j])
                state[tile, i, j] = 1.0
        return torch.tensor(state, dtype=torch.float32)  # Keep device handling if needed


    def reset(self):
        # Original 2048 part
        self.board = np.zeros((myconst.SIZE, myconst.SIZE), dtype=int)
        self.add_new_tile()
        self.add_new_tile()
        self.score = 0
        self.steps = 0
        self.max_tile = 0
        self.gameover = False
        # New part (if you had _reset_buffer, keep it)
        # self._reset_buffer()  # Keep if it exists
        # Process and return "initial" state
        observation = self.get_state()
        return observation.cpu().numpy()  # Return NumPy array

    def add_new_tile(self):
        empty_cells = [(i, j) for i in range(myconst.SIZE) for j in range(myconst.SIZE) if self.board[i, j] == 0]
        if empty_cells:
            i, j = random.choice(empty_cells)
            self.board[i, j] = 1 if random.random() < 0.9 else 2

    def step(self, action): #0: left, 1: right, 2: up, 3: down
        moved = False
        reward = 0
        before_board = self.board.copy()
        if action == 0: #left
            moved, reward = self.move_left(self.board)
        elif action == 1: #right
            rotated_board = np.rot90(self.board, k=2)  # Rotate for easier processing
            moved, reward = self.move_left(rotated_board)
            if moved:
                self.board = np.rot90(rotated_board,k=-2)
        elif action == 2: #up
            rotated_board = np.rot90(self.board, k=1)  # Rotate for easier processing
            moved, reward = self.move_left(rotated_board)
            if moved:
                self.board = np.rot90(rotated_board,k=-1)
        elif action == 3: #down
            rotated_board = np.rot90(self.board, k=-1)  # Rotate for easier processing
            moved, reward = self.move_left(rotated_board)
            if moved:
                self.board = np.rot90(rotated_board,k=1)

        if not moved:
            reward = -myconst.PENALTY_FACTOR * myconst.REWARD_SCALING_FACTOR * (self.max_tile ** 2) # Penalty for invalid move
            self.board = np.zeros(self.board.shape) # I don't think you have to set to 0.
            self.gameover = True
        else:
            self.max_tile = max(self.max_tile, int(np.max(self.board)))
            # if the largest tile didn't move, add extra reward
            if np.argmax(self.board) == np.argmax(before_board):
                reward += myconst.EXTRA_FACTOR * myconst.REWARD_SCALING_FACTOR * (self.max_tile ** 2)
            if not (before_board == self.board).all():
                self.add_new_tile()

        self.score += reward
        self.steps += 1
        observation = self.get_state()

        return observation.cpu().numpy(), reward, self.gameover, {} # Return NumPy array and info dict

    def can_move(self):
        if np.any(self.board == 0):
            return True
        for i in range(myconst.SIZE):
            for j in range(myconst.SIZE):
                if j + 1 < myconst.SIZE and self.board[i,j] == self.board[i,j+1]:
                    return True
                if i + 1 < myconst.SIZE and self.board[i,j] == self.board[i+1,j]:
                    return True
        return False

    def move_left(self, board):
        moved = False
        reward = 0
        for i in range(myconst.SIZE):
            # Merge and move tiles to the left
            merged = [False] * myconst.SIZE #for avoid merging twice.
            for j in range(1, myconst.SIZE):
                if board[i, j] != 0:
                    k = j
                    while k > 0 and board[i, k - 1] == 0:
                        board[i, k - 1] = board[i, k]
                        board[i, k] = 0
                        k -= 1
                        moved = True
                    if k > 0 and board[i, k - 1] == board[i, k] and not merged[k-1]:
                        board[i, k - 1] += 1
                        reward += myconst.REWARD_SCALING_FACTOR * (board[i,k-1] ** 2) # REWARD CALCULATION
                        board[i, k] = 0
                        merged[k-1] = True
                        moved = True
        return moved, reward

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def action_space_func(self):
        return len(self.actions)

    def print_board(self):
        for i in range(myconst.SIZE):
            for j in range(myconst.SIZE):
                print(str(int(self.board[i][j])) + ' ', end='')
            print()
        print()

    def close(self):
        pass