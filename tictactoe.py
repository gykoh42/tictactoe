import numpy as np
import random
from collections import Counter

class TicTacToeOptimizedFinal:
    def __init__(self, epsilon = 0.1, n_step = 3):
        self.board = np.zeros((3, 3))
        self.state_values = {}
        self.epsilon = epsilon
        self.n_step = n_step

    def reset_board(self):
        self.board = np.zeros((3, 3))

    def choose_action(self, player):
        available_actions = [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(available_actions)
        else:
            best_value = -float('inf')
            best_action = None
            for action in available_actions:
                next_board = self.board.copy()
                next_board[action] = player
                state_key = self.get_state_key(next_board)
                state_value = self.state_values.get(state_key, 0)
                if state_value > best_value:
                    best_value = state_value
                    best_action = action
            return best_action if best_action else random.choice(available_actions)

    def update_board(self, position, player):
        self.board[position] = player

    def is_winner(self, player):
        for i in range(3):
            if np.all(self.board[i, :] == player) or np.all(self.board[:, i] == player):
                return True
        if np.all(np.diag(self.board) == player) or np.all(np.diag(np.fliplr(self.board)) == player):
                return True
        return False

    def is_draw(self):
        return np.all(self.board != 0) and not self.is_winner(1) and not self.is_winner(-1)

    def get_reward(self, player, action):
        reward = 0
        if self.is_winner(player):
            reward = 1
        elif self.is_winner(-player):
            reward = -1

        if self.board.sum() == player and action == (1, 1):
            reward += 0.5

        corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
        if action in corners:
            reward += 0.3

        additional_reward = self.evaluate_double_threat_and_blocking(player, action)
        reward += additional_reward

        return reward

    def creates_double_threat(self, board, player):
        threat_count = 0

        for i in range(3):
            if np.sum(board[i, :]) == 2 * player and 0 in board[i, :]:
                threat_count += 1
            if np.sum(board[:, i]) == 2 * player and 0 in board[:, i]:
                threat_count += 1

        if np.sum(np.diag(board)) == 2 * player and 0 in np.diag(board):
            threat_count += 1
        if np.sum(np.diag(np.fliplr(board))) == 2 * player and 0 in np.diag(np.fliplr(board)):
            threat_count += 1

        return threat_count >= 2

    def blocks_opponent_win(self, temp_board, original_board, opponent):
        for i in range(3):

            if np.sum(original_board[i, :]) == 2 * opponent and temp_board[i, :].sum() != 2 * opponent:
                return True
            if np.sum(original_board[:, i]) == 2 * opponent and temp_board[:, i].sum() != 2 * opponent:
                return True

        if np.sum(np.diag(original_board)) == 2 * opponent and np.sum(np.diag(temp_board)) != 2 * opponent:
            return True
        if np.sum(np.diag(np.fliplr(original_board))) == 2 * opponent and np.sum(np.diag(np.fliplr(temp_board))) != 2 * opponent:
            return True

        return False

    def evaluate_double_threat_and_blocking(self, player, action):
        additional_reward = 0
        temp_board = self.board.copy()
        temp_board[action] = player

        if self.creates_double_threat(temp_board, player):
            additional_reward += 0.3
        if self.blocks_opponent_win(temp_board, self.board, -player):
            additional_reward += 0.2

        return additional_reward

    def play_game(self):
        self.reset_board()
        players = [1, -1]
        state_history = []
        while True:
            for player in players:
                action = self.choose_action(player)
                state_history.append((self.get_state_key(self.board), action))
                self.update_board(action, player)
                if self.is_winner(player):
                    return player, state_history
                if self.is_draw():
                    return 0, state_history

    def update_state_values(self, winner, state_history):
        reward = self.get_reward(winner, state_history[-1][1])
        for state_key, action in reversed(state_history):
            if state_key not in self.state_values:
                self.state_values[state_key] = 0
            self.state_values[state_key] += reward
            reward *= 0.9

    def get_state_key(self, board):
        rotations = [np.rot90(board, k=i) for i in range(4)]
        flips = [np.fliplr(rot) for rot in rotations]
        unique_states = rotations + flips
        sorted_states = sorted([tuple(state.flatten()) for state in unique_states])
        return tuple(sorted_states[0])

    def train(self, iterations=1000):
        winners = []
        draws = 0
        for _ in range(iterations):
            winner, state_history = self.play_game()
            if winner == 0:
                draws += 1
            else:
                winners.append(winner)
            self.update_state_values(winner, state_history)
            self.print_board()

        print("Draws:", draws)
        return winners

    def print_board(self):
        print("-" * 16)
        for row in self.board:
            print(" | ".join("{:^3}".format(int(cell)) for cell in row))
            print("-" * 16)
        print()

    def train(self, iterations=1000):
        winners = []
        for _ in range(iterations):
            winner, state_history = self.play_game()
            self.update_state_values(winner, state_history)
            winners.append(winner)
            self.print_board()
        return winners

    def print_state_values(self):
        for state, value in self.state_values.items():
            print(f"State: {state}, Value: {value}")

    def count_winners(self, winners):
        winners_counter = Counter(winners)
        print("Winners (-1):", winners_counter[-1])
        print("Winners (1):", winners_counter[1])

game = TicTacToeOptimizedFinal(epsilon=0.1, n_step=3)
winners = game.train(iterations=10000)

print("Winners:", winners)
print()
game.count_winners(winners)
print()
game.print_state_values()