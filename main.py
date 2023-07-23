import chess
import chess.pgn
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
import os
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MCTS_SIMULATIONS = 100

class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = self.bn1(torch.relu(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = torch.relu(out)
        return out

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv = nn.Conv2d(19, 256, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(256)
        self.resblocks = nn.Sequential(*(ResidualBlock(256) for _ in range(10)))
        self.fc_policy = nn.Linear(256 * 8 * 8, 64 * 64)
        self.fc_value = nn.Linear(256 * 8 * 8, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, legal_moves=None):
        x = self.bn(torch.relu(self.conv(x)))
        x = self.resblocks(x)
        x = self.dropout(x)
        policy = self.fc_policy(x.view(x.size(0), -1)).view(-1, 64 * 64)
        policy = torch.softmax(policy, dim=1)
        if legal_moves is not None:
            policy = policy * legal_moves
        value = torch.tanh(self.fc_value(x.view(x.size(0), -1)))
        return policy, value


class MCTS:
    def __init__(self, network):
        self.network = network
        self.children = {}  # Stores the child nodes
        self.v = 0  # Value estimate
        self.n = 0  # Visit count
        self.p = 0  # Prior probability from the network
        self.is_expanded = False

    def select(self):
        # Use UCT formula to select a child node
        best_value = -np.inf
        best_node = None
        for move, child in self.children.items():
            if child.n == 0:
                ucb_score = np.inf  # this will ensure that unvisited nodes are selected first
            else:
                ucb_score = (child.q / child.n) + np.sqrt(2 * np.log(self.n) / child.n) + child.p
            if ucb_score > best_value:
                best_value = ucb_score
                best_node = child
        return best_node

    def expand(self, priors):
        policy, value = self.get_policy_value(self.board)
        self.v = value
        self.is_expanded = True  # Move the "is_expanded" flag to here to mark the node as expanded

        for move in self.board.legal_moves:
            child_board = self.board.copy()
            child_board.push(move)
            self.children[move] = MCTSNode(parent=self)  # No need to pass move as it's already stored in the node
            self.children[move].p = policy[move.from_square * 64 + move.to_square]


    def backpropagate(self, value):
        self.q += value
        self.n += 1
        if self.parent:
            self.parent.backpropagate(-value)

    def get_policy_value(self, board):
        board_tensor = board_to_tensor(board).unsqueeze(0).to(device)
        legal_moves = legal_moves_to_tensor(board).to(device)
        with torch.no_grad():
            policy, value = self.network(board_tensor, legal_moves)
        return policy.view(-1).cpu().numpy(), value.item()

    def is_leaf_node(self):
        return len(self.children) == 0

    def is_game_over(self):
        return self.board.is_game_over()

    def simulate_single(self):
        board = self.board.copy()
        while not board.is_game_over():
            board_tensor = board_to_tensor(board).unsqueeze(0)
            legal_moves = legal_moves_to_tensor(board).unsqueeze(0).to(device)
            policy, value = self.network(board_tensor, legal_moves)

            # Choose the move with the highest visit count
            best_move = max(self.children.items(), key=lambda item: item[1].n)[0]
            self = self.children[best_move]
            board.push(best_move)
        return result_to_value(board.result())

    def simulate(self, num_simulations):
        self.v = 0
        self.n = 0
        with ThreadPoolExecutor(max_workers=24) as executor:  # Set max_workers to 24
            results = list(executor.map(lambda _: self.simulate_single(), range(num_simulations)))
        self.v = np.mean(results)
        self.n = num_simulations

class MCTSNode:
    def __init__(self, move=None, parent=None, network=None):
        self.parent = parent
        self.move = move
        self.children = {}
        self.n = 0  
        self.v = 0  
        self.p = 0  
        self.is_expanded = False
        self.board = parent.board.copy() if parent is not None else chess.Board() 
        self.network = network if parent is None else parent.network

    def expand(self):
        self.is_expanded = True
        policy, value = self.get_policy_value(self.board)
        self.v = value
        for move in self.board.legal_moves:
            self.children[move] = MCTSNode(move=move, parent=self)
            self.children[move].p = policy[move.from_square * 64 + move.to_square]

    def simulate(self):
        board = self.board.copy()
        board_tensor = board_to_tensor(board).unsqueeze(0)  # Add unsqueeze(0) here
        legal_moves = legal_moves_to_tensor(board).unsqueeze(0).to(device)
        policy, value = self.network(board_tensor, legal_moves)
        self.v = value
        return self.v


    def backpropagate(self, value):
        self.v += value
        self.n += 1
        if self.parent:
            self.parent.backpropagate(-value)

    def get_policy_value(self, board):
        board_tensor = board_to_tensor(board).unsqueeze(0).to(device)
        legal_moves = legal_moves_to_tensor(board).to(device)
        with torch.no_grad():
            policy, value = self.network(board_tensor, legal_moves)
        return policy.view(-1).cpu().numpy(), value.item()


    def select(self):
        best_value = -np.inf
        best_node = None
        for move, child in self.children.items():
            if child.n == 0:
                ucb_score = np.inf  
            else:
                ucb_score = (child.v / child.n) + np.sqrt(2 * np.log(self.n) / child.n) + child.p
            if ucb_score > best_value:
                best_value = ucb_score
                best_node = child
        return best_node


# Prepare the board state for the neural network
def board_to_tensor(board):
    tensor = np.zeros((19, 8, 8))
    pieces = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5, 'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}
    for i in range(64):
        piece = board.piece_at(i)
        if piece is not None:
            tensor[pieces[str(piece)], i // 8, i % 8] = 1
    if board.turn:
        tensor[12, :, :] = 1
    tensor[13, :, :] = int(board.has_kingside_castling_rights(chess.WHITE))
    tensor[14, :, :] = int(board.has_queenside_castling_rights(chess.WHITE))
    tensor[15, :, :] = int(board.has_kingside_castling_rights(chess.BLACK))
    tensor[16, :, :] = int(board.has_queenside_castling_rights(chess.BLACK))
    if board.ep_square is not None:
        tensor[17, board.ep_square // 8, board.ep_square % 8] = 1
    tensor[18, :, :] = board.halfmove_clock / 50
    return torch.tensor(tensor).float().to(device)


# Convert the legal moves to a binary mask
def legal_moves_to_tensor(board):
    legal_moves = np.zeros(64 * 64)
    for move in board.legal_moves:
        legal_moves[move.from_square * 64 + move.to_square] = 1  
    return torch.tensor(legal_moves).float().to(device)


def train(network, board_states, moves_from, moves_to, game_results, optimizer, lr_scheduler, early_stopping_epochs):
    network.train()
    optimizer.zero_grad()

    board_states = torch.stack(board_states).to(device)
    moves_from = torch.tensor(moves_from, dtype=torch.long).to(device)
    moves_to = torch.tensor(moves_to, dtype=torch.long).to(device)
    game_results = torch.tensor(game_results, dtype=torch.float32).to(device)

    predicted_policy, predicted_value = network(board_states, None)

    # Compute the policy loss
    policy_loss_from = -torch.log(predicted_policy[:, :64][range(len(moves_from)), moves_from])
    policy_loss_to = -torch.log(predicted_policy[:, 64:][range(len(moves_to)), moves_to])
    policy_loss = torch.mean(policy_loss_from + policy_loss_to)

    # Compute the value loss
    value_loss = (predicted_value.squeeze() - game_results) ** 2
    value_loss = torch.mean(value_loss)

    # Compute the total loss
    loss = policy_loss + value_loss
    loss.backward()

    # Clip the gradients to avoid explosion
    torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)

    optimizer.step()

    # Update the learning rate
    lr_scheduler.step()

    return loss.item()

def self_play(network, games, optimizer, lr_scheduler, epsilon=0.1, early_stopping_epochs=5):
    network.eval()
    best_loss = float("inf")
    epochs_without_improvement = 0
    for i in tqdm(range(games), desc="Self-play progress"):
        game = chess.pgn.Game()
        game.headers["Event"] = "Self-play"
        board = chess.Board()  # Initialize the board object
        root = MCTSNode(network=network)  # Initializing the root node with no parent.
        root.expand()  # No need to pass any arguments
        board_states = []
        moves_from = []
        moves_to = []
        game_results = []

        while not board.is_game_over():
            board_tensor = board_to_tensor(board).unsqueeze(0).to(device)
            legal_moves = legal_moves_to_tensor(board).unsqueeze(0).to(device)
            with torch.no_grad():
                policy, value = network(board_tensor, legal_moves)

            for _ in range(MCTS_SIMULATIONS):
                child = root.select()
                if child is not None and child.move is not None:
                    child.expand()  # No need to pass the prior for the selected move

                value = child.simulate()  # No need to pass the board since it's stored in the child node
                child.backpropagate(value)

            # Consider only legal moves
            legal_moves = set(board.legal_moves)
            mcts_moves = set(root.children.keys())
            valid_moves = legal_moves & mcts_moves

            if not valid_moves:  # If no valid moves, game is over
                break

            move = max(valid_moves, key=lambda move: root.children[move].n)

            root = root.children[move]
            board_states.append(board_tensor.squeeze(0))
            moves_from.append(move.from_square)
            moves_to.append(move.to_square)
            game_results.append(result_to_value(board.result()))
            board.push(move)

            loss = train(network, board_states, moves_from, moves_to, game_results, optimizer, lr_scheduler, early_stopping_epochs)
            print(f'Loss after game {i+1}: {loss}')

            # Save the best model and its games
            if loss < best_loss:
                best_loss = loss
                epochs_without_improvement = 0
                save_model(network, 'best_model.pth')
                with open(f"best_game_{i+1}.pgn", "w") as f:
                    print(game, file=f)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_epochs:
                    print(f'Early stopping after {early_stopping_epochs} epochs without loss improvement.')
                    break

            lr_scheduler.step()


# Convert game result to numerical value
def result_to_value(result):
    if result == '1-0':
        return 1
    elif result == '0-1':
        return -1
    else:
        return 0

def play(network):
    board = chess.Board()
    while not board.is_game_over():
        print(board)
        if board.turn:  # Human player's turn (playing as white)
            move = chess.Move.from_uci(input('Enter your move: '))
            if move in board.legal_moves:
                board.push(move)
            else:
                print("Illegal move. Try again.")
        else:  # AI's turn
            board_tensor = torch.from_numpy(board_to_tensor(board)).float().unsqueeze(0).to(device)
            legal_moves = torch.from_numpy(legal_moves_to_tensor(board)).float().unsqueeze(0).to(device)
            policy, _ = network(board_tensor, legal_moves)
            move_from = torch.argmax(policy[0, :64]).item()
            move_to = torch.argmax(policy[0, 64:]).item()
            move = chess.Move(move_from, move_to)
            board.push(move)
    print(board.result())

def save_model(network, file_path='model.pth'):
    torch.save(network.state_dict(), file_path)

def load_model(network, file_path='model.pth'):
    network.load_state_dict(torch.load(file_path))

# Initialize the network and optimizer
network = ChessNet().to(device)
if os.path.isfile('chess_model.pth'):
    load_model(network, 'chess_model.pth')
    print("Loaded model from disk")

# Using Adam optimizer with a lower learning rate and gradient clipping
optimizer = optim.Adam(network.parameters(), lr=0.00001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

# Self-play phase with 1000 games
self_play(network, 20000, optimizer, scheduler, early_stopping_epochs=10)

# Save the model
save_model(network, 'chess_model.pth')

# Play against the network
play(network)
