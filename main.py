import chess
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
import os

# Set the device for GPU support
device = torch.device("mps")

# Neural network definition
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(19, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.fc_policy = nn.Linear(32 * 8 * 8, 64 * 2)
        self.fc_value = nn.Linear(32 * 8 * 8, 1)

    def forward(self, x, legal_moves=None):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        policy = self.fc_policy(x.view(x.size(0), -1)).view(-1, 64 * 2)
        policy = torch.softmax(self.fc_policy(x.view(x.size(0), -1)).view(-1, 64 * 2), dim=1)
        if legal_moves is not None:
            policy = policy * legal_moves
        value = torch.tanh(self.fc_value(x.view(x.size(0), -1)))
        return policy, value


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
    return tensor

# Convert the legal moves to a binary mask
def legal_moves_to_tensor(board):
    legal_moves = np.zeros(64 * 2)
    for move in board.legal_moves:
        legal_moves[move.from_square] = 1
        legal_moves[move.to_square + 64] = 1
    return legal_moves

# Self-play function for generating training data
def self_play(network, games, optimizer, epsilon=0.1):
    for _ in tqdm(range(games), desc="Self-play progress"):  # Add a progress bar here
        board = chess.Board()
        board_states = []
        moves_from = []
        moves_to = []
        game_results = []

        while not board.is_game_over():
            board_tensor = torch.from_numpy(board_to_tensor(board)).float().unsqueeze(0).to(device)
            legal_moves = torch.from_numpy(legal_moves_to_tensor(board)).float().unsqueeze(0).to(device)
            policy, value = network(board_tensor, legal_moves)

            if random.random() < epsilon:
                move = random.choice(list(board.legal_moves))
            else:
                move = None
                while move is None or move not in board.legal_moves:
                    move_from = torch.multinomial(policy[0, :64] + 1e-8, 1).item()
                    move_to = torch.multinomial(policy[0, 64:] + 1e-8, 1).item()
                    move = chess.Move(move_from, move_to)

            board_states.append(board_tensor.squeeze(0))
            moves_from.append(move.from_square)
            moves_to.append(move.to_square)
            game_results.append(result_to_value(board.result()))

            board.push(move)

        loss = train(network, board_states, moves_from, moves_to, game_results, optimizer)
        print(f'Loss: {loss}')


# Training function
def train(network, board_states, moves_from, moves_to, game_results, optimizer):
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

    optimizer.step()

    return loss.item()


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

optimizer = optim.Adam(network.parameters())

# Self-play phase
self_play(network, 100, optimizer)

# save the model
save_model(network, 'chess_model.pth')

# Play against the network
play(network)

