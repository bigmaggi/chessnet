import chess
import chess.pgn
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
import os

# Set the device for GPU support
device = torch.device("mps")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Neural network definition
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        x = self.bn1(torch.relu(self.conv1(x)))
        x = self.bn2(torch.relu(self.conv2(x)))
        return torch.relu(x + residual)

class MultiHeadAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_channels = in_channels // num_heads
        assert self.head_channels * num_heads == in_channels, "in_channels must be divisible by num_heads."

        self.q_linear = nn.Linear(in_channels, in_channels)
        self.k_linear = nn.Linear(in_channels, in_channels)
        self.v_linear = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        # Shape of x: [batch_size, seq_len, in_channels]
        batch_size, seq_len, _ = x.size()

        # Transform queries, keys, values
        queries = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_channels).transpose(1, 2)
        keys = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_channels).transpose(1, 2)
        values = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_channels).transpose(1, 2)

        # Compute scaled dot-product attention
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / np.sqrt(self.head_channels)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attended_values = torch.matmul(attention_weights, values).transpose(1, 2).contiguous()
        attended_values = attended_values.view(batch_size, seq_len, self.in_channels)

        return attended_values

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(19, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.res1 = ResidualBlock(32, 64)
        self.res2 = ResidualBlock(64, 128)
        self.res3 = ResidualBlock(128, 256)
        self.res4 = ResidualBlock(256, 32)
        self.mha = MultiHeadAttention(32 * 8 * 8)
        self.fc_policy = nn.Linear(32 * 8 * 8, 64 * 2)
        self.fc_value = nn.Linear(32 * 8 * 8, 1)

    def forward(self, x, legal_moves=None):
        x = self.bn1(torch.relu(self.conv1(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.mha(x.unsqueeze(1)).squeeze(1)  # apply multi-head attention
        policy = torch.softmax(self.fc_policy(x), dim=1)
        if legal_moves is not None:
            policy = policy * legal_moves
        value = torch.tanh(self.fc_value(x))
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

def self_play(network, games, optimizer, epsilon=0.1, learning_rate=0.01):
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    network.eval()
    best_loss = float("inf")
    for i in tqdm(range(games), desc="Self-play progress"):
        board = chess.Board()
        game = chess.pgn.Game()
        game.headers["Event"] = "Self-play"
        node = game
        board_states = []
        moves_from = []
        moves_to = []
        game_results = []

        while not board.is_game_over():
            board_tensor = torch.from_numpy(board_to_tensor(board)).float().unsqueeze(0).to(device)
            legal_moves = torch.from_numpy(legal_moves_to_tensor(board)).float().unsqueeze(0).to(device)
            with torch.no_grad():
                policy, value = network(board_tensor, legal_moves)

            if random.random() < epsilon:
                move = random.choice(list(board.legal_moves))
            else:
                # Sort the moves by their probabilities
                sorted_moves_from = torch.argsort(policy[0, :64], descending=True)
                sorted_moves_to = torch.argsort(policy[0, 64:], descending=True)

                # Select the highest-probability legal move
                move = None
                for move_from in sorted_moves_from:
                    for move_to in sorted_moves_to:
                        potential_move = chess.Move(move_from.item(), move_to.item())
                        if potential_move in board.legal_moves:
                            move = potential_move
                            break
                    if move is not None:
                        break

                # If no legal move was found (which should not happen if the policy is properly normalized),
                # select a random legal move
                if move is None:
                    move = random.choice(list(board.legal_moves))

            node = node.add_variation(move)
            board_states.append(board_tensor.squeeze(0))
            moves_from.append(move.from_square)
            moves_to.append(move.to_square)
            game_results.append(result_to_value(board.result()))
            board.push(move)

        loss = train(network, board_states, moves_from, moves_to, game_results, optimizer)
        print(f'Loss after game {i+1}: {loss}')

        # Save the best model and its games
        if loss < best_loss:
            best_loss = loss
            save_model(network, 'best_model.pth')
            with open("best_game.pgn", "w") as f:
                print(game, file=f)



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
self_play(network, 100000, optimizer, learning_rate=0.001)

# save the model
save_model(network, 'chess_model.pth')

# Play against the network
play(network)

