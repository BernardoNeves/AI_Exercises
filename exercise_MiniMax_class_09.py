"""
    @file exercise_MiniMax_class_09.py
    @author Bernardo Neves (a23494@alunos.ipca.pt)
    @brief MiniMax Algorithm implementation
    @date 2023-11-10
"""
import numpy as np


def print_board(board: list) -> None:
    for row in board:
        print("\t", " | ".join(row))
        print("\t", "-" * 9)


def check_sequence(sequence) -> str or None:
    if sequence[0] != " " and len(set(sequence)) == 1:
        return sequence[0]
    return None


def check_winner(board: list) -> str or None:
    for i in range(len(board)):
        if (result := check_sequence("".join(board[i]))) is not None:
            return result
        if (result := check_sequence("".join(np.transpose(board)[i]))) is not None:
            return result
    if (result := check_sequence("".join(board[i][i] for i in range(len(board))))) is not None:
        return result
    if (result := check_sequence("".join(board[i][len(board) - i - 1] for i in range(len(board))))) is not None:
        return result
    return None


def get_score(board: list, player: str) -> int or None:
    winner = check_winner(board)
    score = None
    if winner:
        score = 1 if winner == player else -1
    elif " " not in board:
        score = 0
    return score


def minimax(board: list, player: str) -> int or None:
    max_score = -np.inf
    best_move = None

    for row in range(len(board)):
        for col in range(len(board[row])):
            if board[row][col] == " ":
                board[row][col] = player
                score = min_value(board, player)
                board[row][col] = " "

                if score > max_score:
                    max_score = score
                    best_move = (row, col)
    return best_move


def max_value(board: list, player: str) -> int:
    score = get_score(board, player)
    if score is not None:
        return score

    max_score = -np.inf
    for row in range(len(board)):
        for col in range(len(board[row])):
            if board[row][col] == " ":
                board[row][col] = player
                max_score = max(max_score, min_value(board, player))
                board[row][col] = " "
    return max_score


def min_value(board: list, player: str) -> int:
    score = get_score(board, player)
    if score is not None:
        return score

    min_score = np.inf
    for row in range(len(board)):
        for col in range(len(board[row])):
            if board[row][col] == " ":
                board[row][col] = "X" if player == "O" else "O"
                min_score = min(min_score, max_value(board, player))
                board[row][col] = " "
    return min_score


def test_all_first_moves() -> str:
    for r in range(3):
        for c in range(3):
            row, col = r, c
            board = np.full((3, 3), " ")
            player = "X"
            while True:
                board[row][col] = player
                print(f"\n\t{player} played ({row},{col})\n")
                print_board(board)
                winner = check_winner(board)
                if winner or " " not in board:
                    break

                player = "X" if player == "O" else "O"
                row, col = minimax(board, player) 
                
            if winner:
                return winner
            print("\n\tTied the Game\n")
            input("Press Enter to continue...")
    return None


def main():
    result = test_all_first_moves()
    if result is None:
        print("Success, all tests passed!")
    else:
        print(f"Failure, {result} Won the Game")

    
if __name__ == "__main__":
    main()
