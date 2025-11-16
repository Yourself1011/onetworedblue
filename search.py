from typing import Tuple, Callable
import chess
from chess import Board


def alpha_beta(
    board_instance: Board,
    max_depth: int,
    current_depth: int,
    alpha: float,
    beta: float,
    evaluate_fn: Callable[[Board], float]
) -> float:
    if current_depth == 0:
        return evaluate_fn(board_instance)
    
    if board_instance.is_checkmate():
        return -10000.0 if board_instance.turn == chess.WHITE else 10000.0
    
    if board_instance.is_stalemate() or board_instance.is_insufficient_material() or \
        board_instance.is_seventyfive_moves() or board_instance.is_fivefold_repetition():
        return 0.0
    
    if board_instance.turn == chess.WHITE:
        best_score = float('-inf')
        
        for legal_move in board_instance.generate_legal_moves():
            board_instance.push(legal_move)
            node_score = alpha_beta(
                board_instance, max_depth, current_depth - 1, alpha, beta, evaluate_fn
            )
            board_instance.pop()
            best_score = max(best_score, node_score)
            alpha = max(alpha, best_score)
            
            if beta <= alpha:
                return best_score
        
        return best_score
    else:
        best_score = float('inf')
        
        for legal_move in board_instance.generate_legal_moves():
            board_instance.push(legal_move)
            node_score = alpha_beta(
                board_instance, max_depth, current_depth - 1, alpha, beta, evaluate_fn
            )
            board_instance.pop()
            best_score = min(best_score, node_score)
            beta = min(beta, best_score)

            if beta <= alpha:
                return best_score
        
        return best_score


def alpha_beta_search(
    board: Board,
    max_depth: int,
    evaluate_fn: Callable[[Board], float]
) -> Tuple[chess.Move, float]:
    if max_depth == 0:
        return None, evaluate_fn(board)
    
    legal_moves = list(board.generate_legal_moves())
    if not legal_moves:
        return None, evaluate_fn(board)
    
    best_move = None
    best_score = float('-inf') if board.turn == chess.WHITE else float('inf')
    alpha = float('-inf')
    beta = float('inf')
    
    for move in legal_moves:
        board.push(move)
        
        eval_score = alpha_beta(
            board, max_depth, max_depth - 1, alpha, beta, evaluate_fn
        )
        
        board.pop()
        
        if board.turn == chess.WHITE:
            if eval_score > best_score:
                best_score = eval_score
                best_move = move
            alpha = max(alpha, eval_score)
        else:
            if eval_score < best_score:
                best_score = eval_score
                best_move = move
            beta = min(beta, eval_score)
        
        if (board.turn == chess.WHITE and best_score >= 10000.0) or \
           (board.turn == chess.BLACK and best_score <= -10000.0):
            break
    
    return best_move, best_score


def simple_evaluation(board: Board) -> float:
    if board.is_checkmate():
        return -10000.0 if board.turn == chess.WHITE else 10000.0
    
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0
    
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    
    score = 0.0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_values[piece.piece_type]
            if piece.color == chess.WHITE:
                score += value
            else:
                score -= value
    
    return score


print("=== Alpha-Beta Search ===")

board = Board("2r5/1B5Q/8/P5pN/1p1b3q/2r2P1P/K1Nk4/R7 w - - 2 2")

print(f"Position FEN: {board.fen()}")
print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")
print(f"Legal moves: {len(list(board.generate_legal_moves()))}\n")

best_move, best_score = alpha_beta_search(
    board, max_depth=4, evaluate_fn=simple_evaluation
)

print(f"Best move: {best_move}")
print(f"Best score: {best_score}")
