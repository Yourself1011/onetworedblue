from pathlib import Path
from typing import Tuple, Callable
import chess
from chess import Board
import torch
from time import time

from src.nnue import feedforwardIntermediate, load
from src.parseevaluations import fen_to_tensor

# from nnue import feedforwardIntermediate, load
# from parseevaluations import fen_to_tensor


# skips = 0
# totalSteps = 0
# nnCalls = 0
# nnTime = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 1000,
}


def alpha_beta(
    board_instance: Board,
    max_depth: int,
    current_depth: int,
    alpha: float,
    beta: float,
    evaluate_fn: Callable[[Board], float],
    startTime,
    timeLimit,
) -> float:
    # global skips, totalSteps, nnCalls, nnTime
    # totalSteps += 1
    if current_depth == 0:
        # start = time()
        # nnCalls += 1
        result = quiescence_search(board_instance, alpha, beta, evaluate_fn)
        # nnTime += time() - start
        return result

    if board_instance.is_checkmate():
        return -10000.0 if board_instance.turn == chess.WHITE else 10000.0

    if (
        board_instance.is_stalemate()
        or board_instance.is_insufficient_material()
        or board_instance.is_seventyfive_moves()
        or board_instance.is_fivefold_repetition()
    ):
        return 0.0

    if time() - startTime > timeLimit:
        return 0

    legal_moves = list(board_instance.generate_legal_moves())
    sortedLegal = [chess.Move.null()] * len(legal_moves)
    idx = 0
    for i in range(len(legal_moves)):
        if legal_moves[i] and board_instance.is_capture(legal_moves[i]):
            attacker = board_instance.piece_at(legal_moves[i].from_square)
            defender = board_instance.piece_at(legal_moves[i].to_square)
            if (
                attacker
                and defender
                and PIECE_VALUES[defender.piece_type] * 10
                >= PIECE_VALUES[attacker.piece_type]
            ):
                sortedLegal[idx] = legal_moves[i]
                legal_moves[i] = chess.Move.null()
                idx += 1

    for i in range(len(legal_moves)):
        if legal_moves[i]:
            sortedLegal[idx] = legal_moves[i]
            idx += 1

    legal_moves = sortedLegal

    if board_instance.turn == chess.WHITE:
        best_score = float("-inf")

        for legal_move in legal_moves:
            board_instance.push(legal_move)
            node_score = alpha_beta(
                board_instance,
                max_depth,
                current_depth - 1,
                alpha,
                beta,
                evaluate_fn,
                startTime,
                timeLimit,
            )
            board_instance.pop()
            best_score = max(best_score, node_score)
            alpha = max(alpha, best_score)

            if beta <= alpha:
                # skips += 1
                return best_score

        return best_score
    else:
        best_score = float("inf")

        for legal_move in legal_moves:
            board_instance.push(legal_move)
            node_score = alpha_beta(
                board_instance,
                max_depth,
                current_depth - 1,
                alpha,
                beta,
                evaluate_fn,
                startTime,
                timeLimit,
            )
            board_instance.pop()
            best_score = min(best_score, node_score)
            beta = min(beta, best_score)

            if beta <= alpha:
                # skips += 1
                return best_score

        return best_score


def alpha_beta_search(
    board: Board,
    max_depth: int,
    evaluate_fn: Callable[[Board], float],
    previousBest: chess.Move | None,
    startTime,
    timeLimit,
) -> Tuple[chess.Move, float]:
    if max_depth == 0:
        return None, evaluate_fn(board)

    legal_moves = list(board.generate_legal_moves())
    if not legal_moves:
        return None, evaluate_fn(board)

    best_move = None
    best_score = float("-inf") if board.turn == chess.WHITE else float("inf")
    alpha = float("-inf")
    beta = float("inf")

    idx = 0
    sortedLegal = [chess.Move.null()] * len(legal_moves)
    if previousBest:
        for i in range(len(legal_moves)):
            if previousBest.uci() == legal_moves[i].uci():
                sortedLegal[idx] = legal_moves[i]
                legal_moves[i] = chess.Move.null()
                idx += 1
                # legal_moves[0], legal_moves[i] = legal_moves[i], legal_moves[0]
                break

    for i in range(len(legal_moves)):
        if legal_moves[i] and board.is_capture(legal_moves[i]):
            attacker = board.piece_at(legal_moves[i].from_square)
            defender = board.piece_at(legal_moves[i].to_square)
            if (
                attacker
                and defender
                and PIECE_VALUES[defender.piece_type] * 10
                >= PIECE_VALUES[attacker.piece_type]
            ):
                sortedLegal[idx] = legal_moves[i]
                legal_moves[i] = chess.Move.null()
                idx += 1

    for i in range(len(legal_moves)):
        if legal_moves[i]:
            sortedLegal[idx] = legal_moves[i]
            idx += 1

    legal_moves = sortedLegal

    for move in legal_moves:
        board.push(move)

        eval_score = alpha_beta(
            board,
            max_depth,
            max_depth - 1,
            alpha,
            beta,
            evaluate_fn,
            startTime,
            timeLimit,
        )
        if time() - startTime > timeLimit:
            return None, 0

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

        if (board.turn == chess.WHITE and best_score >= 10000.0) or (
            board.turn == chess.BLACK and best_score <= -10000.0
        ):
            break

    # print(skips, nnCalls, nnTime, totalSteps)
    return best_move, best_score


def iterativeDeepening(
    board: Board,
    max_depth: int,
    evaluate_fn: Callable[[Board], float],
    timeLimit: float,
) -> Tuple[chess.Move, float]:
    bestMove = None
    bestScore = 0

    start = time()
    for i in range(1, max_depth + 1):
        move, score = alpha_beta_search(
            board, i, evaluate_fn, bestMove, start, timeLimit
        )

        if time() - start > timeLimit:
            print("depth", i, "t", time() - start)
            break

        bestMove = move
        bestScore = score

    return bestMove, bestScore


def quiescence_search(
    board: Board, alpha: float, beta: float, evaluate_fn: Callable[[Board], float]
) -> float:
    stand_pat = evaluate_fn(board)

    if board.turn == chess.WHITE:
        if stand_pat >= beta:
            return stand_pat
        if stand_pat > alpha:
            alpha = stand_pat

        best_value = stand_pat

        for move in board.generate_legal_moves():
            if board.is_capture(move):
                board.push(move)
                score_after_capture = quiescence_search(board, alpha, beta, evaluate_fn)
                board.pop()

                if score_after_capture >= beta:
                    return score_after_capture
                if score_after_capture > best_value:
                    best_value = score_after_capture
                if score_after_capture > alpha:
                    alpha = score_after_capture

        return best_value
    else:
        if stand_pat <= alpha:
            return stand_pat
        if stand_pat < beta:
            beta = stand_pat

        best_value = stand_pat

        for move in board.generate_legal_moves():
            if board.is_capture(move):
                board.push(move)
                score_after_capture = quiescence_search(board, alpha, beta, evaluate_fn)
                board.pop()

                if score_after_capture <= alpha:
                    return score_after_capture
                if score_after_capture < best_value:
                    best_value = score_after_capture
                if score_after_capture < beta:
                    beta = score_after_capture

        return best_value


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
        chess.KING: 0,
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


w1, b1, w2, b2, _ = load(path=Path("./src/weights/save.pth"))


lastAccumulator = None
lastInput = None


def nnueEvaluation(board):
    global lastAccumulator, lastInput

    if lastAccumulator is None or lastInput is None:
        lastInput = fen_to_tensor(board.fen(), device=device).reshape(1, 12, 8, 8)
        lastAccumulator, evaluation = feedforwardIntermediate(
            lastInput,
            w1,
            b1,
            w2,
            b2,
        )
        return evaluation.item()

    boardInput = fen_to_tensor(board.fen(), device=device).reshape(1, 12, 8, 8)
    diff = boardInput - lastInput
    lastAccumulator += torch.concat(
        (
            torch.reshape(diff, (1, -1)) @ w1,
            torch.reshape(torch.flip(diff, [2, 3]), (1, -1)) @ w1,
        ),
        dim=1,
    )

    evaluation = (
        torch.clamp(
            lastAccumulator,
            0,
            1,
        )
        ** 2
        @ w2
        + b2
    )

    lastInput = boardInput
    return evaluation.item()


if __name__ == "__main__":
    print("=== Alpha-Beta Search ===")

    board = Board("2r5/1B5Q/8/P5pN/1p1b3q/2r2P1P/K1Nk4/R7 w - - 2 2")

    print(f"Position FEN: {board.fen()}")
    print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")
    print(f"Legal moves: {len(list(board.generate_legal_moves()))}\n")

    best_move, best_score = iterativeDeepening(
        board, max_depth=4, evaluate_fn=nnueEvaluation, timeLimit=1
    )

    print(f"Best move: {best_move}")
    print(f"Best score: {best_score}")
