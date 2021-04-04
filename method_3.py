import numpy as np

def ChangeBoard(board, setrow, setcol, changeIndex, color):
    board[setrow, setcol]=color
    board[changeIndex]=color
    return board