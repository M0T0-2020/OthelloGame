import numpy as np

#右
def searchRight(board, available_board):
    #右端の列を0
    board[:,7]=0
    #右移動
    available_board_right = available_board*np.roll(board, shift=1,  axis=1)
    return available_board_right
#左
def searchLeft(board, available_board):
    #左端の列を0
    board[:,0]=0
    #左移動
    available_board_left = available_board*np.roll(board, shift=-1,  axis=1)
    return available_board_left
#上
def searchUp(board, available_board):
    #上端の行を0
    board[0,:]=0
    #上移動
    available_board_up = available_board*np.roll(board, shift=-1,  axis=0)
    return available_board_up
#下
def searchDown(board, available_board):
    #下端の行を0
    board[7,:]=0
    #下移動
    available_board_down = available_board*np.roll(board, shift=1,  axis=0)
    return available_board_down

#斜め上右
def searchUpRight(board, available_board):
    #上端の行を0
    board[0,:]=0
    #上移動
    board = np.roll(board, shift=-1,  axis=0)

    #右端の列を0
    board[:,7]=0
    #右移動
    board = np.roll(board, shift=1,  axis=1)

    available_board_upright = available_board*board
    return available_board_upright

#斜め上左
def searchUpLeft(board, available_board):
    #上端の行を0
    board[0,:]=0
    #上移動
    board = np.roll(board, shift=-1,  axis=0)

    #左端の列を0
    board[:,0]=0
    #左移動
    board = np.roll(board, shift=-1,  axis=1)
    

    available_board_upleft = available_board*board
    return available_board_upleft
#斜め下右
def searchDownRight(board, available_board):
    #下端の行を0
    board[7,:]=0
    #下移動
    board = np.roll(board, shift=1,  axis=0)

    #右端の列を0
    board[:,7]=0
    #左移動
    board = np.roll(board, shift=1,  axis=1)
    
    
    available_board_downright = available_board*board
    return available_board_downright
#斜め下左
def searchDownLeft(board, available_board):
    #下端の行を0
    board[7,:]=0
    #下移動
    board = np.roll(board, shift=1,  axis=0)

    #左端の列を0
    board[:,0]=0
    #左移動
    board = np.roll(board, shift=-1,  axis=1)
    
    
    available_board_downleft = available_board*board
    return available_board_downleft
    
def SearchAvailablePosition(board, color):
    available_board = board.copy()
    available_board[available_board==0]=10
    available_board[available_board<10]=0
    #if black color to white、vice versa
    color = int(color%2 + 1)
    board_2 = board.copy()
    board_2[board_2!=color]=0
    board_2[board_2==color]=1
    
    return_board = np.zeros(board.shape)
    
    for func in [searchRight, searchLeft, searchUp, searchDown,searchUpRight, searchUpLeft, searchDownRight, searchDownLeft]:
        return_board+=func(board_2.copy(), available_board)
    return_board[return_board!=0]=1
    return return_board.astype(int)