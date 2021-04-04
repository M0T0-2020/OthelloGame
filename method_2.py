import numpy as np

def searchActionUp(board, idx, color):
    idx_row = idx[0]
    idx_col = idx[1]
    if idx_row==0:
        return False
    self_color = int(color%2 + 1)
    changePos_row = []
    changePos_col = []
    for idx in range(idx_row-1,-1, -1):
        x = board[idx, idx_col]
        if x==0:
            return False
        if x==color:
            changePos_row.append(idx)
            changePos_col.append(idx_col)
        if x==self_color:
            if len(changePos_row)==0:
                return False
            return (changePos_row, changePos_col)

def searchActionDown(board, idx, color):
    idx_row = idx[0]
    idx_col = idx[1]
    if idx_row==7:
        return False
    self_color = int(color%2 + 1)
    changePos_row = []
    changePos_col = []
    for idx in range(idx_row+1,8,1):
        x = board[idx, idx_col]
        if x==0:
            return False
        if x==color:
            changePos_row.append(idx)
            changePos_col.append(idx_col)
        if x==self_color:
            if len(changePos_row)==0:
                return False
            return (changePos_row, changePos_col)

def searchActionRight(board, idx, color):
    idx_row = idx[0]
    idx_col = idx[1]
    if idx_col==7:
        return False
    self_color = int(color%2 + 1)
    changePos_row = []
    changePos_col = []
    for idx in range(idx_col+1,8,1):
        x = board[idx_row, idx]
        if x==0:
            return False
        if x==color:
            changePos_row.append(idx_row)
            changePos_col.append(idx)
        if x==self_color:
            if len(changePos_row)==0:
                return False
            return (changePos_row, changePos_col)

def searchActionLeft(board, idx, color):
    idx_row = idx[0]
    idx_col = idx[1]
    if idx_col==0:
        return False
    self_color = 2 if color==1 else 1
    changePos_row = []
    changePos_col = []
    for idx in range(idx_col-1,-1, -1):
        x = board[idx_row, idx]
        if x==0:
            return False
        if x==color:
            changePos_row.append(idx_row)
            changePos_col.append(idx)
        if x==self_color:
            if len(changePos_row)==0:
                return False
            return (changePos_row, changePos_col)

def searchActionUpRight(board, idx, color):
    idx_row = idx[0]
    idx_col = idx[1]
    if idx_row==0 or idx_col==7:
        return False
    self_color = int(color%2 + 1)
    changePos_row = []
    changePos_col = []
    rank = min(idx_row, 7-idx_col)
    for i in range(1,rank+1,1):
        idx_r = idx_row - i
        idx_c = idx_col + i
        x = board[idx_r, idx_c]
        if x==0:
            return False
        if x==color:
            changePos_row.append(idx_r)
            changePos_col.append(idx_c)
        if x==self_color:
            if len(changePos_row)==0:
                return False
            return (changePos_row, changePos_col)

def searchActionUpLeft(board, idx, color):
    idx_row = idx[0]
    idx_col = idx[1]
    if idx_row==0 or idx_col==0:
        return False
    self_color = int(color%2 + 1)
    changePos_row = []
    changePos_col = []
    rank = min(idx_row, idx_col)
    for i in range(1,rank+1,1):
        idx_r = idx_row - i
        idx_c = idx_col - i
        x = board[idx_r, idx_c]
        if x==0:
            return False
        if x==color:
            changePos_row.append(idx_r)
            changePos_col.append(idx_c)
        if x==self_color:
            if len(changePos_row)==0:
                return False
            return (changePos_row, changePos_col)

def searchActionDownRight(board, idx, color):
    idx_row = idx[0]
    idx_col = idx[1]
    if idx_row==7 or idx_col==7:
        return False
    self_color = int(color%2 + 1)
    changePos_row = []
    changePos_col = []
    rank = min(7-idx_row, 7-idx_col)
    for i in range(1,rank+1,1):
        idx_r = idx_row + i
        idx_c = idx_col + i
        x = board[idx_r, idx_c]
        if x==0:
            return False
        if x==color:
            changePos_row.append(idx_r)
            changePos_col.append(idx_c)
        if x==self_color:
            if len(changePos_row)==0:
                return False
            return (changePos_row, changePos_col)

def searchActionDownLeft(board, idx, color):
    idx_row = idx[0]
    idx_col = idx[1]
    if idx_row==7 or idx_col==0:
        return False
    self_color = int(color%2 + 1)
    changePos_row = []
    changePos_col = []
    rank = min(7-idx_row, idx_col)
    for i in range(1,rank+1,1):
        idx_r = idx_row + i
        idx_c = idx_col - i
        x = board[idx_r, idx_c]
        if x==0:
            return False
        if x==color:
            changePos_row.append(idx_r)
            changePos_col.append(idx_c)
        if x==self_color:
            if len(changePos_row)==0:
                return False
            return (changePos_row, changePos_col)    
            
def SearchAction(board, available_position, color):
    Position_Row=[]
    Position_Col=[]
    Change_Position = []
    color = int(color%2 + 1)
    for idx in np.vstack(np.where(available_position==1)).T:
        Change_Position_row = []
        Change_Position_col = []
        flag=False
        for func in [searchActionUp, searchActionDown, searchActionRight, searchActionLeft, searchActionUpRight, searchActionUpLeft, searchActionDownRight, searchActionDownLeft]:  
            a = func(board, idx, color)
            if a:
                flag=True
                Change_Position_row+=a[0]
                Change_Position_col+=a[1]
        if flag:
            Position_Row.append(idx[0])
            Position_Col.append(idx[1])
            Change_Position.append((Change_Position_row, Change_Position_col))
    return Position_Row, Position_Col, Change_Position