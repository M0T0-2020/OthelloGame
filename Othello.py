import numpy as np
from method_1 import SearchAvailablePosition
from method_2 import SearchAction
from method_3 import ChangeBoard
class Othello:
    """
    # black 1 white 2
    board, changeable_Pos, done = othello.make()
    for _ in range(100):
    rowpos, colpos  = np.where(changeable_Pos==1)
    if len(rowpos)==0:
        setrow, setcol=0,0
    else:
        idx = np.random.randint(len(rowpos))
        setrow, setcol = rowpos[idx], colpos[idx]
    print(board)
    print("########")
    board, changeable_Pos, done = othello.step(setrow, setcol)
    if done:
        print(board)
        print("########")
        break
    """
    
    def __init__(self):
        self.num_step = 0
        self.playlog = []

    def make(self):
        self.num_step = 0
        self.playlog = []
        self.cannot_play_flag = 0
        
        self.board = np.zeros((8,8),dtype=int)
        self.board[3,3]=1
        self.board[4,4]=1
        self.board[3,4]=2
        self.board[4,3]=2
        self.color=1
        self.playlog.append(self.board.copy())
        
        available_position = SearchAvailablePosition(self.board, self.color)
        self.Position_Row, self.Position_Col, self.Change_Position = SearchAction(self.board, available_position, self.color)
        self.changeable_Pos = np.zeros((8,8),dtype=int)
        self.changeable_Pos[(self.Position_Row, self.Position_Col)]=1
        return self.board, self.changeable_Pos, self.Position_Row, self.Position_Col,  self.Change_Position, False

    def get_currentFeature(self):
        return self.board, self.changeable_Pos

    def change_board(self, setrow, setcol):
        # if player cannot set othello => pass
        if len(self.Position_Row)>0:
            changeIndex=0
            for i, (_row, _col) in enumerate(zip(self.Position_Row, self.Position_Col)):
                if setrow==_row and setcol==_col:
                    changeIndex = self.Change_Position[i]
                    break
            self.board = ChangeBoard(self.board, setrow, setcol, changeIndex, self.color)
            self.playlog.append(self.board.copy())
            self.cannot_play_flag = 0

            if len(self.board[self.board==0])==0:
                return True
            else:
                return False
        else:
            self.cannot_play_flag+=1
            #if both player cannot othello => end
            if self.cannot_play_flag==2:
                return True
            else:
                return False
    
    def step(self, setrow, setcol):
        done = self.change_board(setrow, setcol)
        self.num_step+=1
        # game is done
        if done:
            self.changeable_Pos = np.zeros((8,8),dtype=int)
            self.Position_Row = []
            self.Position_Col = []
            self.Change_Position = []
            self.color = int(1 + self.color%2)
            return self.board, self.changeable_Pos, self.Position_Row, self.Position_Col, self.Change_Position, done

        self.color = int(1 + self.color%2)
        available_position = SearchAvailablePosition(self.board, self.color)
        self.Position_Row, self.Position_Col, self.Change_Position = SearchAction(self.board, available_position, self.color)
        self.changeable_Pos = np.zeros((8,8),dtype=int)
        self.changeable_Pos[(self.Position_Row, self.Position_Col)]=1

        
        return self.board, self.changeable_Pos, self.Position_Row, self.Position_Col, self.Change_Position, done