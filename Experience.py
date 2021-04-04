import sys, os
import numpy as np
import time
import pygame
from pygame.locals import *

from warnings import filterwarnings
filterwarnings('ignore')

#from GameVision import OthelloGame
from GameVision import OthelloGame
from Othello import Othello

class randomAgent:
    def take_action(self, board, changeable_Pos, Position_Row, Position_Col, Change_Position):
        if len(Change_Position)==0:
            setrow, setcol = 0,0
        else:
            idx = np.random.randint(len(Position_Row))
            setrow, setcol = Position_Row[idx], Position_Col[idx]
        return setrow, setcol

class greedyAgent:
    def take_action(self, board, changeable_Pos, Position_Row, Position_Col, Change_Position):
        if len(Change_Position)==0:
            setrow, setcol = 0,0
        else:
            Change_Position_len = [len(l[0]) for l in Change_Position]
            idx_l = [i for i, l in enumerate(Change_Position_len) if l==max(Change_Position_len)]
            idx = np.random.choice(idx_l)
            setrow, setcol = Position_Row[idx], Position_Col[idx]
        return setrow, setcol

if __name__ == "__main__":
    game = OthelloGame()
    othello = Othello()
    game.main()
    agent_1 = randomAgent()
    agent_2 = greedyAgent()
    while (1):
        for event in pygame.event.get():
            # 閉じるボタンが押されたら終了
            if event.type == QUIT:                    
                pygame.display.quit()
                # Pygameの終了(画面閉じられる)
                pygame.quit()                                   
                sys.exit()

        # black 1 white 2
        board, changeable_Pos, Position_Row, Position_Col, Change_Position, done = othello.make()
        game.updateBoard(board)
        for _ in range(100):
            setrow, setcol = agent_1.take_action(board, changeable_Pos, Position_Row, Position_Col, Change_Position)
            board, changeable_Pos, Position_Row, Position_Col, Change_Position, done = othello.step(setrow, setcol)
            
            game.updateBoard(board)
            if done:
                time.sleep(0.5)
                break

            setrow, setcol = agent_2.take_action(board, changeable_Pos, Position_Row, Position_Col, Change_Position)
            board, changeable_Pos, Position_Row, Position_Col, Change_Position, done = othello.step(setrow, setcol)
            
            game.updateBoard(board)
            if done:
                time.sleep(0.5)
                break

    sys.exit(0)