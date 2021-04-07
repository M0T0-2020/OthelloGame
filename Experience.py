import sys, os
import numpy as np
import time
import pygame
from pygame.locals import *

import torch

from warnings import filterwarnings
filterwarnings('ignore')

#from GameVision import OthelloGame
from GameVision import OthelloGame
from Othello import Othello
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from RL.rule_agent import randomAgent, greedyAgent
from RL.DQN_agent import agent as dqn_agent
from RL.A2C_agent import agent as A2C_agent

if __name__ == "__main__":
    game = OthelloGame()
    othello = Othello()
    game.main()

    random_agent = randomAgent()
    greedy_agent = greedyAgent()

    agent_1 = A2C_agent(input_dim=3, lam=0.8, gamma=0.99, lr=1e-4)
    param = torch.load('a2c_1_param1.pt')
    agent_1.model.load_state_dict(param)

    agent_2 = A2C_agent(input_dim=3, lam=0.8, gamma=0.99, lr=1e-4)
    param = torch.load('a2c_2_param1.pt')
    agent_2.model.load_state_dict(param)

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
            #setrow, setcol = agent_1.take_action(board, changeable_Pos, Position_Row, Position_Col, Change_Position)
            setrow, setcol = agent_1.take_determ_action(board, changeable_Pos, Position_Row, Position_Col, Change_Position)
            
            board, changeable_Pos, Position_Row, Position_Col, Change_Position, done = othello.step(setrow, setcol)
            
            game.updateBoard(board)
            if done:
                time.sleep(0.5)
                break

            #setrow, setcol = agent_2.take_action(board, changeable_Pos, Position_Row, Position_Col, Change_Position)
            #setrow, setcol = random_agent.take_action(board, changeable_Pos, Position_Row, Position_Col, Change_Position)
            setrow, setcol = greedy_agent.take_action(board, changeable_Pos, Position_Row, Position_Col, Change_Position)
            
            board, changeable_Pos, Position_Row, Position_Col, Change_Position, done = othello.step(setrow, setcol)
            
            game.updateBoard(board)
            if done:
                time.sleep(0.5)
                break
    sys.exit(0)