import pandas as pd
import numpy as np
import sys, os
from warnings import filterwarnings
import random, time, math
#import matplotlib.pyplot as plt
import pygame
from pygame.locals import *
#from pygame.locals import QUIT
filterwarnings('ignore')

class OthelloGame:

    def __init__(self):
        # W-H  600-500の画面を生成
        self.W = 600
        self.H = 600                          
        self.blue = (0, 130, 254)
        self.green = (34,139,34)
        self.emerarudo_green = (46, 244, 208)
        self.yellow = (255,255,51)
        self.white = (255,255,255)
        self.black = (0,0,0)
        self.r = min(self.H/18, self.W/18)
        self.init_flag = True

        self.first_board = np.zeros((8,8),dtype=int)
        self.first_board[3,3]=1
        self.first_board[4,4]=1
        self.first_board[3,4]=2
        self.first_board[4,3]=2

        self.changesmooth_sleep_time = 0.6
        self.showResult_sleep_time = 0.1
        self.drawCercle_sleep_time = 0.2



    def main(self):
        self.screen = pygame.display.set_mode(size=(self.W, self.H))               
        # タイトルバーに表示する文字 
        pygame.display.set_caption("Othello")
        #clock = pygame.time.Clock()
        self.drawBoardLine()
        pygame.display.update()

    def drawCercle(self, board):
        self.drawBoardLine()
        for i, row in enumerate(board):
            v_c = i*self.W/8+self.W/16
            for j, (color) in enumerate(row):
                h_c = j*self.H/8+self.H/16
                center = (h_c, v_c)
                if color==0:
                    pass
                if color==1:
                    color = self.black
                    pygame.draw.circle(self.screen, color, center, self.r)
                if color==2:
                    color = self.white
                    pygame.draw.circle(self.screen, color, center, self.r)
        pygame.display.update()
        time.sleep(self.drawCercle_sleep_time)


    def changesmooth(self, board, board_1, setRow, setCol, changeRow, changeCol):
        for i in range(8):
            w, h = np.cos(i*np.pi/4), np.sin(i*np.pi/4)
            w = int(w/abs(w)) if abs(w)>1e-10 else 0
            h = int(h/abs(h)) if abs(h)>1e-10 else 0
            sleepflag = False
            for j in range(1,8):
                r = int(setRow + j*w)
                c = int(setCol + j*h)
                if r<0 or r>7 or c<0 or c>7:
                    break
                if board[r,c]==board_1[r,c]:
                    break
                if board[r,c]!=board_1[r,c]:
                    board_1[r,c] = board[r,c]
                    self.drawBoardLine()
                    self.drawCercle(board_1)
                    sleepflag=True
            if sleepflag:
                time.sleep(self.changesmooth_sleep_time)
    
    def updateBoard(self, board):
        print(board)
        if len(board[board!=0])==4:
            self.last_board = self.first_board.copy()
            self.drawCercle(board)

        else:
            # game end
            #ひっくり返った場所は前のままにしたboardを作る
            board_1 =  np.where((self.last_board!=board)&(self.last_board!=0), self.last_board, board)
            self.drawBoardLine()
            self.drawCercle(board_1)
            

            #置いた場所
            setRow, setCol = np.where((board!=0)&(self.last_board==0))
            if len(setRow)==0:
                return None
            
            setRow, setCol = setRow[0], setCol[0]

            #ひっくり返った場所
            changeRow, changeCol = np.where((self.last_board!=board)&(self.last_board!=0))
            changeRow, changeCol = list(changeRow), list(changeCol)

            self.changesmooth(board, board_1, setRow, setCol, changeRow, changeCol)
            self.last_board = board.copy()
            #time.sleep(0.75)
            if len(board[board==0])==0:
                self.showResult(board)
            

    def showResult(self, board):
        board_0 = np.zeros(board.shape).flatten()
        board_1 = np.sort(board.flatten())[::-1]
        for i in range(32):
            board_0[i] = board_1[i]
            board_0[-i-1] = board_1[-i-1]
            showboard = board_0.reshape(board.shape).T
            self.drawCercle(showboard)
        time.sleep(self.showResult_sleep_time)
        return None
            

    def drawBoardLine(self):
        self.screen.fill(color=self.green)

        for i in range(1,8):
            v_x_1, v_y_1 = i*(self.W/8), 0
            v_x_2, v_y_2 = i*(self.W/8), self.H

            h_x_1, h_y_1 = 0, i*(self.H/8)
            h_x_2, h_y_2 = self.W, i*(self.H/8)
            pygame.draw.line(self.screen, color=self.black, start_pos=(v_x_1, v_y_1), end_pos=(v_x_2, v_y_2), width=3)   
            pygame.draw.line(self.screen, color=self.black, start_pos=(h_x_1, h_y_1), end_pos=(h_x_2, h_y_2), width=3)
        

if __name__ == "__main__":
    game = OthelloGame()
    game.main()