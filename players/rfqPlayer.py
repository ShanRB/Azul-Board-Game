""" 
    Project     : COMP90055@Unimelb, 2020 S1
    Author      : Rongbing Shan
    Date        : 2020.05.31
    Description : Model free multi agent 
                  Reinforcement Q Learning Linear Approximation
"""


from advance_model import *
from utils import *
from model import PlayerState,GameState
import numpy as np
import pandas as pd
import random
from copy import deepcopy
import sys
sys.path.append('players/Azul_project_group_29/')
from myutils import *

class myPlayer(AdvancePlayer):
    TRAIN = True
    FEATURE_NUM = 54
    WEIGHT_FILE = 'Q_54_CTR'
    LR = 0.1
    DR = 0.7
    EPSILON = 0.1
    # initialize
    # The following function should not be changed at all
    def __init__(self,_id):
        super().__init__(_id)
        self.prevQ = 0
        self.prevR = 0
        self.prevM = None
        self.prevF = None

    # Each player is given 5 seconds when a new round started
    # If exceeds 5 seconds, all your code will be terminated and 
    # you will receive a timeout warning
    def StartRound(self,game_state):
        return None

    # Each player is given 1 second to select next best move
    # If exceeds 5 seconds, all your code will be terminated, 
    # a random action will be selected, and you will receive 
    # a timeout warning
    def SelectMove(self, moves, game_state):
        print('*' * 30)
        plr_state = game_state.players[self.id]
        moves = filter_moves(moves,plr_state)
        # load weight
        self.weights = load_weight(self.WEIGHT_FILE)
        print('##Start - ', self.weights.shape,flush=True)
        #print('#rfPlayer: current weight shape',self.weights.shape,flush=True)

        maxQ,best_move = self.getMaxQvalue(game_state,moves)

        # if training mode
        if self.TRAIN:
            if epsilon_gready(self.EPSILON):
                #random choose next move
                print('#rfPlayer: random pick',flush=True)
                best_move = random.choice(moves)
                
            print('#rfPlayer: picked move ',StringOfMove(best_move),flush=True)
            # if not the first move
            if self.prevM is not None:
                self.update_weight(maxQ)
            save_weight(self.weights,self.WEIGHT_FILE)
        
            self.prevM = best_move
            self.prevF = getfeatures(game_state,self.id,best_move)
            self.prevQ = self.getQvalue(game_state,best_move)
            self.prevR = Reward(game_state,self.id,best_move).CurrentTileReward(scale=True)

        return best_move


    def update_weight(self,maxQ):
        moveString = StringOfMove(self.prevM)
        print('*update weight for previous move: ', moveString,flush=True)
        #print('$Qvalue - ',Qvalue,flush=True)
        #print('!!update maxQ: ', maxQvalue, flush=True)
        for i in range(self.FEATURE_NUM):
            changeV = self.LR * (self.prevR + self.DR *maxQ -self.prevQ)*self.prevF[i]
            #print('!!update before: ', self.weights.at[i,moveString], flush=True)
            #print(f'!!change value({changeV}),reward({reward}),maxQ({maxQvalue}),Q({Qvalue},f({features[i]}))',flush=True)
            self.weights.at[i,moveString] += changeV
            #print('!!update after : ', self.weights.at[i,moveString], flush=True)

    def getQvalue(self,gamestate,move):
        moveString = StringOfMove(move)
        features = getfeatures(gamestate,self.id,move)
        moveString = StringOfMove(move)
        #if the action has been played
        if moveString in self.weights:
            weight_arrays = self.weights[moveString].to_numpy()
            assert(weight_arrays.shape == features.shape)
            #print('#getQ: weight arrays',weight_arrays,flush=True)
            #print('#getQ: features',features,flush=True)
            qValue = weight_arrays.dot(features) 
            #print('#getQ: Q',qValue,flush=True)
        else:
            # initialize the weights for the action
            tempW = np.zeros(self.FEATURE_NUM)
            self.weights.insert(0,moveString,tempW)
            qValue = 0
        return qValue

    def getMaxQvalue(self,game_state,moves):
        maxQ = -100
        maxMoves = []
        print('#length of moves: ', len(moves),flush=True)
        for move in moves:
            tempQ = self.getQvalue(game_state,move)  
            if tempQ > maxQ:
                maxQ = tempQ
                maxMoves = [move]
            if tempQ == maxQ:
                maxMoves.append(move)
        #random tie break
        maxMove = random.choice(maxMoves)
        return maxQ, maxMove 