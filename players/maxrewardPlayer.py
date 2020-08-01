""" 
    Project     : COMP90054@Unimelb, 2020 S1
    Author      : Rongbing Shan
    Date        : 2020.05.38
    Description : Player to get max reward for each move
"""
from advance_model import *
from copy import deepcopy
import sys
sys.path.append('players/Azul_project_group_29/')
from myutils import Reward

class myPlayer(AdvancePlayer):

    # initialize
    # The following function should not be changed at all
    def __init__(self,_id):
        super().__init__(_id)

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
        #plr_state = game_state.players[self.id]
        scores = [Reward(game_state,self.id,move).CurrentTileReward() \
            for move in moves]
        maxscore = max(scores)
        best_move = moves[scores.index(maxscore)]
        return best_move


    # def instantScoreOfMove(self,gamestate,playerstate,move):
    #     ps = deepcopy(playerstate)
    #     gs = deepcopy(gamestate)
    #     tg = move[2]
    #     # The player is taking tiles from the centre
    #     if move[0] == Move.TAKE_FROM_CENTRE: 
    #         if not gs.first_player_taken:
    #             ps.GiveFirstPlayerToken()

    #     if tg.num_to_floor_line > 0:
    #         ttf = []
    #         for i in range(tg.num_to_floor_line):
    #             ttf.append(tg.tile_type)
    #         ps.AddToFloor(ttf)

    #     if tg.num_to_pattern_line > 0:
    #         ps.AddToPatternLine(tg.pattern_line_dest, 
    #             tg.num_to_pattern_line, tg.tile_type)

    #     score,_ = ps.ScoreRound()
    #     return score