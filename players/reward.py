""" 
    Project     : COMP90055@Unimelb, 2020 S1
    Author      : Rongbing Shan
    Date        : 2020.05.28
    Description : Function to get reward for a spcified move of a player
"""
from model import PlayerState,GameState
from utils import *
from copy import deepcopy
class Reward:
    def __init__(self,gamestate,player_id,move):
        self.move = move
        self.id = player_id
        self.gs = deepcopy(gamestate)
        self.ps = self.gs.players[player_id]
        self.tg = move[2]
        self.tp = move[0]
    
    def CurrentTileReward(self,scale=False):
        FLOOR_SCORES = [-1,-1,-2,-2,-2,-3,-3]
        num_to_floor = self.tg.num_to_floor_line

        # if first token is taken
        if self.tp == Move.TAKE_FROM_CENTRE: 
            if not self.gs.first_player_taken:
                self.ps.GiveFirstPlayerToken()

        # calculate penalties:
        penalties = 0
        curr_floor = 0
        for i in range(len(self.ps.floor)):
            if self.ps.floor[i] == 1:
                curr_floor += 1
        #print(f'@Reward: curr_floor({curr_floor}), numbertoFLoor({num_to_floor})',flush=True)
        for i in range(num_to_floor):
            if curr_floor + i < len(FLOOR_SCORES):
                penalties += FLOOR_SCORES[curr_floor+i]

        #calculate scores got
        score_inc = 0
        targetline = self.tg.pattern_line_dest
        if self.tg.num_to_pattern_line > 0:
            # add tile to pattern line
            self.ps.AddToPatternLine(targetline, 
                self.tg.num_to_pattern_line, self.tg.tile_type)
            # if pattern line is full, a score is guranteed
            if self.ps.lines_number[targetline] == targetline+1:
                tc = self.ps.lines_tile[targetline]
                col = int(self.ps.grid_scheme[targetline][tc])
                self.ps.number_of[tc] += 1
                self.ps.grid_state[targetline][col] = 1
                # count the number of tiles in a continguous line
                # above, below, to the left and right of the placed tile.
                above = 0
                for j in range(col-1, -1, -1):
                    val = self.ps.grid_state[targetline][j]
                    above += val
                    if val == 0:
                        break
                below = 0
                for j in range(col+1,self.ps.GRID_SIZE,1):
                    val = self.ps.grid_state[targetline][j]
                    below +=  val
                    if val == 0:
                        break
                left = 0
                for j in range(targetline-1, -1, -1):
                    val = self.ps.grid_state[j][col]
                    left += val
                    if val == 0:
                        break
                right = 0
                for j in range(targetline+1, self.ps.GRID_SIZE, 1):
                    val = self.ps.grid_state[j][col]
                    right += val
                    if val == 0:
                        break
                # If the tile sits in a contiguous vertical line of 
                # tiles in the grid, it is worth 1*the number of tiles
                # in this line (including itself).
                if above > 0 or below > 0:
                    score_inc += (1 + above + below) * 1.0 

                # In addition to the vertical score, the tile is worth
                # an additional H points where H is the length of the 
                # horizontal contiguous line in which it sits.
                if left > 0 or right > 0:
                    score_inc += (1 + left + right) * 1.0 

                # If the tile is not next to any already placed tiles
                # on the grid, it is worth 1 point.                
                if above == 0 and below == 0 and left == 0 \
                    and right == 0:
                    score_inc += 1
                # complete a column
                if above + below == 4:
                    score_inc += 7
                # complete a row
                if left+right == 4:
                    score_inc += 2
                # complete a set
                if self.ps.number_of[self.tg.tile_type] ==5:
                    score_inc += 10

        #print(f'@Reward: score inc:({score_inc}), penalties:({penalties})', flush=True)
        if scale:    
            return (score_inc+penalties)/30
        else:
            return (score_inc+penalties)

    def instantScoreOfMove(self,scale=False):
        # The player is taking tiles from the centre
        if self.tp == Move.TAKE_FROM_CENTRE: 
            if not self.gs.first_player_taken:
                self.ps.GiveFirstPlayerToken()

        if self.tg.num_to_floor_line > 0:
            ttf = []
            for i in range(self.tg.num_to_floor_line):
                ttf.append(self.tg.tile_type)
            self.ps.AddToFloor(ttf)

        if self.tg.num_to_pattern_line > 0:
            self.ps.AddToPatternLine(self.tg.pattern_line_dest, 
                self.tg.num_to_pattern_line, self.tg.tile_type)

        score,_ = self.ps.ScoreRound()
        bonus = self.ps.EndOfGameScore()
        return score+bonus
