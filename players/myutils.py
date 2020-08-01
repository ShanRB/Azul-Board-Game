""" 
    Project     : COMP90054@Unimelb, 2020 S1
    Author      : Rongbing Shan
    Date        : 2020.05.28
    Description : utilies used in agents
"""

from utils import *
from model import *
from os import path
import random, heapq
import pandas as pd
import numpy as np
from copy import deepcopy

def filter_moves(moves,plr_state):
    filtered_moves = []
    snd_filmove = []
    for move in moves:
        tg = move[2]
        if tg.number != tg.num_to_floor_line:
            filtered_moves.append(move)
    if len(filtered_moves)==0:
        return moves
    
    for move in filtered_moves:
        if (tg.num_to_pattern_line + plr_state.lines_number[tg.pattern_line_dest] < tg.pattern_line_dest + 1) and\
            tg.num_to_floor_line > 0:
            pass;
        else:
            snd_filmove.append(move)
    
    return snd_filmove

def load_weight(WEIGHT_FILE):
    dir_path = path.dirname(path.realpath(__file__))
    savepath = path.join(dir_path,WEIGHT_FILE+'.pkl')
    #print('$$$ - load weight',savepath ,flush=True)
    if path.exists(savepath):
        weights = pd.read_pickle(savepath)
    else:
        weights = pd.DataFrame()
    return weights

def save_weight(weights,WEIGHT_FILE):
    dir_path = path.dirname(path.realpath(__file__))
    savepath = path.join(dir_path,WEIGHT_FILE+'.pkl')
    #print('$$$ - save weight',savepath, flush=True)
    weights.to_pickle(savepath)


def StringOfMove(move):
    tg = move[2]
    Movestring = "{}: {} to line {}, {} to floor".format(\
        TileToString(tg.tile_type), tg.num_to_pattern_line,\
            tg.pattern_line_dest,tg.num_to_floor_line)
    return Movestring

def epsilon_gready(epsilon):
    """if choose move randomly by epsilon gready MAB

    Arguments:
        epsilon {float} -- threshold for random choosing

    Returns:
        Bool -- True of random choose
    """
    if random.uniform(0, 1) < epsilon:
        return True
    return False


def getfeatures(gamestate,player_id,move):
    """
        features independent of move
    """
    tg = move[2]
    ps = gamestate.players[player_id]
    op_ps = gamestate.players[abs(1-player_id)]
    # feature 1, bias
    features = [1]
    
    # feature 2 - 6, bag tile percent
    for tile in Tile:
        if not gamestate.bag and len(gamestate.bag) != 0:
            tile_prt = gamestate.bag.count(tile) * 1.0 / len(gamestate.bag)
        else:
            tile_prt = 0
        features.append(tile_prt) 
    # feature 7 - 11, center tile percent
    for tile in Tile:
        if gamestate.centre_pool.total > 0:
            tile_prt = gamestate.centre_pool.tiles[tile] * 1.0/gamestate.centre_pool.total
        else:
            tile_prt = 0
        features.append(tile_prt)
    # feature 12 - 36, f1-f5 tile percent
    for fac in gamestate.factories:
        for tile in Tile:
            tile_prt = fac.tiles[tile] / 5
            features.append(tile_prt)

    #feature 37, option of tile remaining
    op_remain = 0
    for fac in gamestate.factories:
        for tile in Tile:
            if fac.tiles[tile] > 0:
                op_remain += 1
    if gamestate.centre_pool.total > 0:
        for tile in Tile:
            if gamestate.centre_pool.tiles[tile] > 0:
                op_remain += 1
    features.append(op_remain / 20) 

    #feature 38, floor fill percentage
    floor_cnt = 0
    for i in range(len(ps.floor)):
        floor_cnt += ps.floor[i]
    features.append(floor_cnt/len(ps.floor))
    #feature 39, line fill percentage
    if tg.pattern_line_dest != -1:
        line_prt = ps.lines_number[tg.pattern_line_dest]/ (tg.pattern_line_dest+1)
    else:
        line_prt = ps.lines_number[tg.pattern_line_dest]/ 6
    features.append(line_prt)
    #feature 40, no. of completions
    features.append(ps.GetCompletedSets()/ps.GRID_SIZE)
    #feature 41, no of rows completed
    features.append(ps.GetCompletedRows()/ps.GRID_SIZE)
    #feature 42, no of comlumns completed
    features.append(ps.GetCompletedColumns()/ps.GRID_SIZE)
    #feature 43, row completion percent of the tile
    row = 0
    for i in range(ps.GRID_SIZE):
        row += ps.grid_state[tg.pattern_line_dest][i]
    features.append(row / ps.GRID_SIZE)
    #feature 44, column completion percent of tile
    col = 0
    for i in range(ps.GRID_SIZE):
        col += ps.grid_state[i][tg.tile_type]
    features.append(col / ps.GRID_SIZE)
    #featuer 45, set completion percentage
    features.append(ps.number_of[tg.tile_type] / ps.GRID_SIZE)

    #feature 46-50, opponent line fill percentage
    for i in range(op_ps.GRID_SIZE):
        line_prt = op_ps.lines_number[i] / (i+1)
        features.append(line_prt)
    #feature 51, opponent floor fill percentage
    floor_cnt = 0
    for i in range(len(op_ps.floor)):
        floor_cnt += op_ps.floor[i]
    features.append(floor_cnt/len(op_ps.floor))
    #feature 52-54. opponment completion
    features.append(op_ps.GetCompletedSets()/op_ps.GRID_SIZE)
    features.append(op_ps.GetCompletedRows()/op_ps.GRID_SIZE)
    features.append(op_ps.GetCompletedColumns()/op_ps.GRID_SIZE)

    return np.array(features)

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
        score_inc = 0.0
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
                # # complete a column
                # if above + below == 4:
                #     score_inc += 7
                # # complete a row
                # if left+right == 4:
                #     score_inc += 2
                # # complete a set
                # if self.ps.number_of[self.tg.tile_type] ==5:
                #     score_inc += 10
                score_inc += 7 * (above+below+1)/5
                score_inc += 2 * (above+below+1)/5
                score_inc += 10 * (self.ps.number_of[self.tg.tile_type])/5
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

class Node:
#Monte Carlo Tree Node
    def __init__(self, game_state,player_id,parent=None,from_move=None,moves=None):
        if parent is None:
            self.rootID = player_id
        else:
            self.rootID = parent.rootID
        self.game_state = game_state
        self.player_id = player_id
        self.plr_state = game_state.players[player_id]
        self.parent = parent
        self.from_move = from_move
        self.childs = []
        if moves is None:
            self.moves = self.plr_state.GetAvailableMoves(self.game_state)
        else:
            self.moves = moves
        self.moves = filter_moves(self.moves, self.plr_state)
        self.Q = 0
        self.visited = 0
        self.Value = 0

    def setRootID(self,id):
        self.rootID = id

    def add_child(self,Node):
        self.childs.append(Node)
    
    def makeRoot(self):
        assert(self.player_id == self.rootID)
        self.parent = None
        self.from_move = None

    def fully_expanded(self):
        return len(self.moves) == 0

    def bestChild(self):
        Values = [child.Value for child in self.childs]
        return self.childs[np.argmax(Values)]

    def select(self):
        UCB = []
        for child in self.childs:
            UCB1 = child.Value + \
                np.sqrt(2*np.log(self.visited)/child.visited)
            UCB.append(UCB1)
        return self.childs[np.argmax(UCB)]

    def expand(self):
        move = self.moves.pop()
        child_gs = deepcopy(self.game_state)
        child_plrid = abs(1-self.player_id)
        child_gs.ExecuteMove(self.player_id,move)
        childNode = Node(child_gs,child_plrid,self,move)
        self.add_child(childNode)
        return childNode

    def simulate(self,type='bestR'):
        game_state = deepcopy(self.game_state)
        curr_plr_id = self.player_id
        while not isTerminal(game_state):
            plr = game_state.players[curr_plr_id]
            moves = plr.GetAvailableMoves(game_state)
            move = naive_search(moves)
            game_state.ExecuteMove(curr_plr_id,move)
            curr_plr_id = abs(1-curr_plr_id)
        # calculate rewards for root player
        myPlr = game_state.players[self.rootID]
        opPlr = game_state.players[abs(self.rootID-1)]
        myPlr.ScoreRound()
        myPlr.EndOfGameScore()
        opPlr.ScoreRound()
        opPlr.EndOfGameScore()
        return myPlr.score - opPlr.score/10,opPlr.score - myPlr.score/10
    
    def backpropagate(self,myQ,opQ,gamma=1.0):
        myQ = myQ * gamma
        opQ = opQ * gamma
        self.visited += 1
        if self.player_id == self.rootID:
            self.Q += opQ 
        else:
            self.Q += myQ 
        self.Value = self.Q / self.visited
        #print(f'Q({self.Q}),visited({self.visited}),Value({self.Value})')
        if self.parent is not None:
            self.parent.backpropagate(myQ,opQ,gamma)

    def isLeaf(self):
        for plr in self.game_state.players:
            if plr.GetCompletedRows() >0:
                return True
        if not self.game_state.TilesRemaining():
            return True
        return False

    def rollout(self,gamma=1.0):
        node = self
        full_expansion = True
        while not node.isLeaf():
            if node.fully_expanded():
                #print('full expansion')
                node = node.select()
                continue
            else:
                #print('expand node')
                node = node.expand()
                myQ,opQ = node.simulate()
                node.backpropagate(myQ,opQ,gamma)
                full_expansion = False
                break;
        return full_expansion

def isTerminal(game_state):
    for plr in game_state.players:
        if plr.GetCompletedRows() > 0:
            return True
    if not game_state.TilesRemaining():
        return True
    return False

def max_reward_search(game_state,player_id,moves):
    plr_state = game_state.players[player_id]
    max_reward = -10
    selected_moves = []
    for move in moves:
        tempr = Reward(game_state,player_id,move).CurrentTileReward()
        if tempr > max_reward:
            selected_moves = [move]
        if tempr == max_reward:
            selected_moves.append(move)
    #random tie break
    return random.choice(selected_moves)


def naive_search(moves):
    # Select move that involves placing the most number of tiles
        # in a pattern line. Tie break on number placed in floor line.
        most_to_line = -1
        corr_to_floor = 0

        best_move = None

        for mid,fid,tgrab in moves:
            if most_to_line == -1:
                best_move = (mid,fid,tgrab)
                most_to_line = tgrab.num_to_pattern_line
                corr_to_floor = tgrab.num_to_floor_line
                continue

            if tgrab.num_to_pattern_line > most_to_line:
                best_move = (mid,fid,tgrab)
                most_to_line = tgrab.num_to_pattern_line
                corr_to_floor = tgrab.num_to_floor_line
            elif tgrab.num_to_pattern_line == most_to_line and \
                tgrab.num_to_pattern_line < corr_to_floor:
                best_move = (mid,fid,tgrab)
                most_to_line = tgrab.num_to_pattern_line
                corr_to_floor = tgrab.num_to_floor_line

        return best_move

def same_move(move1,move2):
    if move1[0] != move2[0]:
        return False
    elif move1[1] != move2[1]:
        return False
    else:
        return SameTG(move1[2],move2[2])