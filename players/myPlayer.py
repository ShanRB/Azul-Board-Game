""" 
    Project     : Azul Project COMP90054@Unimelb, 2020 S1
    Author      : Rongbing Shan
    Date        : 2020.06.02
    Description : Monte Carlo Tree Search agent
"""


from advance_model import *
from utils import *
from copy import deepcopy
import numpy as np
import random,time


class myPlayer(AdvancePlayer):
    EXPAND_LIMIT = 1
    # initialize
    # The following function should not be changed at all
    def __init__(self,_id):
        super().__init__(_id)
        self.round = 0

    # Each player is given 5 seconds when a new round started
    # If exceeds 5 seconds, all your code will be terminated and 
    # you will receive a timeout warning
    def StartRound(self,game_state):
        self.round += 1
        self.moves = 0
        return None

    # Each player is given 1 second to select next best move
    # If exceeds 5 seconds, all your code will be terminated, 
    # a random action will be selected, and you will receive 
    # a timeout warning
    def SelectMove(self, moves, game_state):
        # start time
        start_time = time.time()
        #moves made increase by 1
        self.moves += 1
        # if first few moves, use naive search to fill pattern line full as most as possible
        if self.round == 1 and self.moves < 3:
            #print(self.moves,' - first two move in a game, naive search')
            best_move = advance_naive_search(moves)
        #else goes to MCT
        else: 
            #initialize the root node for the tree
            MCT = Node(game_state,self.id,None,None,moves)
            #within allowed time, roll out the tree
            while time.time() - start_time < self.EXPAND_LIMIT - 0.15:
                if MCT.rollout():
                    break
            #get the best move
            best_move = MCT.bestChild().from_move
        return best_move


class Node:
    """Monte Carlo Tree Node"""
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

    def add_child(self,Node):
        # add child to a node
        self.childs.append(Node)
    
    def fully_expanded(self):
        # if the node has been fully expanded
        return len(self.moves) == 0

    def bestChild(self):
        # find the best child for the node with largest value
        Values = [child.Value for child in self.childs]
        return self.childs[np.argmax(Values)]

    def select(self):
        # using UCB to select node to explore
        UCB = []
        for child in self.childs:
            UCB1 = child.Value + \
                np.sqrt(2*np.log(self.visited)/child.visited)
            UCB.append(UCB1)
        return self.childs[np.argmax(UCB)]

    def expand(self):
        # expand a node to add a not yet explored node to its childs
        move = self.moves.pop()
        child_gs = deepcopy(self.game_state)
        child_plrid = abs(1-self.player_id)
        child_gs.ExecuteMove(self.player_id,move)
        childNode = Node(child_gs,child_plrid,self,move)
        self.add_child(childNode)
        return childNode

    def simulate(self):
        # simulate, using naive serach to next state untile a leaf node is met
        game_state = deepcopy(self.game_state)
        curr_plr_id = self.player_id
        #using naive search to explore the path until leaf node
        while game_state.TilesRemaining():
            plr = game_state.players[curr_plr_id]
            moves = plr.GetAvailableMoves(game_state)
            move = naive_search(moves)
            game_state.ExecuteMove(curr_plr_id,move)
            curr_plr_id = abs(1-curr_plr_id)
        # calculate rewards for both players
        myPlr = game_state.players[self.rootID]
        opPlr = game_state.players[abs(self.rootID-1)]
        myPlr.ScoreRound()
        myPlr.EndOfGameScore()
        opPlr.ScoreRound()
        opPlr.EndOfGameScore()
        # changing the rewards can impact the behavior of agent
        Q1 = myPlr.score
        Q2 = opPlr.score - myPlr.score
        if isGameEnd(game_state):
            if opPlr.score > myPlr.score:
                Q1 -= 50
            elif opPlr.score == myPlr.score:
                Q1 -= 30
        return Q1,Q2
    
    def backpropagate(self,myQ,opQ,gamma=1.0):
        # back propagate the rewards
        myQ = myQ * gamma
        opQ = opQ * gamma
        self.visited += 1
        if self.player_id == self.rootID:
            self.Q += opQ 
        else:
            self.Q += myQ 
        self.Value = self.Q / self.visited
        if self.parent is not None:
            self.parent.backpropagate(myQ,opQ,gamma)


    def rollout(self,gamma=1.0):
        # roll out the Monte Carlo Tree search for 1 round
        node = self
        full_expansion = True
        while node.game_state.TilesRemaining():
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

def isGameEnd(game_state):
    # check if game state is a round end or game end
    for plr in game_state.players:
        if plr.GetCompletedRows() > 0:
            return True
    return False


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

def filter_moves(moves,plr_state):
    """ filter moves 
        to filter out moves that put all tiles to floor
        and filter out moves that didn't fill out a patter line full but put tiles to floor
    """

    if len(moves) <=10:
        return moves

    filtered_moves = []
    snd_filmove = []

    # filter out moves that put all tiles to floor
    for move in moves:
        tg = move[2]
        if tg.number != tg.num_to_floor_line:
            filtered_moves.append(move)
    # if all remaining moves are targeted moves, undo the filtering
    if len(filtered_moves)==0:
        return moves
    
    # filter out moves that didn't fill out a pattern line full, but put some tile to floor
    for move in filtered_moves:
        if (tg.num_to_pattern_line + plr_state.lines_number[tg.pattern_line_dest] < tg.pattern_line_dest + 1) and\
            tg.num_to_floor_line > 0:
            pass;
        else:
            snd_filmove.append(move)
    
    return snd_filmove

def advance_naive_search(moves):
    # Select move that involves placing the most number of tiles
    # to fully fit a pattern line. Tie break on number placed in floor line.
        most_to_line = -1
        corr_to_floor = 0
        best_move = None
        found = False
        for mid,fid,tgrab in moves:
            # check if the move can fill the destination patter line
            full_pattern = False
            if tgrab.pattern_line_dest >= 0:
                full_pattern = (tgrab.num_to_pattern_line == tgrab.pattern_line_dest+1) and\
                    (tgrab.num_to_floor_line == 0)
            else:
                full_pattern = False

            if most_to_line == -1:
                best_move = (mid,fid,tgrab)
                most_to_line = tgrab.num_to_pattern_line
                corr_to_floor = tgrab.num_to_floor_line
                continue
            
            if full_pattern:
                found = True

            if full_pattern and tgrab.num_to_pattern_line > most_to_line:
                best_move = (mid,fid,tgrab)
                most_to_line = tgrab.num_to_pattern_line
                corr_to_floor = tgrab.num_to_floor_line
            elif full_pattern and \
                tgrab.num_to_pattern_line == most_to_line and \
                tgrab.num_to_pattern_line < corr_to_floor:
                best_move = (mid,fid,tgrab)
                most_to_line = tgrab.num_to_pattern_line
                corr_to_floor = tgrab.num_to_floor_line

        if not found:
            best_move = naive_search(moves) 
        
        return best_move