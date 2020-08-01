""" 
    Project     : COMP90054@Unimelb, 2020 S1
    Author      : Rongbing Shan
    Date        : 2020.05.30
    Description : Monte Carlo Tree Search with rewards myscore-opscore/10, opscore
"""


from advance_model import *
from utils import *
from copy import deepcopy
import numpy as np
import random,time,sys
sys.path.append('players/Azul_project_group_29/')
from myutils import filter_moves

class myPlayer(AdvancePlayer):
    EXPAND_LIMIT = 1
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
        start_time = time.time()
        MCT = Node(game_state,self.id,None,None,moves)
        while time.time() - start_time < self.EXPAND_LIMIT - 0.15:
            if MCT.rollout():
                break
        best_move = MCT.bestChild().from_move
        return best_move


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
        #using naive search to find
        while not isTerminal(game_state):
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
        return myPlr.score - opPlr.score/10,opPlr.score
    
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