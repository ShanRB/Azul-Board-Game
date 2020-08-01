"""
    Project     : COMP90055@Unimelb, 2020 S1
    Author      : Rongbing Shan
    Date        : 2020.05.30
    Description : Advanced random player with filter
"""
from advance_model import *
from utils import *
from model import TileGrab

import time
import random

class myPlayer(AdvancePlayer):
    def __init__(self,_id):
        super().__init__(_id)
    
    def SelectMove(self,moves,game_state):
        moves = filter_moves(moves)
        return random.choice(moves)

def filter_moves(moves):
    filtered_moves = []

    for move in moves:
        tg = move[2]
        if tg.number != tg.num_to_floor_line:
            filtered_moves.append(move)
    if len(filtered_moves)==0:
        filtered_moves = moves
    
    return filtered_moves