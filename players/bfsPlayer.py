""" 
    Project     : COMP90054@Unimelb, 2020 S1
    Author      : Rongbing Shan
    Date        : 2020.05.30
    Description : Best First Search Algorithm
"""


from advance_model import *
from utils import *
from copy import deepcopy
import random,time,sys,heapq
sys.path.append('players/Azul_project_group_29/')
from myutils import filter_moves,Reward


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
        start_time = time.time()
        pq = PriorityQueue()
        moves = filter_moves(moves, game_state.players[self.id])
        pq.push((game_state,self.id,moves,[],0),0)
        while time.time() - start_time <0.8:
            node = pq.pop()
            gs = node[0]
            plr_id = node[1]
            path = node[3]
            #get successors
            for move in node[2]:
                id =abs(1-plr_id)
                reward = Reward(gs,plr_id,move).CurrentTileReward() + node[-1]
                state = deepcopy(gs)
                state.ExecuteMove(plr_id,move)
                childplr = state.players[id]
                childMoves = childplr.GetAvailableMoves(state)
                childMoves = filter_moves(childMoves,childplr)
                path.append(move)
                pq.push((state,id,childMoves,path,reward),reward)
            pq.push(node,node[-1])

        path = pq.pop()[3]
        print('length of path: ', len(path),flush= True)
        bestmove = path[0]
        return bestmove


class PriorityQueue:
    """
      deveoped by CS188 of Berkeley
      copied from packman assignment 1
    """
    def  __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (-priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)
