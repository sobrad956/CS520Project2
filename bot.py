
from queue import Queue
from queue import PriorityQueue
import math
import numpy as np
import random




class Bot:
    def __init__(self, row, col, ship, type):
        self.row = row
        self.col = col
        self.ship = ship
        self.type = type
        self.ship.ship[self.row][self.col].add_bot()

    def __gt__(self, other):
        return self.priority()
    
    def move_up(self):
        self.ship.ship[self.row][self.col].remove_bot()
        self.row -= 1
        self.ship.ship[self.row][self.col].add_bot()
        if self.ship.ship[self.row][self.col].contains_alien():
            return("Mission Failed, Bot Captured")

    def move_down(self):
        self.ship.ship[self.row][self.col].remove_bot()
        self.row += 1
        self.ship.ship[self.row][self.col].add_bot()
        if self.ship.ship[self.row][self.col].contains_alien():
            return("Mission Failed, Bot Captured")
    
    def move_right(self):
        self.ship.ship[self.row][self.col].remove_bot()
        self.col += 1
        self.ship.ship[self.row][self.col].add_bot()
        if self.ship.ship[self.row][self.col].contains_alien():
            return("Mission Failed, Bot Captured")

    def move_left(self):
        self.ship.ship[self.row][self.col].remove_bot()
        self.col -= 1
        self.ship.ship[self.row][self.col].add_bot()
        if self.ship.ship[self.row][self.col].contains_alien():
            return("Mission Failed, Bot Captured")
        
    def found_crew(self):
        if(self.ship.ship[self.row][self.col].contains_crew()):
            self.ship.ship[self.row][self.col].remove_crew()
            return True
            
    def get_row(self):
        return self.row
    def get_col(self):
        return self.col
    def get_type(self):
        return self.type
        
    
    

    #Search algos...

    #Standard bfs works for bot1 and bot2
    def cell_check(self, cell, dict):
        #dict len check since we proceed with search unless the alien is there on first move
        #Can't just do dict length, depends on order for first 4 nodes, look backward?
        return ((not cell.contains_alien()) or len(dict)>2) and cell.is_open()


    def bfs(self, start_cell):
        
        
        fringe = Queue()
        visited = []
        #Use dict to generate path since no pointers in Python
        path = []
        prev = {start_cell: None}

        cur_state = start_cell
        fringe.put(cur_state)
        
    
        while not fringe.empty():
            cur_state = fringe.get()

            visited.append(cur_state)
            if cur_state.contains_crew():
                #print(cur_state.get_location())
                path.append([cur_state.get_location()])
                while(prev[cur_state] != start_cell):
                    path.append([prev[cur_state].get_location()])
                    cur_state = prev[cur_state]
                path.reverse()
                return path
            
            children = []
            cur_row = cur_state.row
            cur_col = cur_state.col
            
            if(cur_row != 0):
                children.append(self.ship.ship[cur_row-1][cur_col])
            if(cur_row != (self.ship.D -1)):
                children.append(self.ship.ship[cur_row+1][cur_col])
            if(cur_col != 0):
                children.append(self.ship.ship[cur_row][cur_col-1])
            if(cur_col != (self.ship.D - 1)):
                children.append(self.ship.ship[cur_row][cur_col+1])
            
            for child in children:
                if(child.is_open() and (child not in visited)):
                    #Make sure the first move doesn't contain an alien, otherwise proceed assuming alien will move (per zulip)
                    if((not child.contains_alien()) or (not cur_state == start_cell)):
                        fringe.put(child)
                        prev[child] = cur_state
        print("f")
        return [0]
    
    def children_clear(self, parent):
        par_row, par_col = parent.get_location()
        #par_col = parent.get_col()
        if par_row > 0:
            if self.ship.ship[par_row -1][par_col].contains_alien():
                return False
        if par_row < self.ship.D - 1:
            if self.ship.ship[par_row +1][par_col].contains_alien():
                return False
        if par_col > 0:
            if self.ship.ship[par_row][par_col-1].contains_alien():
                return False
        if par_col < self.ship.D - 1:
            if self.ship.ship[par_row][par_col+1].contains_alien():
                return False
        return True
    
    def mod_bfs(self, start_cell):
           
        
        fringe = Queue()
        visited = []
        #Use dict to generate path since no pointers in Python
        path = []
        prev = {start_cell: None}

        cur_state = start_cell
        fringe.put(cur_state)
        
    
        while not fringe.empty():
            cur_state = fringe.get()

            visited.append(cur_state)
            if cur_state.contains_crew():
                #print(cur_state.get_location())
                path.append([cur_state.get_location()])
                while(prev[cur_state] != start_cell):
                    path.append([prev[cur_state].get_location()])
                    cur_state = prev[cur_state]
                path.reverse()
                return path
            
            children = []
            cur_row = cur_state.row
            cur_col = cur_state.col
            
            if(cur_row != 0):
                children.append(self.ship.ship[cur_row-1][cur_col])
            if(cur_row != (self.ship.D -1)):
                children.append(self.ship.ship[cur_row+1][cur_col])
            if(cur_col != 0):
                children.append(self.ship.ship[cur_row][cur_col-1])
            if(cur_col != (self.ship.D - 1)):
                children.append(self.ship.ship[cur_row][cur_col+1])
            
            for child in children:
                #print(not cur_state == start_cell)
                if(child.is_open() and (child not in visited)):
                    #Make sure the first move doesn't contain an alien, otherwise proceed assuming alien will move (per zulip)
                    if(((not child.contains_alien()) and self.children_clear(child)) or (not cur_state == start_cell)):
                        fringe.put(child)
                        prev[child] = cur_state
        print("f")
        return [0]
    
    def priority(self, cell):
        ###Define the priority of a cell to be its euclidean distance from the crew member minus the number of adjacent squares that are open.
        ### Our goal is to avoid being backed into corners in order to best avoid the aliens
        crew_loc = self.ship.get_crew_loc()
        priority = ((crew_loc[0]-self.get_row())) + ((crew_loc[1]-self.get_col())) #Shortest possible distance between current loc and crew member
        #Reduce priority for each open square bordering
        if(self.get_row() != 0):
            if(self.ship.ship[self.get_row()-1][self.get_col()].is_open()):
                priority -= 1
        if(self.get_row() != self.ship.D-1):
            if(self.ship.ship[self.get_row()+1][self.get_col()].is_open()):
                priority -= 1
        if(self.get_col() != 0):
            if(self.ship.ship[self.get_row()][self.get_col()-1].is_open()):
                priority -= 1
        if(self.get_col() != self.ship.D-1):
            if(self.ship.ship[self.get_row()][self.get_col()+1].is_open()):
                priority -= 1

        

        return priority
        
    

    
        
    def bot4path(self, start_cell):
        fringe = PriorityQueue()
        visited = []
        #Use dict to generate path since no pointers in Python
        path = []
        prev = {start_cell: None}

        cur_state = start_cell
        fringe.put((self.priority(cur_state), cur_state))
        
        depth = 0
        while not fringe.empty():
            cur_state = fringe.get()[1]
            #print(cur_state)

            visited.append(cur_state)
            if cur_state.contains_crew():
                #print(cur_state.get_location())
                path.append([cur_state.get_location()])
                while(prev[cur_state] != start_cell):
                    path.append([prev[cur_state].get_location()])
                    cur_state = prev[cur_state]
                path.reverse()
                return path
            
            children = []
            cur_row = cur_state.row
            cur_col = cur_state.col
            
            
            if(cur_row != 0):
                child = self.ship.ship[cur_row-1][cur_col]
                child.set_depth(cur_state.get_depth() + 1)
                children.append(child)
            if(cur_row != (self.ship.D -1)):
                child = self.ship.ship[cur_row+1][cur_col]
                child.set_depth(cur_state.get_depth() + 1)
                children.append(child)
            if(cur_col != 0):
                child = self.ship.ship[cur_row][cur_col-1]
                child.set_depth(cur_state.get_depth() + 1)
                children.append(child)
            if(cur_col != (self.ship.D - 1)):
                child = self.ship.ship[cur_row][cur_col+1]
                child.set_depth(cur_state.get_depth() + 1)
                children.append(child)

            for child in children:
                #print(not cur_state == start_cell)
                if(child.is_open() and (child not in visited)):
                    #Make sure the first move doesn't contain an alien, otherwise proceed assuming alien will move (per zulip)
                    if(((not child.contains_alien()) and self.children_clear(child)) or (not cur_state == start_cell)):
                        fringe.put((self.priority(child) + child.get_depth(), child))
                        prev[child] = cur_state
            depth += 1
        print("f")
        return [0]
        
    
    def path_to_move(self, path):
        """Converts the sequence of coordinates to a series of moves"""
        #1 = left, 2 = right, 3 = up 4 = down
        move_seq = Queue()

        prev_row = self.get_row()
        prev_col = self.get_col()

        if(path == [0]):
            return [0]

        for coords in path:
            next_row = coords[0][0]
            next_col = coords[0][1]
            #print("prev row", prev_row)
            #print("next_row")
            if(next_col == prev_col-1):
                move_seq.put(1)
            elif(next_col == prev_col+1):
                move_seq.put(2)
            elif(next_row == prev_row-1):
                move_seq.put(3)
            else:
                move_seq.put(4)

            prev_row = coords[0][0]
            prev_col = coords[0][1]
        
        return(move_seq)
    
    def move(self, move_seq):
        if(move_seq == [0]):
            return(move_seq)
        
        next_move = move_seq.get()
        if(next_move == 1):
            self.move_left()
        elif(next_move == 2):
            self.move_right()
        elif(next_move == 3):
            self.move_up()
        else:
            self.move_down()
        return(move_seq)