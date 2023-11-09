import random
import math
import numpy as np

#https://rutgers.app.box.com/s/ne763kb0rn7pbel6m3c9y78jyy5386vc
class Cell:
    """ This class is used to record the state of a cell on the ship and any occupants on the cell """
    
    def __init__(self, row, col):
        """ By default, a cell is closed and nothing occupies it """
        self.state = '#'
        self.row = row
        self.col = col
        self.alien = False
        self.crew = False
        self.bot = False
        self.alien1_prob = 0
        self.alien2_prob = 0
        self.crew1_prob = 0
        self.crew2_prob = 0

    def get_crew1_prob(self):
        return self.crew1_prob

    def set_crew1_prob(self, p):
        self.crew1_prob = p

    def get_crew2_prob(self):
        return self.crew2_prob

    def set_crew2_prob(self, p):
        self.crew2_prob = p

    def get_alien1_prob(self):
        return self.alien1_prob

    def set_alien1_prob(self, p):
        self.alien1_prob = p

    def get_alien2_prob(self):
        return self.alien2_prob

    def set_alien2_prob(self, p):
        self.alien2_prob = p

    def get_location(self):
        return self.row, self.col

    def get_state(self):
        return self.state
    
    def is_open(self):
        return self.state == 'O'
    
    def open_cell(self):
        self.state = 'O'
        
    def close_cell(self):
        self.state = '#'

    def contains_bot(self):
        return self.bot

    def add_bot(self):
        self.bot = True

    def remove_bot(self):
        self.bot = False

    def contains_alien(self):
        return self.alien
    
    def add_alien(self):
        self.alien = True

    def remove_alien(self):
        self.alien = False

    def contains_crew(self):
        return self.crew

    def add_crew(self):
        self.crew = True

    def remove_crew(self):
        self.crew = False


class Ship:
    """ This class is used to arrange cells in a grid to represent the ship and generate it at time T=0 """
    
    def __init__(self):
        self.D = 50  # The dimension of the ship as a square
        self.ship = np.asarray([[Cell(i, j) for j in range(self.D)] for i in range(self.D)])  # creates a DxD 2D grid of closed cells
        self.crew_loc = [-1, -1]
        self.crew_probs = self.ship = np.asarray([[0 for j in range(self.D)] for i in range(self.D)])
        self.alien_probs = self.ship = np.asarray([[0 for j in range(self.D)] for i in range(self.D)])

    def get_crew_probs(self):
        return self.crew_probs

    def get_alien_probs(self):
        return self.alien_probs

    def set_crew_probs(self, i, j, p):
        self.crew_probs[i][j] = p

    def set_alien_probs(self, i, j, p):
        self.alien_probs[i][j] = p

    def set_crew_loc(self, i, j):
        self.crew_loc = [i, j]
    
    def get_crew_loc(self):
        return self.crew_loc

    def get_sensor_region(self, i, j, k):
        return self.ship[max(i - k, 0):min(i + k + 1, self.D), max(j - k, 0):min(j + k + 1, self.D)]

    def print_ship(self):
        """ This function is used to visualize the current state of cells in the ship """
        for i in range(len(self.ship)):
            for j in range(len(self.ship[0])):
                if self.ship[i][j].contains_alien() and self.ship[i][j].contains_bot():
                    print('X', end=" ")
                else:
                    if self.ship[i][j].contains_alien() and self.ship[i][j].contains_crew():
                        print('D', end=" ")
                    else:
                        if self.ship[i][j].contains_alien():
                            print('a', end=" ")
                        else:
                            if self.ship[i][j].contains_crew():
                                print('c', end=" ")
                            else:
                                if self.ship[i][j].contains_bot():
                                    print('b', end=" ")
                                else:
                                    print(self.ship[i][j].get_state(), end=" ")
            print()
            
    def one_neighbor(self, i, j):
        """ If the given cell has more than 1 open neighbor, returns False, otherwise returns True """
        neighbors = 0
        if i > 0:  # If not in top row, check neighbor above
            up = self.ship[i-1][j]
            if up.is_open():
                neighbors += 1
        if i < self.D-1:  # If not in last row, check neighbor below
            down = self.ship[i+1][j]
            if down.is_open():
                neighbors += 1
        if j > 0:  # If not in left-most column, check neighbor to the left
            left = self.ship[i][j-1]
            if left.is_open():
                neighbors += 1
        if j < self.D-1:  # If not in right-most column, check neighbor to the right
            right = self.ship[i][j+1]
            if right.is_open():
                neighbors += 1
        if neighbors == 1:
            return True
        else:
            return False
            
    def closed_neighbors(self, i, j):
        """ Returns a list of closed neighbors for a given cell """
        neighbors = []  # keeps track of neighboring cells that are closed
        if i > 0 and not self.ship[i-1][j].is_open():
            neighbors.append((i-1, j))
        if i < self.D-1 and not self.ship[i+1][j].is_open():
            neighbors.append((i+1, j))
        if j > 0 and not self.ship[i][j-1].is_open():
            neighbors.append((i, j-1))
        if j < self.D-1 and not self.ship[i][j+1].is_open():
            neighbors.append((i, j+1))
        return neighbors
    
    def get_unoccupied_cell(self, is_crew):
        while True:
            i = random.randint(0, (self.D-1)) 
            j = random.randint(0, (self.D-1))
            if not is_crew:
                if self.ship[i][j].is_open() and not self.ship[i][j].contains_alien() and not self.ship[i][j].contains_bot():
                    return i, j
            else:
                if self.ship[i][j].is_open() and not self.ship[i][j].contains_bot():
                    return i, j

    def get_unoccupied_alien_cell(self, k):
        while True:
            i = random.randint(0, (self.D-1))
            j = random.randint(0, (self.D-1))

            if (i < (self.crew_loc[0] - k)) or (i > (self.crew_loc[0] + k)):
                if (j < (self.crew_loc[1] - k)) or (j > (self.crew_loc[1] + k)):
                    if self.ship[i][j].is_open() and not self.ship[i][j].contains_bot():
                        return i, j

    def empty_ship(self):
        for i in range(self.D):
            for j in range(self.D):
                if self.ship[i][j].contains_bot():
                    self.ship[i][j].remove_bot()
                if self.ship[i][j].contains_alien():
                    self.ship[i][j].remove_alien()
                if self.ship[i][j].contains_crew():
                    self.set_crew_loc(-1, -1)
                    self.ship[i][j].remove_crew()

    def generate_ship(self):
        """ This function generates the ship at time T=0 """
        
        # Randomly select a cell on the interior of the ship (not on the edge)
        i = random.randint(1, (self.D-2)) 
        j = random.randint(1, (self.D-2))
                
        self.ship[i][j].open_cell()  # Open the selected cell
        
        flag = True
        while flag:
            blocked = []  # Keeps track of all blocked cells that neighbor exactly one open cell
            for i in range(self.D):
                for j in range(self.D):
                    if self.ship[i][j].get_state() == "#" and self.one_neighbor(i, j): 
                        blocked.append((i, j))  # Add cell to list if it is blocked and neighbors exactly one open cell
            if len(blocked) == 0:  # Exit the loop if there are no more such cells
                flag = False
                continue
            i, j = random.sample(blocked, 1)[0]  # Randomly select one of these cells
            self.ship[i][j].open_cell()  # Open the selected cell
        
        opened = []  # Keeps track of all open cells that neighbor exactly one open cell (dead ends)
        for i in range(self.D):
            for j in range(self.D):
                # neighboringCells = self.oneNeighbor(i, j)
                if self.ship[i][j].state == "O" and self.one_neighbor(i, j):
                    opened.append(self.closed_neighbors(i, j))
        
        opened = random.sample(opened, math.floor(len(opened)/2))  # Randomly select half of the dead ends
        for cell in opened:
            i, j = random.sample(cell, 1)[0]  # For each dead end, randomly select one closed neighbor
            self.ship[i][j].open_cell()  # Open the selected closed neighbor




