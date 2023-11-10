import random
import math
import numpy as np
from queue import Queue
import copy

import bot


class Cell:
    """ This class is used to record the state of a cell on the ship and any occupants on the cell """
    
    def __init__(self, row, col, d):
        """ By default, a cell is closed and nothing occupies it """
        self.state = '#'
        self.row = row
        self.col = col
        self.alien = False
        self.crew = False
        self.bot = False
        self.d = d
        #self.distances = [-1, -1]  # distance from crew member for each crew member, negative if no crew or cell closed

        self.distances = np.asarray([[-1 for j in range(self.d)] for i in range(self.d)])
        #self.alien1_prob = 0
        #self.alien2_prob = 0
        #self.crew1_prob = 0
        #self.crew2_prob = 0

    def set_distance(self, row, col, dist):
        self.distances[row][col] = dist

    # def get_crew1_prob(self):
    #     return self.crew1_prob
    #
    # def set_crew1_prob(self, p):
    #     self.crew1_prob = p
    #
    # def get_crew2_prob(self):
    #     return self.crew2_prob
    #
    # def set_crew2_prob(self, p):
    #     self.crew2_prob = p
    #
    # def get_alien1_prob(self):
    #     return self.alien1_prob
    #
    # def set_alien1_prob(self, p):
    #     self.alien1_prob = p
    #
    # def get_alien2_prob(self):
    #     return self.alien2_prob
    #
    # def set_alien2_prob(self, p):
    #     self.alien2_prob = p

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
    
    def __init__(self, k):
        self.D = 12 # The dimension of the ship as a square
        self.ship = np.asarray([[Cell(i, j, self.D) for j in range(self.D)] for i in range(self.D)])  # creates a DxD 2D grid of closed cells
        self.bot_loc = [-1, -1]  # Stores the initial position of the bot, used to restrict alien generation cells
        self.crew_probs = np.asarray([[0.0 for j in range(self.D)] for i in range(self.D)])
        self.alien_probs = np.asarray([[0.0 for j in range(self.D)] for i in range(self.D)])
        self.k = k
        self.bot = None
        self.num_open_cells = None
        self.open_neighbors = np.asarray([[0.0 for j in range(self.D)] for i in range(self.D)])

    def get_crew_probs(self):
        return self.crew_probs
    #
    def get_alien_probs(self):
        return self.alien_probs
    #
    def set_crew_probs(self, i, j, p):
        self.crew_probs[i][j] = p
    #
    def set_alien_probs(self, i, j, p):
        self.alien_probs[i][j] = p

    def set_bot_loc(self, i, j):
        self.bot_loc = [i, j]
    
    def get_bot_loc(self):
        return self.bot_loc

    def get_sensor_region(self, i, j):
        return self.ship[max(i - self.k, 0):min(i + self.k + 1, self.D), max(j - self.k, 0):min(j + self.k + 1, self.D)]

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
        self.open_neighbors[i][j] = neighbors
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
    
    def get_unoccupied_cell(self):
        """ Returns an unoccupied cell, crew members restricted from bot cell"""
        while True:
            i = random.randint(0, (self.D-1)) 
            j = random.randint(0, (self.D-1))
            if self.ship[i][j].is_open() and not self.ship[i][j].contains_bot():
                return i, j

    def get_unoccupied_alien_cell(self, k):
        """ Returns an unoccupied cell, aliens restricted from detection square"""
        while True:
            i = random.randint(0, (self.D-1))
            j = random.randint(0, (self.D-1))

            if i < (self.bot_loc[0] - k) or i > (self.bot_loc[0] + k):
                if j < (self.bot_loc[1] - k) or j > (self.bot_loc[1] + k):
                    if self.ship[i][j].is_open() and not self.ship[i][j].contains_alien() and not self.ship[i][j].contains_bot():
                        return i, j

    def empty_ship(self):
        """ Resets the ship to default without generating new ship layout """
        self.crew_probs = np.asarray([[0.0 for j in range(self.D)] for i in range(self.D)])
        self.alien_probs = np.asarray([[0.0 for j in range(self.D)] for i in range(self.D)])
        self.bot = None
        for i in range(self.D):
            for j in range(self.D):
                if self.ship[i][j].contains_bot():
                    self.set_bot_loc(-1, -1)
                    self.ship[i][j].remove_bot()
                if self.ship[i][j].contains_alien():
                    self.ship[i][j].remove_alien()
                if self.ship[i][j].contains_crew():
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

    def distances_from_crew(self):
        """ Finds the distance from every cell to the crew members """
        start_cells = []
        for start_i in range(self.D):
            for start_j in range(self.D):
                if self.ship[start_i][start_j].is_open():
                    start_cells.append(self.ship[start_i][start_j])

        self.num_open_cells = len(start_cells)
        for i in range(0, len(start_cells)):
            print(i)
            fringe = Queue()
            visited = []
            cur_state = (start_cells[i], 0)
            fringe.put(cur_state)

            while not fringe.empty():

                cur_state = fringe.get()
                cur_state[0].set_distance(start_cells[i].row, start_cells[i].col, cur_state[1])
                visited.append(cur_state[0])

                children = []
                cur_row = cur_state[0].row
                cur_col = cur_state[0].col
                if cur_row != 0:
                    children.append(self.ship[cur_row - 1][cur_col])
                if cur_row != (self.D - 1):
                    children.append(self.ship[cur_row + 1][cur_col])
                if cur_col != 0:
                    children.append(self.ship[cur_row][cur_col - 1])
                if cur_col != (self.D - 1):
                    children.append(self.ship[cur_row][cur_col + 1])

                for child in children:
                    if child.is_open() and (child not in visited):
                        fringe.put((child, cur_state[1]+1))

    
        # for i in range(len(self.ship)):
        #     for j in range(len(self.ship[0])):
        #         print(self.ship[i][j].get_distance(0), end=" ")
        #     print()
        #
        # print()
        # for i in range(len(self.ship)):
        #     for j in range(len(self.ship[0])):
        #         print(self.ship[i][j].get_distance(1), end=" ")
        #     print()

    def init_crew_prob_one(self):
        p = 1 / (self.num_open_cells - 1)
        mask_func = lambda x: x.is_open() and not x.bot
        mask = np.asarray([list(map(mask_func, row)) for row in self.ship])

        #temp = np.asarray([[random.random() for i in range(12)] for j in range(12)])
        #self.crew_probs[mask] = temp[mask]
        self.crew_probs[mask] = p
    
    def init_alien_prob_one(self):
        det_sq = self.get_sensor_region(self.bot_loc[0], self.bot_loc[1])
        count = 0
        for row in det_sq:
            for elem in row:
                if elem.is_open():
                    count += 1
        num_open_out = self.num_open_cells - count
        p = 1 / num_open_out
        
        mask_func = lambda x: x.is_open()
        mask = np.asarray([list(map(mask_func, row)) for row in self.ship])

    
        x,y = self.get_det_sq_indicies()
        mask[x,y] = False

        self.alien_probs[mask] = p






    #Probability updates

    def get_det_sq_indicies(self):
        #return array of the indices within the detection square
        cent = self.bot_loc
        x = []
        y =[]

        for i in range(cent[0]-(self.k), cent[0]+(self.k)+1):
            for j in range(cent[1]-(self.k), cent[1]+(self.k)+1):
                if i >= 0 and i < self.D-1:
                    if j >= 0 and j < self.D-1:
                        x.append(i)
                        y.append(j)
        return x,y
    
    def get_out_det_sq_indicies(self):

        cent = self.bot_loc
        x = []
        y = []

        for i in range(self.D):
            for j in range(self.D):
                if not (i in range(cent[0]-(self.k), cent[0]+(self.k)+1)):
                    if not (j in range(cent[1]-(self.k), cent[1]+(self.k)+1)):
                        x.append(i)
                        y.append(j)
        return x,y

    #One Alien, One Crew 

    def one_one_alien_beep_update(self, beep):
        #Beep is boolean, whether or not aliens were detected

        #indices within detection square based on current location of the bot
        x_in,y_in = self.get_det_sq_indicies()
        x_out, y_out = self.get_out_det_sq_indicies()

        if beep:
            #set probabilities outside the det sq to 0
            self.alien_probs[x_out, y_out] = 0
            
        else:
            #set probabilities inside the det sq to 0
            self.alien_probs[x_in,y_in] = 0
            
        #I think normalization is the same regardless since everything else went to 0
        alien_norm_factor = np.sum(self.get_alien_probs())
        self.alien_probs /= alien_norm_factor
            
    
    def one_one_crew_beep_update(self, beep):
        #This is all wrong
        #Beep is boolean
        bot_row = self.bot_loc[0]
        bot_col = self.bot_loc[1]


        if beep:
            #prob_function = lambda x: self.bot.get_beep_prob(x.row, x.col)
            prob_function = lambda x: self.bot.get_beep_prob(x.row, x.col)
            probs = np.asarray([list(map(prob_function, row)) for row in self.ship])

            
        else:
            prob_function = lambda x: 1 - self.bot.get_beep_prob(x.row, x.col)
            probs = np.asarray([list(map(prob_function, row)) for row in self.ship])
    
        #Denominator
        sum_array = copy.deepcopy(self.crew_probs)
        sum_array = np.multiply(sum_array, probs)
        sum = np.sum(sum_array)
        #Numerator
        self.crew_probs = np.multiply(self.crew_probs, probs)

        #Fraction
        self.crew_probs /= sum


        

    def one_one_bot_move_update(self):
        #This function applies when no alien or crew member was in the square we moved to

        bot_loc = self.get_bot_loc()

        #We know for sure no crew or alien in this square
        self.set_crew_probs(bot_loc[0], bot_loc[1], 0)
        self.set_alien_probs(bot_loc[0], bot_loc[1], 0)

        #Normalize the rest of the values

        crew_norm_factor = np.sum(self.crew_probs)
        alien_norm_factor = np.sum(self.alien_probs)

        self.crew_probs /= crew_norm_factor
        self.alien_probs /= alien_norm_factor


    def one_one_alien_move_update(self):
        for i in range(self.D):
            for j in range(self.D):
                p = 0
                
                if self.ship[i][j].is_open():
                    #loop through adjacent cells
                    if (i-1) > 0:
                        p += ((self.alien_probs[i-1][j]) * (1/ self.open_neighbors[i-1][j]))
                    if (i +1) < self.D -1:
                        p += ((self.alien_probs[i+1][j]) * (1/ self.open_neighbors[i+1][j]))
                    if(j-1) > 0:
                        p += ((self.alien_probs[i][j-1]) * (1/ self.open_neighbors[i][j-1]))
                    if(j+1) < self.D-1:
                        p += ((self.alien_probs[i][j+1]) * (1/ self.open_neighbors[i][j+1]))
                    self.set_alien_probs(i,j,p)

    #One Alien, Two Crew 

    def one_two_crew_beep_update(self):
        pass

    def one_two_bot_move_update(self):
        #This function applies when no alien or crew member was in the square we moved to

        bot_loc = self.get_bot_loc()

        #We know for sure alien in this square
        self.set_alien_probs(bot_loc[0], bot_loc[1], 0)

        #We know for sure no crew member in this square -> any pair containing this square has p = 0
        


        #Normalize the rest of the values
        crew_norm_factor = np.sum(self.crew_pair_probs())
        alien_norm_factor = np.sum(self.alien_probs())

        self.crew__pair_probs /= crew_norm_factor
        self.alien_probs /= alien_norm_factor

    #Two Alien, Two Crew

    def two_two_alien_beep_update(self, beep):
        #Beep is boolean, whether or not aliens were detected

        #indices within detection square based on current location of the bot
        x,y = self.get_det_sq_indicies()

        if beep:
            #set probabilities outside the det sq to 0
            #for all the pairs, if both are outside the detection square, probability = 0

            pass
            
        else:
            #for all pairs, if either is inside the detection square, probability = 0
            pass
            
        #I think normalization is the same regardless since everything else went to 0
        alien_norm_factor = np.sum(self.get_alien__pair_probs())
        self.alien__pair_probs /= alien_norm_factor

    def two_two_alien_move_update(self):
        pass

    def two_two_bot_move_update(self):
        #This function applies when no alien or crew member was in the square we moved to

        bot_loc = self.get_bot_loc()


        #We know for sure no crew member or alien in this square -> any pair containing this square has p = 0
        


        #Normalize the rest of the values
        crew_norm_factor = np.sum(self.crew_pair_probs())
        alien_norm_factor = np.sum(self.alien_pair_probs())

        self.crew__pair_probs /= crew_norm_factor
        self.alien__pair_probs /= alien_norm_factor