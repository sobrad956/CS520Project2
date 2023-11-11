import random
import math
import numpy as np
from queue import Queue
import copy
from itertools import product
from itertools import combinations, combinations_with_replacement
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
        self.distances = np.asarray([[-1 for j in range(self.d)] for i in range(self.d)])
        self.open_n = []

    def set_distance(self, row, col, dist):
        self.distances[row][col] = dist

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
    
    def __init__(self, k, D):
        self.D = D  # The dimension of the ship as a square
        self.ship = np.asarray([[Cell(i, j, self.D) for j in range(self.D)] for i in range(self.D)])  # creates a DxD 2D grid of closed cells
        self.bot_loc = [-1, -1]  # Stores the initial position of the bot, used to restrict alien generation cells
        self.crew_probs = np.asarray([[0.0 for j in range(self.D)] for i in range(self.D)])
        self.alien_probs = np.asarray([[0.0 for j in range(self.D)] for i in range(self.D)])
        self.k = k
        self.bot = None
        self.num_open_cells = None
        self.open_neighbors = np.asarray([[0.0 for j in range(self.D)] for i in range(self.D)])
        self.two_crew_prob = np.asarray([[[[0.0 for j in range(self.D)] for i in range(self.D)] for k in range(self.D)] for l in range(self.D)])
        self.two_alien_prob = np.asarray([[[[0.0 for j in range(self.D)] for i in range(self.D)] for k in range(self.D)] for l in range(self.D)])
        self.open_cell_mask = np.asarray([[False for j in range(self.D)] for i in range(self.D)])

    def get_crew_probs(self):
        return self.crew_probs

    def get_two_crew_probs(self):
        return self.two_crew_prob
    
    def get_alien_probs(self):
        return self.alien_probs
    
    def get_two_alien_probs(self):
        return self.two_alien_prob

    def set_crew_probs(self, i, j, p):
        self.crew_probs[i][j] = p

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

    def calculate_open_cells(self):
        for row in range(self.D):
            for col in range(self.D):
                if self.ship[row][col].is_open():
                    self.open_cell_mask[row][col] = True
                    if(row > 0):
                        if self.ship[row-1][col].is_open():
                            self.ship[row][col].open_n.append((row-1, col))
                    if(row < self.D -1):
                        if self.ship[row+1][col].is_open():
                            self.ship[row][col].open_n.append((row+1,col))
                    if(col > 0):
                        if self.ship[row][col-1].is_open():
                            self.ship[row][col].open_n.append((row,col-1))
                    if(col < self.D -1):
                        if self.ship[row][col+1].is_open():
                            self.ship[row][col+1].open_n.append((row, col+1))
                    

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
        self.calculate_open_cells()

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


    def init_crew_prob_one(self):
        p = 1 / (self.num_open_cells - 1)
        mask_func = lambda x: x.is_open() and not x.contains_bot()
        mask = np.asarray([list(map(mask_func, row)) for row in self.ship])
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


    def get_det_sq_indicies(self): 
        r1 = range(max(self.bot.row - self.k, 0), min(self.bot.row + self.k + 1, self.D))
        r2 = range(max(self.bot.col - self.k, 0),min(self.bot.col + self.k + 1, self.D))
        r3 = list(product(r1, r2))
        return [b[0] for b in r3], [b[1] for b in r3]


    # #Probability updates
    
    def get_out_det_sq_indicies(self):

        cent = (self.bot.row, self.bot.col)
        x = []
        y = []

        for i in range(self.D):
            for j in range(self.D):
                if not ((i in range(cent[0]-(self.k), cent[0]+(self.k)+1)) and (j in range(cent[1]-(self.k), cent[1]+(self.k)+1))):
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

        np.around(self.alien_probs, decimals=10)
            
    
    def one_one_crew_beep_update(self, beep):
    
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

        #Avoids negative numbers from floating point error
        np.clip(self.crew_probs, a_min = 0,a_max =1, out = self.crew_probs)


    def one_one_bot_move_update(self):
        #This function applies when no alien or crew member was in the square we moved to

        bot_row = self.bot.get_row()
        bot_col = self.bot.get_col()

        #We know for sure no crew or alien in this square
        self.set_crew_probs(bot_row, bot_col, 0)
        self.set_alien_probs(bot_row, bot_col, 0)

        #Normalize the rest of the values

        crew_norm_factor = np.sum(self.crew_probs)
        alien_norm_factor = np.sum(self.alien_probs)

        self.crew_probs /= crew_norm_factor
        self.alien_probs /= alien_norm_factor
        np.around(self.alien_probs, decimals=10)


    def one_one_alien_move_update(self):
        for i in range(self.D):
            for j in range(self.D):
                p = 0
                
                if self.ship[i][j].is_open():
                    #loop through adjacent cells
                    if (i-1) >= 0:
                        if self.ship[i-1,j].is_open():
                            p += ((self.alien_probs[i-1][j]) * (1/ self.open_neighbors[i-1][j]))
                    if (i +1) <= self.D -1:
                        if self.ship[i+1,j].is_open():
                            p += ((self.alien_probs[i+1][j]) * (1/ self.open_neighbors[i+1][j]))
                    if(j-1) >= 0:
                        if self.ship[i,j-1].is_open():
                            p += ((self.alien_probs[i][j-1]) * (1/ self.open_neighbors[i][j-1]))
                    if(j+1) <= self.D-1:
                        if self.ship[i,j+1].is_open():
                            p += ((self.alien_probs[i][j+1]) * (1/ self.open_neighbors[i][j+1]))
                    self.set_alien_probs(i,j,p)
                    np.around(self.alien_probs, decimals=10)

    #One Alien, Two Crew 

    def open_cell_indices(self, repeat):
        mask_func = lambda x: x.is_open() and not x.contains_bot()
        mask = np.asarray([list(map(mask_func, row)) for row in self.ship])
        mask_trues = np.array(np.where(mask==True)).T
        if repeat:
            coords = list(combinations_with_replacement(mask_trues, 2))
        else:
            coords = list(combinations(mask_trues, 2))
        return coords

    def init_crew_prob_two(self):
        p = 1/(math.comb(self.num_open_cells, 2) - (self.num_open_cells - 1))
        coords = self.open_cell_indices(False)
        for pair in coords:
            self.two_crew_prob[pair[0][0], pair[0][1], pair[1][0], pair[1][1]] = p
        print(p)
        print("sum of crew pair probs", np.sum(self.two_crew_prob))


    def one_two_crew_beep_update(self, beep): 
        if beep:
            coords = self.open_cell_indices(False)
            probs = [self.bot.get_beep_prob_two(x[0][0],x[0][1], x[1][0], x[0][1]) * self.two_crew_prob[x[0][0],x[0][1], x[1][0], x[0][1]] for x in coords]
        else:
            coords = self.open_cell_indices(False)
            probs = [self.bot.get_beep_prob_two(x[0][0],x[0][1], x[1][0], x[0][1]) * (1 - self.two_crew_prob[x[0][0],x[0][1], x[1][0], x[0][1]]) for x in coords]
        sum = np.sum(probs)
        p = probs/sum
        assignment = zip(p, coords)
        for pairs in assignment:
            self.two_crew_prob[pairs[1][0][0], pairs[1][0][1], pairs[1][1][0], pairs[1][1][1]] = pairs[0]
        #Avoids negative numbers from floating point error
        np.clip(self.crew_probs, a_min = 0,a_max =1, out = self.crew_probs)

    def one_two_bot_move_update(self):
        #This function applies when no alien or crew member was in the square we moved to

        bot_row = self.bot.get_row()
        bot_col = self.bot.get_col()

        #We know for sure alien in this square
        self.set_alien_probs(bot_row, bot_col, 0)

        #We know for sure no crew member in this square -> any pair containing this square has p = 0

        self.two_crew_prob[bot_row, bot_col] = 0
        self.two_crew_prob[:,:,bot_row, bot_col] = 0
        
        #Normalize the rest of the values
        crew_norm_factor = np.sum(self.two_crew_prob)
        alien_norm_factor = np.sum(self.alien_probs)

        self.two_crew_prob /= crew_norm_factor
        self.alien_probs /= alien_norm_factor



    #Two Alien, Two Crew

    def init_alien_prob_two(self):
        #Sets pairs containing cell inside detection square to 0
        coords = self.open_cell_indices(True)
        for coord in coords:
            self.two_alien_prob[coord[0], coord[1]] = 0
            self.two_alien_prob[:,:,coord[0],coord[1]] = 0

        #Counts the number of open cells outside the detection square
        det_sq = self.get_sensor_region(self.bot_loc[0], self.bot_loc[1])
        count = 0
        for row in det_sq:
            for elem in row:
                if elem.is_open():
                    count += 1
        num_open_out = self.num_open_cells - count

        #Normalized probability for each open cell outside detection square

        p = 1 / math.comb(num_open_out+1,2)
        #p = 1 / ( (math.comb(self.num_open_cells+1,2) ) - (count*count + count*num_open_out))
        #p = 1/ (num_open_out*num_open_out)

        cent = self.bot_loc
        coords = self.open_cell_indices(True)
        asdf = 0
        cent = self.get_bot_loc()
        for pair in coords:
            i = pair[0][0]
            j = pair[0][1]

            m = pair[1][0]
            n = pair[1][1]

            if not ((i in range(cent[0]-(self.k), cent[0]+(self.k)+1)) and (j in range(cent[1]-(self.k), cent[1]+(self.k)+1))):
                if not ((m in range(cent[0]-(self.k), cent[0]+(self.k)+1)) and (n in range(cent[1]-(self.k), cent[1]+(self.k)+1))):
                    self.two_alien_prob[pair[0][0], pair[0][1], pair[1][0], pair[1][1]] = p
                    asdf += 1

        print("times in loop = ", asdf)
        print("total number of pairs = ", math.comb(self.num_open_cells+1,2))
        print("number open squares = ", self.num_open_cells)
        print("number open cells inside detection square = ", count)
        print("number open cells outside detection square = ", num_open_out)
        print(p)
        print("sum of alien pair probs", np.sum(self.two_alien_prob))


    def two_two_alien_beep_update(self, beep):
        #Beep is boolean, whether or not aliens were detected

        #indices within detection square based on current location of the bot
        coords = coords = self.open_cell_indices(True)

        if beep:
            total_sum = np.sum(self.two_alien_prob)
            cent = (self.bot.get_row(), self.bot.get_col())
            for pair in coords:
                if not ((pair[0][0] in range(cent[0]-(self.k), cent[0]+(self.k)+1)) and (pair[0][1] in range(cent[1]-(self.k), cent[1]+(self.k)+1))):
                    if not ((pair[1][0] in range(cent[0]-(self.k), cent[0]+(self.k)+1)) and (pair[1][1] in range(cent[1]-(self.k), cent[1]+(self.k)+1))):
                        total_sum -= self.two_alien_prob[pair[0][0], pair[0][1], pair[1][0], pair[1][1]]
                        total_sum -= self.two_alien_prob[pair[1][0], pair[1][1],pair[0][0], pair[0][1]]
                        self.two_alien_prob[pair[0][0], pair[0][1], pair[1][0], pair[1][1]] = 0
                        self.two_alien_prob[pair[1][0], pair[1][1],pair[0][0], pair[0][1]] = 0

                else:
                    self.two_alien_prob /= total_sum    
        else:
            total_sum = np.sum(self.two_alien_prob)
            cent = (self.bot.get_row(), self.bot.get_col())
            for pair in coords:
                if not ((pair[0][0] in range(cent[0]-(self.k), cent[0]+(self.k)+1)) and (pair[0][1] in range(cent[1]-(self.k), cent[1]+(self.k)+1))) or not ((pair[1][0] in range(cent[0]-(self.k), cent[0]+(self.k)+1)) and (pair[1][1] in range(cent[1]-(self.k), cent[1]+(self.k)+1))):
                    total_sum -= self.two_alien_prob[pair[0][0], pair[0][1], pair[1][0], pair[1][1]]
                    total_sum -= self.two_alien_prob[pair[1][0], pair[1][1],pair[0][0], pair[0][1]]
                    self.two_alien_prob[pair[0][0], pair[0][1], pair[1][0], pair[1][1]] = 0
                    self.two_alien_prob[pair[1][0], pair[1][1],pair[0][0], pair[0][1]] = 0
                else:
                    self.two_alien_prob /= total_sum 

        alien_norm_factor = np.sum(self.get_two_alien_probs())
        self.two_alien_prob /= alien_norm_factor




        open_cells = np.array(np.where(self.open_cell_mask == True)).T
        Js = list(combinations_with_replacement(open_cells, 2))
        for pairs in Js:
            j1 = [pairs[0][0], pairs[0][1]]
            j2 = [pairs[1][0], pairs[1][1]]
            Ks = np.asarray(list(product(self.ship[j1[0], j1[1]].open_n, self.ship[j2[0], j2[1]].open_n)))
            #init neighbor pair arrys
            self.neighbor_pair_array[j1[0], j1[1], j2[0], j2[1]] = Ks
        

    def two_two_alien_move_update(self):
        p = 0
        for adjacent_pairs in neighbor_pair_array:
            j1 = [adjacent_pairs[0][0][0], adjacent_pairs[0][0][1]]
            j2 = [adjacent_pairs[1][0][0], adjacent_pairs[1][0][1]]
            k1 = [adjacent_pairs[2][0][0], adjacent_pairs[2][0][1]]
            k2 = [adjacent_pairs[3][1][0], adjacent_pairs[3[1][1]]]
            p+= self.two_alien_prob[k1[0], k1[1], k2[0], k2[1]] * (1/self.open_neighbors[k1[0], k1[1]]) * (1/self.open_neighbors[k2[0], k2[1]])
            self.two_alien_prob[j1[0], j2[0], j1[0], j2[0]] = p
            self.two_alien_prob[j2[0], j2[1],j1[0], j1[1]] = p
        
        p = 0
        for pairs in Js:
            j1 = [pairs[0][0], pairs[0][1]]
            j2 = [pairs[1][0], pairs[1][1]]
            Ks = np.asarray(list(product(self.ship[j1[0], j1[1]].open_n, self.ship[j2[0], j2[1]].open_n)))
            for pair in Ks:
                k1 = [pair[0][0], pair[0][1]]
                k2 = [pair[1][0], pair[1][1]]
                p+= self.two_alien_prob[k1[0], k1[1], k2[0], k2[1]] * (1/self.open_neighbors[k1[0], k1[1]]) * (1/self.open_neighbors[k2[0], k2[1]])
            self.two_alien_prob[j1[0], j2[0], j1[0], j2[0]] = p
            self.two_alien_prob[j2[0], j2[1],j1[0], j1[1]] = p

        
        

    def two_two_bot_move_update(self):
        #This function applies when no alien or crew member was in the square we moved to

        bot_row = self.bot.get_row()
        bot_col = self.bot.get_col()

        #We know for sure no crew member/ alien in this square -> any pair containing this square has p = 0

        self.two_crew_prob[bot_row, bot_col] = 0
        self.two_crew_prob[:,:,bot_row, bot_col] = 0

        self.two_alien_prob[bot_row, bot_col] = 0
        self.two_alien_prob[:,:,bot_row, bot_col] = 0
        
        #Normalize the rest of the values
        crew_norm_factor = np.sum(self.two_crew_prob)
        alien_norm_factor = np.sum(self.two_alien_prob)

        self.two_crew_prob /= crew_norm_factor
        self.two_alien_prob /= alien_norm_factor