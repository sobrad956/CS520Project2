import math
import numpy as np
import random


class Bot:
    def __init__(self, row, col, k, ship, type, alpha):
        self.row = row
        self.col = col
        self.k = k
        self.ship = ship
        self.type = type
        self.ship.ship[self.row][self.col].add_bot()
        self.ship.set_bot_loc(self.row, self.col)
        self.alpha = alpha

    def move_up(self):
        self.ship.ship[self.row][self.col].remove_bot()
        self.row -= 1
        self.ship.ship[self.row][self.col].add_bot()
        if self.ship.ship[self.row][self.col].contains_alien():
            return ("Mission Failed, Bot Captured")

    def move_down(self):
        self.ship.ship[self.row][self.col].remove_bot()
        self.row += 1
        self.ship.ship[self.row][self.col].add_bot()
        if self.ship.ship[self.row][self.col].contains_alien():
            return ("Mission Failed, Bot Captured")

    def move_right(self):
        self.ship.ship[self.row][self.col].remove_bot()
        self.col += 1
        self.ship.ship[self.row][self.col].add_bot()
        if self.ship.ship[self.row][self.col].contains_alien():
            return ("Mission Failed, Bot Captured")

    def move_left(self):
        self.ship.ship[self.row][self.col].remove_bot()
        self.col -= 1
        self.ship.ship[self.row][self.col].add_bot()
        if self.ship.ship[self.row][self.col].contains_alien():
            return ("Mission Failed, Bot Captured")

    def found_crew(self):
        if self.ship.ship[self.row][self.col].contains_crew():
            self.ship.ship[self.row][self.col].remove_crew()
            return True

    def get_row(self):
        return self.row

    def get_col(self):
        return self.col

    def get_type(self):
        return self.type

    def get_sensor_region(self, i, j):
        """ Returns the cells within the alien sensor region """
        return self.ship.get_sensor_region(self.row, self.col)

    def detect_alien(self):
        """ Returns True if an alien is in the sensor region, False otherwise """
        region = self.get_sensor_region(self.row, self.col)
        for r in range(len(region)):
            for c in range(len(region[0])):
                if region[r][c].contains_alien():
                    return True
        return False

    def get_beep_prob(self, row, col):
        #crewnum is the crew number's index
        d = self.ship.ship[row][col].distances[self.row][self.col]
        prob = math.exp(-self.alpha * (d - 1))
        #print(prob)
        return prob
    
    def get_beep_prob_two(self, row1, col1, row2, col2):
        #crewnum is the crew number's index
        d1 = self.ship.ship[row1][col1].distances[self.row][self.col]
        d2 = self.ship.ship[row2][col2].distances[self.row][self.col]
        prob = math.exp(-self.alpha * (d1 - 1)) + math.exp(-self.alpha * (d2 - 1)) - (math.exp(-self.alpha * (d1 - 1)) *  math.exp(-self.alpha * (d2 - 1)))
        return prob
    


    def detect_crew(self, crew_locs):
        """ Returns a beep with probability for each crew member based on distance """
        for i in range(len(crew_locs)):
            prob = self.get_beep_prob(crew_locs[i].row, crew_locs[i].col)
            if random.random() < prob:
                return True
        return False


    def find_mult_max(self, values):
        mult_max = []
        idx = 0
        max_val = max(values)
        for val in values:
            if val == max_val:
                mult_max.append(idx)
            idx += 1
        return mult_max

    def bot1_move(self):
        next_move = 10
        cur_row = self.row
        cur_col = self.col

        #print(self.ship.get_alien_probs())

        if cur_row > 0 and self.ship.ship[cur_row-1][cur_col].is_open():
            up_crew_prob = self.ship.get_crew_probs()[cur_row-1][cur_col]
            up_alien_prob = self.ship.get_alien_probs()[cur_row-1][cur_col]
        else:
            up_crew_prob = -1
            up_alien_prob = 100
        
        if cur_row < self.ship.D - 1 and self.ship.ship[cur_row+1][cur_col].is_open():
            down_crew_prob = self.ship.get_crew_probs()[cur_row+1][cur_col]
            down_alien_prob = self.ship.get_alien_probs()[cur_row+1][cur_col]
        else:
            down_crew_prob = -1
            down_alien_prob = 100

        if cur_col > 0 and self.ship.ship[cur_row][cur_col-1].is_open():
            left_crew_prob = self.ship.get_crew_probs()[cur_row][cur_col-1]
            left_alien_prob = self.ship.get_alien_probs()[cur_row][cur_col-1]
        else:
            left_crew_prob = -1
            left_alien_prob = 100
        
        if cur_col < self.ship.D-1 and self.ship.ship[cur_row][cur_col+1].is_open():
            right_crew_prob = self.ship.get_crew_probs()[cur_row][cur_col+1]
            right_alien_prob = self.ship.get_alien_probs()[cur_row][cur_col+1]
        else:
            right_crew_prob = -1
            right_alien_prob = 100

        crew_probs = [left_crew_prob, right_crew_prob, up_crew_prob, down_crew_prob]
        alien_probs = [left_alien_prob, right_alien_prob, up_alien_prob, down_alien_prob]

        mult_max = self.find_mult_max(crew_probs)
        max_idx = random.choice(mult_max)
        
        while(next_move == 10):
            if 0 not in alien_probs:
                #print("no safe move")
                #print(self.ship.get_alien_probs())
                next_move = 100
            elif alien_probs[max_idx] == 0:
                next_move = max_idx
            else:
                crew_probs[max_idx] = -1
                mult_max = self.find_mult_max(crew_probs)
                max_idx = random.choice(mult_max)
        if next_move == 0:
            self.move_left()
        elif next_move == 1:
            self.move_right()
        elif next_move == 2:
            self.move_up()
        elif next_move == 3:
            self.move_down()
        else:
            pass  # Do nothing

    def bot2_move(self):

        next_move = 10
        cur_row = self.row
        cur_col = self.col

        if cur_row > 0 and self.ship.ship[cur_row - 1][cur_col].is_open():
            up_crew_prob = self.ship.get_crew_probs()[cur_row - 1][cur_col]
            up_alien_prob = self.ship.get_alien_probs()[cur_row - 1][cur_col]
        else:
            up_crew_prob = -1
            up_alien_prob = 100

        if cur_row < self.ship.D - 1 and self.ship.ship[cur_row + 1][cur_col].is_open():
            down_crew_prob = self.ship.get_crew_probs()[cur_row + 1][cur_col]
            down_alien_prob = self.ship.get_alien_probs()[cur_row + 1][cur_col]
        else:
            down_crew_prob = -1
            down_alien_prob = 100

        if cur_col > 0 and self.ship.ship[cur_row][cur_col - 1].is_open():
            left_crew_prob = self.ship.get_crew_probs()[cur_row][cur_col - 1]
            left_alien_prob = self.ship.get_alien_probs()[cur_row][cur_col - 1]
        else:
            left_crew_prob = -1
            left_alien_prob = 100

        if cur_col < self.ship.D - 1 and self.ship.ship[cur_row][cur_col + 1].is_open():
            right_crew_prob = self.ship.get_crew_probs()[cur_row][cur_col + 1]
            right_alien_prob = self.ship.get_alien_probs()[cur_row][cur_col + 1]
        else:
            right_crew_prob = -1
            right_alien_prob = 100

        crew_probs = [left_crew_prob, right_crew_prob, up_crew_prob, down_crew_prob]
        alien_probs = [left_alien_prob, right_alien_prob, up_alien_prob, down_alien_prob]

        #Our utility of the move is the difference between the crew probability and the alien prob. 
        util = np.subtract(crew_probs, alien_probs)

        mult_max = self.find_mult_max(util)
        max_idx = random.choice(mult_max)

        next_move = max_idx
        if next_move == 0:
            self.move_left()
        elif next_move == 1:
            self.move_right()
        elif next_move == 2:
            self.move_up()
        elif next_move == 3:
            self.move_down()
        else:
            pass  # Do nothing


    def bot4_move(self):    
        next_move = 10
        cur_row = self.row
        cur_col = self.col

        #print(self.ship.get_alien_probs())

        if cur_row > 0 and self.ship.ship[cur_row-1][cur_col].is_open():
            up_crew_prob = np.sum(self.ship.get_two_crew_probs()[cur_row-1,cur_col]) + np.sum(self.ship.get_two_crew_probs()[:,:,cur_row-1,cur_col])
            up_alien_prob = self.ship.get_alien_probs()[cur_row-1][cur_col]
        else:
            up_crew_prob = -1
            up_alien_prob = 100
        
        if cur_row < self.ship.D - 1 and self.ship.ship[cur_row+1][cur_col].is_open():
            down_crew_prob = np.sum(self.ship.get_two_crew_probs()[cur_row+1,cur_col]) + np.sum(self.ship.get_two_crew_probs()[:,:,cur_row+1,cur_col])
            down_alien_prob = self.ship.get_alien_probs()[cur_row+1][cur_col]
        else:
            down_crew_prob = -1
            down_alien_prob = 100

        if cur_col > 0 and self.ship.ship[cur_row][cur_col-1].is_open():
            left_crew_prob = np.sum(self.ship.get_two_crew_probs()[cur_row,cur_col-1]) + np.sum(self.ship.get_two_crew_probs()[:,:,cur_row,cur_col-1])
            left_alien_prob = self.ship.get_alien_probs()[cur_row][cur_col-1]
        else:
            left_crew_prob = -1
            left_alien_prob = 100
        
        if cur_col < self.ship.D-1 and self.ship.ship[cur_row][cur_col+1].is_open():
            right_crew_prob = np.sum(self.ship.get_two_crew_probs()[cur_row,cur_col+1]) + np.sum(self.ship.get_two_crew_probs()[:,:,cur_row,cur_col+1])
            right_alien_prob = self.ship.get_alien_probs()[cur_row][cur_col+1]
        else:
            right_crew_prob = -1
            right_alien_prob = 100

        crew_probs = [left_crew_prob, right_crew_prob, up_crew_prob, down_crew_prob]
        alien_probs = [left_alien_prob, right_alien_prob, up_alien_prob, down_alien_prob]

        mult_max = self.find_mult_max(crew_probs)
        max_idx = random.choice(mult_max)
        
        while(next_move == 10):
            if 0 not in alien_probs:
                next_move = 100
            elif alien_probs[max_idx] == 0:
                next_move = max_idx
            else:
                crew_probs[max_idx] = -1
                mult_max = self.find_mult_max(crew_probs)
                max_idx = random.choice(mult_max)
        if next_move == 0:
            self.move_left()
        elif next_move == 1:
            self.move_right()
        elif next_move == 2:
            self.move_up()
        elif next_move == 3:
            self.move_down()
        else:
            pass  # Do nothing

    def bot5_move(self):    
        next_move = 10
        cur_row = self.row
        cur_col = self.col

        #print(self.ship.get_alien_probs())

        if cur_row > 0 and self.ship.ship[cur_row-1][cur_col].is_open():
            up_crew_prob = np.sum(self.ship.get_two_crew_probs()[cur_row-1,cur_col]) + np.sum(self.ship.get_two_crew_probs()[:,:,cur_row-1,cur_col])
            up_alien_prob = self.ship.get_alien_probs()[cur_row-1][cur_col]
        else:
            up_crew_prob = -1
            up_alien_prob = 100
        
        if cur_row < self.ship.D - 1 and self.ship.ship[cur_row+1][cur_col].is_open():
            down_crew_prob = np.sum(self.ship.get_two_crew_probs()[cur_row+1,cur_col]) + np.sum(self.ship.get_two_crew_probs()[:,:,cur_row+1,cur_col])
            down_alien_prob = self.ship.get_alien_probs()[cur_row+1][cur_col]
        else:
            down_crew_prob = -1
            down_alien_prob = 100

        if cur_col > 0 and self.ship.ship[cur_row][cur_col-1].is_open():
            left_crew_prob = np.sum(self.ship.get_two_crew_probs()[cur_row,cur_col-1]) + np.sum(self.ship.get_two_crew_probs()[:,:,cur_row,cur_col-1])
            left_alien_prob = self.ship.get_alien_probs()[cur_row][cur_col-1]
        else:
            left_crew_prob = -1
            left_alien_prob = 100
        
        if cur_col < self.ship.D-1 and self.ship.ship[cur_row][cur_col+1].is_open():
            right_crew_prob = np.sum(self.ship.get_two_crew_probs()[cur_row,cur_col+1]) + np.sum(self.ship.get_two_crew_probs()[:,:,cur_row,cur_col+1])
            right_alien_prob = self.ship.get_alien_probs()[cur_row][cur_col+1]
        else:
            right_crew_prob = -1
            right_alien_prob = 100

        crew_probs = [left_crew_prob, right_crew_prob, up_crew_prob, down_crew_prob]
        alien_probs = [left_alien_prob, right_alien_prob, up_alien_prob, down_alien_prob]

        util = np.subtract(crew_probs, alien_probs)

        mult_max = self.find_mult_max(util)
        max_idx = random.choice(mult_max)
        
        next_move = max_idx

        if next_move == 0:
            self.move_left()
        elif next_move == 1:
            self.move_right()
        elif next_move == 2:
            self.move_up()
        elif next_move == 3:
            self.move_down()
        else:
            pass  # Do nothing

    def bot7_move(self):    
        next_move = 10
        cur_row = self.row
        cur_col = self.col

        #print(self.ship.get_alien_probs())

        if cur_row > 0 and self.ship.ship[cur_row-1][cur_col].is_open():
            up_crew_prob = np.sum(self.ship.get_two_crew_probs()[cur_row-1,cur_col]) + np.sum(self.ship.get_two_crew_probs()[:,:,cur_row-1,cur_col])
            up_alien_prob = np.sum(self.ship.get_two_alien_probs()[cur_row-1,cur_col]) + np.sum(self.ship.get_two_alien_probs()[:,:,cur_row-1,cur_col])
        else:
            up_crew_prob = -1
            up_alien_prob = 100
        
        if cur_row < self.ship.D - 1 and self.ship.ship[cur_row+1][cur_col].is_open():
            down_crew_prob = np.sum(self.ship.get_two_crew_probs()[cur_row+1,cur_col]) + np.sum(self.ship.get_two_crew_probs()[:,:,cur_row+1,cur_col])
            down_alien_prob = up_alien_prob = np.sum(self.ship.get_two_alien_probs()[cur_row+1,cur_col]) + np.sum(self.ship.get_two_alien_probs()[:,:,cur_row+1,cur_col])
        else:
            down_crew_prob = -1
            down_alien_prob = 100

        if cur_col > 0 and self.ship.ship[cur_row][cur_col-1].is_open():
            left_crew_prob = np.sum(self.ship.get_two_crew_probs()[cur_row,cur_col-1]) + np.sum(self.ship.get_two_crew_probs()[:,:,cur_row,cur_col-1])
            left_alien_prob = up_alien_prob = np.sum(self.ship.get_two_alien_probs()[cur_row,cur_col-1]) + np.sum(self.ship.get_two_alien_probs()[:,:,cur_row,cur_col-1])
        else:
            left_crew_prob = -1
            left_alien_prob = 100
        
        if cur_col < self.ship.D-1 and self.ship.ship[cur_row][cur_col+1].is_open():
            right_crew_prob = np.sum(self.ship.get_two_crew_probs()[cur_row,cur_col+1]) + np.sum(self.ship.get_two_crew_probs()[:,:,cur_row,cur_col+1])
            right_alien_prob = up_alien_prob = np.sum(self.ship.get_two_alien_probs()[cur_row,cur_col+1]) + np.sum(self.ship.get_two_alien_probs()[:,:,cur_row,cur_col+1])
        else:
            right_crew_prob = -1
            right_alien_prob = 100

        crew_probs = [left_crew_prob, right_crew_prob, up_crew_prob, down_crew_prob]
        alien_probs = [left_alien_prob, right_alien_prob, up_alien_prob, down_alien_prob]

        util = np.subtract(crew_probs, alien_probs)

        mult_max = self.find_mult_max(util)
        max_idx = random.choice(mult_max)
        
        next_move = max_idx

        if next_move == 0:
            self.move_left()
        elif next_move == 1:
            self.move_right()
        elif next_move == 2:
            self.move_up()
        elif next_move == 3:
            self.move_down()
        else:
            pass  # Do nothing

    def bot8_move(self):    
        next_move = 10
        cur_row = self.row
        cur_col = self.col

        #print(self.ship.get_alien_probs())

        if cur_row > 0 and self.ship.ship[cur_row-1][cur_col].is_open():
            up_crew_prob = np.sum(self.ship.get_two_crew_probs()[cur_row-1,cur_col]) + np.sum(self.ship.get_two_crew_probs()[:,:,cur_row-1,cur_col])
            up_alien_prob = np.sum(self.ship.get_two_alien_probs()[cur_row-1,cur_col]) + np.sum(self.ship.get_two_alien_probs()[:,:,cur_row-1,cur_col])
        else:
            up_crew_prob = -1
            up_alien_prob = 100
        
        if cur_row < self.ship.D - 1 and self.ship.ship[cur_row+1][cur_col].is_open():
            down_crew_prob = np.sum(self.ship.get_two_crew_probs()[cur_row+1,cur_col]) + np.sum(self.ship.get_two_crew_probs()[:,:,cur_row+1,cur_col])
            down_alien_prob = up_alien_prob = np.sum(self.ship.get_two_alien_probs()[cur_row+1,cur_col]) + np.sum(self.ship.get_two_alien_probs()[:,:,cur_row+1,cur_col])
        else:
            down_crew_prob = -1
            down_alien_prob = 100

        if cur_col > 0 and self.ship.ship[cur_row][cur_col-1].is_open():
            left_crew_prob = np.sum(self.ship.get_two_crew_probs()[cur_row,cur_col-1]) + np.sum(self.ship.get_two_crew_probs()[:,:,cur_row,cur_col-1])
            left_alien_prob = up_alien_prob = np.sum(self.ship.get_two_alien_probs()[cur_row,cur_col-1]) + np.sum(self.ship.get_two_alien_probs()[:,:,cur_row,cur_col-1])
        else:
            left_crew_prob = -1
            left_alien_prob = 100
        
        if cur_col < self.ship.D-1 and self.ship.ship[cur_row][cur_col+1].is_open():
            right_crew_prob = np.sum(self.ship.get_two_crew_probs()[cur_row,cur_col+1]) + np.sum(self.ship.get_two_crew_probs()[:,:,cur_row,cur_col+1])
            right_alien_prob = up_alien_prob = np.sum(self.ship.get_two_alien_probs()[cur_row,cur_col+1]) + np.sum(self.ship.get_two_alien_probs()[:,:,cur_row,cur_col+1])
        else:
            right_crew_prob = -1
            right_alien_prob = 100

        crew_probs = [left_crew_prob, right_crew_prob, up_crew_prob, down_crew_prob]
        alien_probs = [left_alien_prob, right_alien_prob, up_alien_prob, down_alien_prob]

        util = np.subtract(crew_probs, alien_probs)

        mult_max = self.find_mult_max(util)
        max_idx = random.choice(mult_max)
        
        next_move = max_idx

        if next_move == 0:
            self.move_left()
        elif next_move == 1:
            self.move_right()
        elif next_move == 2:
            self.move_up()
        elif next_move == 3:
            self.move_down()
        else:
            pass  # Do nothing