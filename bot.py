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
        return self.ship.get_sensor_region(self.row, self.col, self.k)

    def detect_alien(self):
        """ Returns True if an alien is in the sensor region, False otherwise """
        region = self.get_sensor_region(self.row, self.col)
        for r in range(len(region)):
            for c in range(len(region[0])):
                if region[r][c].contains_alien():
                    return True
        return False

    def get_beep_prob(self, crewnum):
        #crewnum is the crew number's index
        d = self.ship.ship[self.row][self.col].distances[crewnum]
        prob = math.exp(-self.alpha * (d - 1))
        # print(prob)
        return prob

    def detect_crew(self, numcrew):
        """ Returns a beep with probability for each crew member based on distance """
        #numcrew is the total number of crew members
        for i in range(0, numcrew):
            prob = self.get_beep_prob(i, self.alpha)
            if random.random() < prob:
                return True
        return False

    def move(self, move_seq):
        # Replace this to pick adjacent square w/ highest prob
        moves = [1, 2, 3, 4, 5]
        next_move = random.choice(moves)
        if next_move == 1:
            self.move_left()
        elif next_move == 2:
            self.move_right()
        elif next_move == 3:
            self.move_up()
        elif next_move == 4:
            self.move_down()
        else:
            pass  # Do nothing

    # Get square within bounds of ship within detection region:
    #(x[max(i - k, 0):min(i + k + 1, 3), max(j - k, 0):min(j + k + 1, 3)])
    #