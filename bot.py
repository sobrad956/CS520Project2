from queue import Queue
from queue import PriorityQueue
import math
import numpy as np
import random
from collections import deque


class Bot:
    def __init__(self, row, col, k, ship, type):
        self.row = row
        self.col = col
        self.k = k
        self.ship = ship
        self.type = type
        self.ship.ship[self.row][self.col].add_bot()

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
        return self.shp.get_sensor_region(self.row, self.col, self.k)

    def detect_alien(self):
        region = self.ship.get_sensor_region(self.row, self.col, self.k)
        for r in len(region):
            for c in len(region[0]):
                if region[r][c].contains_alien():
                    return True
        return False

    # [REPLACING THIS]
    # def detect_crew(self, numCrew):
    #     if numCrew == 1:
    #         start_cell = [self.row, self.col]
    #         target = self.ship.get_crew_loc
    #
    #         fringe = Queue()
    #         visited = []
    #         cur_state = start_cell
    #         fringe.put(cur_state)
    #
    #         while not fringe.empty():
    #             cur_state = fringe.get()
    #             visited.append(cur_state)
    #             if cur_state.contains_crew():
    #                 print(cur_state.get_location())
    #
    #             children = []
    #             #         cur_row = cur_state.row
    #             #         cur_col = cur_state.col
    #             #
    #             #         if(cur_row != 0):
    #             #             children.append(self.ship.ship[cur_row-1][cur_col])
    #             #         if(cur_row != (self.ship.D -1)):
    #             #             children.append(self.ship.ship[cur_row+1][cur_col])
    #             #         if(cur_col != 0):
    #             #             children.append(self.ship.ship[cur_row][cur_col-1])
    #             #         if(cur_col != (self.ship.D - 1)):
    #             #             children.append(self.ship.ship[cur_row][cur_col+1])
    #             for child in children:
    #         #             if(child.is_open() and (child not in visited)):
    #         #                 #Make sure the first move doesn't contain an alien, otherwise proceed assuming alien will move (per zulip)
    #         #                 if((not child.contains_alien()) or (not cur_state == start_cell)):
    #         #                     fringe.put(child)
    #         #                     prev[child] = cur_state
    #         #     print("f")
    #         #     return [0]
    #
    #     else:
    #         return True
    #
    # class queueNode:
    #     def __init__(self, point, dist):
    #         self.point = point  # Cell coordinates
    #         self.dist = dist  # Cell's distance from the source
    #
    # def check_valid(self, row, col):
    #     return (row >= 0) and (row < 50) and (col >= 0) and (col < 50)
    #
    # def LeeAlgo(self, mat, src, dest):
    #     # Checking if source and destination cell have value 1
    #     if mat[src.x][src.y] != 1 or mat[dest.x][dest.y] != 1:
    #         return -1
    #     visited = [[False for i in range(50)] for j in range(50)]
    #     # Mark the source cell as visited
    #     visited[src.x][src.y] = True
    #     # Create a queue for BFS
    #     q = deque()
    #     # Distance of source cell is 0
    #     s = queueNode(src, 0)
    #     q.append(s)  # Enqueue source cell
    #     # Perform BFS starting from source cell
    #     while q:
    #         curr = q.popleft()  # Dequeue the front cell
    #         # If we have reached the destination cell, return the final distance
    #         point = curr.point
    #         if point.x == dest.x and point.y == dest.y:
    #             return curr.dist
    #
    #             # Otherwise enqueue its adjacent cells with value 1
    #         for i in range(4):
    #             row = point.x + rowNum[i]
    #             col = point.y + colNum[i]
    #
    #             # Enqueue valid adjacent cell that is not visited
    #             if (check_valid(row, col) and mat[row][col] == 1 and not visited[row][col]):
    #                 visited[row][col] = True
    #                 Adjcell = queueNode(Cell(row, col), curr.dist + 1)
    #                 q.append(Adjcell)
    #
    #              # Return -1 if destination cannot be reached
    #     return -1




            #(x[max(i - k, 0):min(i + k + 1, 3), max(j - k, 0):min(j + k + 1, 3)])
    #
    #
    # def path_to_move(self, path):
    #     """Converts the sequence of coordinates to a series of moves"""
    #     #1 = left, 2 = right, 3 = up 4 = down
    #     move_seq = Queue()
    #
    #     prev_row = self.get_row()
    #     prev_col = self.get_col()
    #
    #     if(path == [0]):
    #         return [0]
    #
    #     for coords in path:
    #         next_row = coords[0][0]
    #         next_col = coords[0][1]
    #         #print("prev row", prev_row)
    #         #print("next_row")
    #         if(next_col == prev_col-1):
    #             move_seq.put(1)
    #         elif(next_col == prev_col+1):
    #             move_seq.put(2)
    #         elif(next_row == prev_row-1):
    #             move_seq.put(3)
    #         else:
    #             move_seq.put(4)
    #
    #         prev_row = coords[0][0]
    #         prev_col = coords[0][1]
    #
    #     return(move_seq)
    #
    # def move(self, move_seq):
    #     if(move_seq == [0]):
    #         return(move_seq)
    #
    #     next_move = move_seq.get()
    #     if(next_move == 1):
    #         self.move_left()
    #     elif(next_move == 2):
    #         self.move_right()
    #     elif(next_move == 3):
    #         self.move_up()
    #     else:
    #         self.move_down()
    #     return(move_seq)
