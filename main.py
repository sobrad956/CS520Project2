import numpy as np
import random
import math
import matplotlib.pyplot as plt

from ship import Ship
from alien import Alien
from bot import Bot

def main(k, d, botType, numAliens):
    shp = Ship()
    shp.generate_ship()

    aliens = []
    i, j = shp.get_unoccupied_cell(False)
    bot = Bot(i, j, shp, botType)

    i, j = shp.get_unoccupied_cell(True)
    shp.ship[i][j].add_crew()
    shp.set_crew_loc(i, j)


    #WILL ADD THE BOX RESTRICTION IN A BIT
    for alien in range(numAliens):
        i, j = shp.get_unoccupied_cell(False)
        aliens.append(Alien(i, j, shp))
    shp.print_ship()



# def main(k, bot_type):
#     """ This function is used to test the functionality of ship generation """
#     numTrials = 10
#     numBoards = 30
#     numAliens = 20
#     botTypes = 4
#     k = 1
#     saved_crew_results = np.zeros((botTypes, numAliens))
#     time_survived_results = np.zeros((botTypes, numAliens))
#     #saved_crew_results = list()
#     #time_survived_results = list()
#
#     for board in range(numBoards):
#         shp = Ship(20)
#         shp.generate_ship()
#         for botType in range(botTypes):
#             k=1
#             #for k in range(1,numAliens):
#             while(k < numAliens):
#                 for trial in range(numTrials):
#                     print("botType: ", botType+1, "num aliens:", k, "trial number:",  trial)
#                     aliens = []
#                     bot = None
#                     #shp = Ship(20)
#                     #shp.generate_ship()
#
#
#                     for alien in range(k):
#                         i, j = shp.get_unoccupied_cell(False)
#                         aliens.append(Alien(i, j, shp))
#
#                     i, j = shp.get_unoccupied_cell(False)
#                     bot = Bot(i, j, shp, botType+1)
#
#                     i, j = shp.get_unoccupied_cell(True)
#                     shp.ship[i][j].add_crew()
#                     shp.set_crew_loc(i,j)
#
#                     shp.print_ship()
#
#                     #shp.print_ship()
#
#                     T = 0
#                     flag = True
#                     while T <= 1000 and flag is True:
#                         #print(T, flag)
#                         path = bot.bfs(shp.ship[bot.get_row()][bot.get_col()])
#                         #print(path)
#                         #print()
#
#                         path_moves = bot.path_to_move(path)
#
#
#                         for step in path:
#                             if T >= 1000:
#                                 flag = False
#                                 break
#                             path_moves = bot.move(path_moves)
#                             i = bot.row
#                             j = bot.col
#                             if shp.ship[i][j].contains_crew():
#                                 if shp.ship[i][j].contains_alien():
#                                     print(f"Dead: {T}")
#                                     time_survived_results[botType][k] += T
#                                     flag = False
#                                     break
#                                 shp.ship[i][j].remove_crew()
#                                 #print(f"TELEPORT: {T}")
#                                 saved_crew_results[botType][k] += 1
#                                 i, j = shp.get_unoccupied_cell(True)
#                                 shp.ship[i][j].add_crew()
#                                 shp.set_crew_loc(i,j)
#
#                             for alien in aliens:
#                                 alien.move()
#                                 if shp.ship[alien.row][alien.col].contains_bot():
#                                     print(f"DEAD: {T}")
#                                     time_survived_results[botType][k] += T
#                                     flag = False
#                                     path = []
#                                     break
#                             random.shuffle(aliens)
#                             if(botType + 1 == 1 and path == [0]): #Rerun search for bot 1 only if stayed in same place
#                                 path = bot.bfs(shp.ship[bot.get_row()][bot.get_col()])
#                                 path_moves = bot.path_to_move(path)
#                             elif (botType + 1 == 2):
#                                 path = bot.bfs(shp.ship[bot.get_row()][bot.get_col()])
#                                 path_moves = bot.path_to_move(path)
#                             elif (botType + 1 == 3):
#                                 path = bot.mod_bfs(shp.ship[bot.get_row()][bot.get_col()])
#                                 path_moves = bot.path_to_move(path)
#                             elif (botType + 1 == 4):
#                                 path = bot.bot4path(shp.ship[bot.get_row()][bot.get_col()])
#                                 path_moves = bot.path_to_move(path)
#                             #shp.print_ship()
#
#                             T += 1
#                     if T >= 1000:
#                         time_survived_results[botType][k] += T
#                     shp.empty_ship()
#                 saved_crew_results[botType][k] /= (numTrials*numBoards)
#                 time_survived_results[botType][k] /= (numTrials*numBoards)
#                 k += 3
#
#     print(saved_crew_results)
#     print()
#     print(time_survived_results)
#
#     for botType in range(botTypes):
#         plt.plot(saved_crew_results[botType][1::3], label=f'Bot {botType+1}')
#     plt.xlabel('K: number of aliens')
#     plt.ylabel('Average Number of crew members saved')
#     plt.title('Average Number of Crew Members Saved vs Number of Aliens')
#     plt.legend(loc='best')
#     plt.savefig('crewplot.png')
#     np.save('crewdata.npy', saved_crew_results)
#     plt.plot()
#     plt.close()
#
#     for botType in range(botTypes):
#         plt.plot(time_survived_results[botType][1::3], label=f'Bot {botType+1}')
#     plt.xlabel('K: number of aliens')
#     plt.ylabel('Average Time Limit Reached')
#     plt.title('Average Time Limit Reached vs Number of Aliens')
#     plt.legend(loc='best')
#     plt.savefig('timeplot.png')
#     np.save('timedata.npy', time_survived_results)
#     plt.plot()
#     plt.close()


if __name__ == "__main__":
    main(3, 2, 1, 1)