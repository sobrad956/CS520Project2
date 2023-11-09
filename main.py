import numpy as np
import random
import math
import matplotlib.pyplot as plt

from ship import Ship
from alien import Alien
from bot import Bot


def experiment1(k, alphas):
    numBoards = 30
    numTrials = 10
    bots = [1, 2]
    avg_crew_saved = np.zeros((2, len(alphas)))
    avg_moves_to_save = np.zeros((2, len(alphas)))
    prob_success = np.zeros((2, len(alphas)))

    shp = Ship()
    for board in range(numBoards):
        shp.generate_ship()
        for a, alpha in enumerate(alphas):
            for botnum in bots:
                for trial in range(numTrials):
                    i, j = shp.get_unoccupied_cell()
                    bot = Bot(i, j, k, shp, botnum, alpha)

                    start_cells = []
                    i, j = shp.get_unoccupied_cell()
                    shp.ship[i][j].add_crew()
                    start_cells.append(shp.ship[i][j])

                    i, j = shp.get_unoccupied_alien_cell(k)
                    alien = Alien(i, j, shp)

                    shp.distances_from_crew(start_cells)

                    #Initialize ship probabilities

                    shp.print_ship()
                    print('Board:', board, ' Botnum:', botnum, ' Trial:', trial)
                    T = 0
                    flag = True
                    while flag:
                        aliendetected = bot.detect_alien()
                        crewbeep = bot.detect_crew(1, alpha)

                        #Update probabilities

                        i, j = bot.move()

                        if shp.ship[i][j].contains_alien():
                            print(f"Dead: {T}")
                            avg_moves_to_save[botnum - 1][a] += T / (numBoards * numTrials)
                            flag = False
                            break
                        if shp.ship[i][j].contains_crew():
                            print(f"Saved: {T}")
                            avg_crew_saved[botnum - 1][a] += 1 / (numBoards * numTrials)
                            prob_success[botnum - 1][a] += 1 / (numBoards * numTrials)
                            shp.ship[i][j].remove_crew()
                            flag = False
                            break
                        if alien.move():
                            print(f"Dead: {T}")
                            avg_moves_to_save[botnum - 1][a] += T / (numBoards * numTrials)
                            flag = False
                            break
                        T += 1
                    shp.empty_ship()

    alphas = [str(x) for x in alphas]
    plt.plot(alphas, avg_moves_to_save[0], label='Bot 1')
    plt.plot(alphas, avg_moves_to_save[1], label='Bot 2')
    plt.xlabel('Value for alpha')
    plt.ylabel('Average Number of Moves Needed to Rescue all Crew Members')
    plt.title('Average Number of Moves Needed to Rescue all Crew Members (One Alien, One Crew) vs Alpha')
    plt.legend(loc='best')
    plt.savefig('experiment1_plot1.png')
    np.save('experiment1_plot1.npy', avg_moves_to_save)
    plt.plot()
    plt.close()

    plt.plot(alphas, avg_crew_saved[0], label='Bot 1')
    plt.plot(alphas, avg_crew_saved[1], label='Bot 2')
    plt.xlabel('Value for alpha')
    plt.ylabel('Average Number of Crew Members Saved')
    plt.title('Average Number of Crew Members Saved (One Alien, One Crew) vs Alpha')
    plt.legend(loc='best')
    plt.savefig('experiment1_plot2.png')
    np.save('experiment1_plot2.npy', avg_crew_saved)
    plt.plot()
    plt.close()

    plt.plot(alphas, prob_success[0], label='Bot 1')
    plt.plot(alphas, prob_success[1], label='Bot 2')
    plt.xlabel('Value for alpha')
    plt.ylabel('Probability of Successfully Saving all Crew Members')
    plt.title('Probability of Successfully Saving all Crew Members (One Alien, One Crew) vs Alpha')
    plt.legend(loc='best')
    plt.savefig('experiment1_plot3.png')
    np.save('experiment1_plot3.npy', prob_success)
    plt.plot()
    plt.close()


def experiment2(k, alphas):
    numBoards = 30
    numTrials = 10
    bots = [3, 4, 5]
    avg_crew_saved = np.zeros((2, len(alphas)))
    avg_moves_to_save = np.zeros((2, len(alphas)))
    prob_success = np.zeros((2, len(alphas)))

    shp = Ship()
    for board in range(numBoards):
        shp.empty_ship()
        shp.generate_ship()

        for a, alpha in enumerate(alphas):
            for botnum in bots:
                for trial in range(numTrials):
                    i, j = shp.get_unoccupied_cell(False)
                    bot = Bot(i, j, shp, botnum)

                    crew = []
                    for i in range(2):
                        i, j = shp.get_unoccupied_cell(True)
                        shp.ship[i][j].add_crew()
                        crew.append((i, j))
                        # shp.set_crew_loc(i, j) #DOESNT WORK WITH TWO CREW

                    i, j = shp.get_unoccupied_alien_cell(k)
                    alien = Alien(i, j, shp)

                    shp.print_ship()
                    print('Board:', board, ' Botnum:', botnum, ' Trial:', trial)
                    T = 0
                    saved_counter = 0
                    while saved_counter != 2:
                        # bot moves
                        if shp.ship[i][j].contains_alien():
                            print(f"Dead: {T}")
                            avg_moves_to_save[botnum - 3][board][trial] += T
                            break
                        # THIS NEEDS AN UPDATE TO ACCOUNT FOR THERE BEING TWO CREW
                        if shp.ship[i][j].contains_crew():
                            print(f"Saved: {T}")
                            avg_crew_saved[botnum - 3][board][trial] += 1
                            shp.ship[i][j].remove_crew()
                            i, j = shp.get_unoccupied_cell(True)
                            shp.ship[i][j].add_crew()
                            shp.set_crew_loc(i, j)
                        # alien moves random.shuffle(aliens)
                        if shp.ship[i][j].contains_bot():
                            print(f"Dead: {T}")
                            avg_moves_to_save[botnum - 3][board][trial] += T
                            break
                        # detect alien
                        # detect crew
                        T += 1
                        if T >= 1000:
                            avg_moves_to_save[botnum - 3][board][trial] += T
                    shp.empty_ship()
    print(avg_moves_to_save)
    print()
    print(avg_crew_saved)


def experiment3():
    numBoards = 1
    numTrials = 1
    bots = [6, 7, 8]
    saved_crew_results = np.zeros((2, 1))
    time_survived_results = np.zeros((2, 1))

    shp = Ship()
    for board in range(numBoards):
        shp.empty_ship()
        shp.generate_ship()

        for botnum in bots:
            for trial in range(numTrials):
                i, j = shp.get_unoccupied_cell(False)
                bot = Bot(i, j, shp, botnum)

                i, j = shp.get_unoccupied_cell(True)
                shp.ship[i][j].add_crew()
                shp.set_crew_loc(i, j)

                i, j = shp.get_unoccupied_alien_cell(k)
                alien = Alien(i, j, shp)

                shp.print_ship()
                T = 0
                while T <= 1000:
                    # bot moves
                    if shp.ship[i][j].contains_alien():
                        print(f"Dead: {T}")
                        time_survived_results[botnum][0] += T
                        break
                    if shp.ship[i][j].contains_crew():
                        print(f"Saved: {T}")
                        saved_crew_results[botnum][0] += 1
                        shp.ship[i][j].remove_crew()
                        i, j = shp.get_unoccupied_cell(True)
                        shp.ship[i][j].add_crew()
                        shp.set_crew_loc(i, j)
                    # alien moves random.shuffle(aliens)
                    if shp.ship[i][j].contains_bot():
                        print(f"Dead: {T}")
                        time_survived_results[botnum][0] += T
                        break
                    # detect alien
                    # detect crew
                    T += 1
                    if T >= 1000:
                        time_survived_results[botnum][0] += T
                shp.empty_ship()


def main(k):
    crewnum = 2
    shp = Ship()
    shp.generate_ship()

    i, j = shp.get_unoccupied_cell()
    bot = Bot(i, j, k, shp, 1, 0.1)

    start_cells = []
    for num in range(crewnum):
        i, j = shp.get_unoccupied_cell()
        shp.ship[i][j].add_crew()
        start_cells.append(shp.ship[i][j])

    i, j = shp.get_unoccupied_alien_cell(k)
    alien = Alien(i, j, shp)

    shp.distances_from_crew(start_cells)
    shp.print_ship()
    #print(shp.ship[0][0].distances)
    print("Finished")
    # for a in range(shp.D):
    #     for b in range(shp.D):
    #         #print(shp.ship[a][b].distances)
    #     print(a)


    #print(bot.detect_alien())
    #print(bot.detect_crew(crewnum, 0.1))




#     numBoards = 30
#     numTrials = 10
#     bots = [1, 2]
#     avg_crew_saved = np.zeros((2, len(alphas)))
#     avg_moves_to_save = np.zeros((2, len(alphas)))
#     prob_success = np.zeros((2, len(alphas)))
#
#     shp = Ship()
#     for board in range(numBoards):
#         shp.empty_ship()
#         shp.generate_ship()
#
#         for a, alpha in enumerate(alphas):
#             for botnum in bots:
#                 for trial in range(numTrials):
#                     i, j = shp.get_unoccupied_cell(False)
#                     bot = Bot(i, j, k, shp, botnum)
#
#                     i, j = shp.get_unoccupied_cell(True)
#                     shp.ship[i][j].add_crew()
#                     shp.set_crew_loc(i, j)
#
#                     i, j = shp.get_unoccupied_alien_cell(k)
#                     alien = Alien(i, j, shp)
#
#                     #Initialize ship probabilities
#
#                     shp.print_ship()
#                     print('Board:', board, ' Botnum:', botnum, ' Trial:', trial)
#                     T = 0
#                     flag = True
#                     while flag:
#                         #SENSORS
#                         #aliendetected = bot.detect_alien()
#                         #crewbeep = bot.detect_crew(1)
#                         #Update probabilities
#                         #BOT MOVES
#                         #i, j = bot.move() /need logic to pick highest prob square from ship.crew_probs
#                         if shp.ship[i][j].contains_alien():
#                             print(f"Dead: {T}")
#                             avg_moves_to_save[botnum - 1][a] += T / (numBoards * numTrials)
#                             flag = False
#                             break
#                         if shp.ship[i][j].contains_crew():
#                             print(f"Saved: {T}")
#                             avg_crew_saved[botnum - 1][a] += 1 / (numBoards * numTrials)
#                             prob_success[botnum - 1][a] += 1 / (numBoards * numTrials)
#                             shp.ship[i][j].remove_crew()
#                             flag = False
#                             break
#                         # alien moves
#                         # alien.move()
#                         if shp.ship[i][j].contains_bot():
#                             print(f"Dead: {T}")
#                             avg_moves_to_save[botnum - 1][a] += T / (numBoards * numTrials)
#                             flag = False
#                             break
#                         T += 1
#                     shp.empty_ship()


if __name__ == "__main__":
    main(1)
