import numpy as np
import random as random
import itertools as itertools
import matplotlib.pyplot as plt
import logging as log
import datetime
import time as tlib

#OPTIMAL SCENARIO
# Had12
# 12
# 1652(OPT)(3, 10, 11, 2, 12, 5, 6, 7, 8, 1, 4, 9)

# Had14
# 14
# 2724(OPT)(8, 13, 10, 5, 12, 11, 2, 14, 3, 6, 7, 1, 9, 4)
#
# Had16
# 16
# 3720(OPT)(9, 4, 16, 1, 7, 8, 6, 14, 15, 11, 12, 10, 5, 3, 2, 13)
#
# Had18
# 18
# 5358(OPT)(8, 15, 16, 6, 7, 18, 14, 11, 1, 10, 12, 5, 3, 13, 2, 17, 9, 4)
#
# Had20
# 20
# 6922(OPT)(8, 15, 16, 14, 19, 6, 7, 17, 1, 12, 10, 11, 5, 20, 2, 3, 4, 9, 18, 13)


#########PROPERTIES##############

ITERATIONS = 1                      #LICZBA ITERACJI
POP_SIZE = 10                       #ROZMIAR POPULACJI
GEN = 50                            #LICZBA POKOLEN
PX = 0.7                            #PRAWDOPODOBIENSTWO KRZYŻOWANIA
PM = 0.6                            #PRAWDOPODOIEŃSTWO MUTACJI
TOUR = 5                            #ROZMIAR TURNIEJU
SAVE_STRONGEST = True           #0 FALSE #1 TRUE
TEST_TWO_SAVE_STRONGEST = True     # TEST TWO VARIATIONS OF SAVE_STORNGEST
SELECTION_METHOD = 1             # 0: ROULETTE 1:TOURNAMENT
METHODS_TO_TEST = 1               # IF different than 1 SELECTION METHOD should be 0
DATAFILEPATH = 'input/had12.dat'    #PATH TO FILE WITH INITIALISE DATA
MUTATION_REPEAT = 2                 #REPEATS OF MUTATION ACTION IF MUTATE
########END OF PROPERTIES###########


#load data from file
input_array = np.loadtxt(DATAFILEPATH, skiprows=1)       #load input (two arrays)
dimension = int(open(DATAFILEPATH, 'r').readline())      #read Dimension writed in first row
flow_matrix = input_array[:dimension, :]                 #spit array - first array is FLOW MATRIX
distance_matrix = input_array[dimension:, ]              #split array - second array is DISTANCE_MATRIX

COMBINATIONS = list(itertools.combinations(range(dimension), 2))

def getInitialiseInfo():
    print("Dimension: {}".format(dimension))
    print("Distance matrix: ")
    print(distance_matrix)
    print("Flowmatrix:")
    print(flow_matrix)


def generateIndividual():
    assigments = np.arange(0, dimension)
    random.shuffle(assigments)
    return assigments


def getDistance(assigments, positions):
    a = 0
    try : a = distance_matrix[assigments.tolist().index(positions[0])][assigments.tolist().index(positions[1])]
    except ValueError:
        print("error")
    return a


def getCost(assigments, positions):
    return 2*getDistance(assigments, positions)*getFlow(positions)


def getFlow(positions):
    return flow_matrix[positions[0]][positions[1]]


def costFunction(assigments, combinations):
    cost = 0
    for pair in combinations:
        cost += getCost(assigments, pair)
    return cost


def initialise(populationSize):
    population = np.zeros((populationSize, dimension))
    for i in range(0, populationSize):
        population[i] = generateIndividual()
    return population


def getCostsVector(population):
    costVector = np.zeros(population.shape[0])
    for i in range(0, population.shape[0]):
        costVector[i] = costFunction(population[i], COMBINATIONS)
    return costVector


def rouletteMethod(population, costVector):
    #costVector = (costVector-min(costVector)+1)*1.5             #SHOULDNT BE DONE LIKE THIS
    sumCost = sum(costVector)
    adjVector = sumCost/costVector
    sumAdj = sum(adjVector)
    probabilityVector = adjVector/sumAdj
    parents = np.zeros(population.shape[0], dtype=int)
    newPopulation = np.zeros((population.shape[0], population.shape[1]), dtype=int)
    for i in range(0, population.shape[0]):
        parents[i] = pickOneRoulette(probabilityVector)
        newPopulation[i] = population[parents[i]]
    return newPopulation


def pickOneRoulette(probabilityVector):
    pick = random.uniform(0, 1)
    sum = 0
    for i in range(0, probabilityVector.shape[0]):
        sum += probabilityVector[i]
        if(sum >= pick):
            return i
    return probabilityVector.shape[0]-1


def tournamentMethod(population, costVector):
    parents = np.zeros(population.shape[0], dtype=int)
    newPopulation = np.zeros((population.shape[0], population.shape[1]), dtype=int)

    for i in range(0, population.shape[0]):
        parents[i] = pickOneTournament(costVector)
        newPopulation[i] = population[parents[i]]

    return newPopulation


def pickOneTournament(costVector):
    tournamentTeam = np.zeros(TOUR, dtype=int)

    for i in range(0, tournamentTeam.shape[0]):
        tournamentTeam[i] = getRandomInvidivual()

    return costVector.tolist().index(min(costVector[tournamentTeam]))


def selection(population):
    costVector = getCostsVector(population)
    if SELECTION_METHOD == 0:
        next_population = rouletteMethod(population, costVector)
    if SELECTION_METHOD == 1:
        next_population = tournamentMethod(population, costVector)
    if(SAVE_STRONGEST == True):
        bestIndividual = population[costVector.tolist().index(min(costVector))]
        newCost = getCostsVector(next_population).tolist()
        worstIndividualIndex = newCost.index(min(newCost))
        next_population[worstIndividualIndex] = bestIndividual
    return next_population


def getRandomInvidivual():
    return random.randint(0, POP_SIZE-1)


def allUnique(x):
     seen = set()
     return not any(i in seen or seen.add(i) for i in x)


def repair(individual):
    if allUnique(individual):
        return individual
    list = individual.tolist()
    counts = np.arange(0, individual.shape[0], dtype=int)
    lackOf = []
    for i in range(0, individual.shape[0]):
        counts[i] = list.count(i)
        if i not in individual:
            lackOf.append(i)
    for i in range(0, individual.shape[0]):
        while(counts[i] >1 ):
            pop = lackOf.pop()
            list[list.index(i)] = pop
            counts[i] -= 1
            continue
    return np.array(list)


def crossoverDiscrete(firstI, secondI):
    for i in range(0, firstI.shape[0]):
        if(random.uniform(0,1) <0.5):
            firstI[i] = secondI[i]
    firstI = repair(firstI)
    return firstI


def crossover(population):
    bestIndividualIndex = -1
    if(SAVE_STRONGEST == True):
        costVector = getCostsVector(population)
        bestIndividualIndex = costVector.tolist().index(min(costVector))
    for i in range(0, population.shape[0]):
        if(random.uniform(0, 1) < PX and i!=bestIndividualIndex):
            j = getRandomInvidivual()
            population[i] = crossoverDiscrete(population[i], population[j])
    return population


def mutate(individual):
    for k in range(0, MUTATION_REPEAT):
        i = random.randint(0, individual.shape[0] - 1)
        j = random.randint(0, individual.shape[0] - 1)
        individual[i], individual[j] = individual[j], individual[i]
    return individual


def mutation(population):
    bestIndividualIndex = -1
    if(SAVE_STRONGEST == True):
        costVector = getCostsVector(population)
        bestIndividualIndex = costVector.tolist().index(min(costVector))
    for i in range(0, population.shape[0]):
        if (random.uniform(0, 1) < PM and i != bestIndividualIndex):
            mutate(population[i])
    return population


def getBestIndividual(population):
    costVector = getCostsVector(population)
    bestIndividualIndex = costVector.tolist().index(min(costVector))
    return population[bestIndividualIndex]

def preparePlot(field, minOutput, maxOutput, avgOutput, globalMinOutput):
    plt.figure()
    plt.plot(field, minOutput)
    plt.plot(field, maxOutput)
    plt.plot(field, avgOutput)
    plt.plot(field, globalMinOutput)
    plt.suptitle('COST: {}, \n Params: PX: {}, PM: {} GEN: {}'.format("{} - global min 1652".format(min(minOutput)), PX, PM, GEN))
    plt.legend(("min", "max", "avg", "global min"))
    plt.ylabel("Cost")
    if SELECTION_METHOD == 0:
        plt.xlabel("ITERATIONS \n METHOD: ROULETTE SAVE_STRONEGEST: {}".format(SAVE_STRONGEST))
    else:
        plt.xlabel("ITERATIONS \n METHOD: TOURNAMENT TOUR:{} SAVE_STRONEGEST: {}".format(TOUR, SAVE_STRONGEST))


def geneticAlgorithm():
    starttime = tlib.time()
    pop = initialise(POP_SIZE)
    minOutput = []
    avgOutput = []
    maxOutput = []
    globalMinOutput = []
    minOutput.append(min(getCostsVector(pop)))
    maxOutput.append(max(getCostsVector(pop)))
    avgOutput.append(np.average(getCostsVector(pop)))
    globalMinOutput.append(min(minOutput))
    field = np.arange(0, GEN+1)

    for i in range(0, GEN):
        pop = selection(pop)
        pop = crossover(pop)
        pop = mutation(pop)
        minOutput.append(min(getCostsVector(pop)))
        maxOutput.append(max(getCostsVector(pop)))
        avgOutput.append(np.average(getCostsVector(pop)))
        globalMinOutput.append(min(minOutput))

    endTime = tlib.time()
    preparePlot(field, minOutput, maxOutput, avgOutput, globalMinOutput)
    log.info("min valuse {}".format(min(minOutput)))
    print("min value {} - global min 1652".format(min(minOutput)))
    print("Evaluation time: {} s".format(endTime-starttime))
    print("SELECTION METHOD: {}".format(SELECTION_METHOD))
    return getBestIndividual(pop)

def randomSearch(time):
    bestIndividual = initialise(1)
    minCost = getCostsVector(bestIndividual)
    startTime = tlib.time()
    i =0
    while tlib.time() - startTime < time:
        newIndividual = initialise(1)
        i=i+1
        if getCostsVector(newIndividual) < minCost:
            minCost = getCostsVector(bestIndividual)
            bestIndividual = newIndividual
            print("Iteration: {} Min cost: {}".format(i, minCost))
    return minCost, bestIndividual

def getDate(inFile=True):
    if inFile:
        return datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    return datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

def logInfo():
    log.info(getDate())
    if SAVE_STRONGEST == 1:
        log.info("###SAVE_STRONGEST ENABLED")
    else:
        log.info("###SAVE_STRONGEST DISABLED")

    if SELECTION_METHOD == 0:
        log.info("###ROULETTE METHOD")
    else:
        log.info("###TOURNAMENT METHOD")

# minCost, bestIndividual = randomSearch(10)
# print("Min cost: {} of Individual: {}".format(minCost, bestIndividual))

#def greedySearch():

results=[]

for i in range(0, METHODS_TO_TEST):
    if METHODS_TO_TEST > 1:
        SELECTION_METHOD = i % METHODS_TO_TEST

    if TEST_TWO_SAVE_STRONGEST == True:
        if i == 2*METHODS_TO_TEST-1:
            SAVE_STRONGEST = 0

    logInfo()
    output = np.zeros((ITERATIONS, dimension), dtype=int,)

    for j in range(0, ITERATIONS):
        print("Evaluate: {}".format(j))
        log.info(("#Evaluate: {}".format(j)))
        wynik = geneticAlgorithm()
        output[j] = wynik
        #print("Min cost of iteration:", costFunction(output[j], COMBINATIONS))
        log.info(getDate())
        log.info(("Min cost of iteration: {}".format(costFunction(output[j], COMBINATIONS))))
        log.info(("Min of all evaluations: {}".format(min(getCostsVector(output[0:j+1,:])))))
        #print("Min of all evaluations:", min(getCostsVector(output[0:j+1,:])))

    costVector = getCostsVector(output)
    plt.show()
    log.info("cost Vector")
    log.info(costVector)
    log.info("Minimum cost")
    log.info(min(costVector))
    log.info("of Individual:")
    log.info(output[costVector.tolist().index(min(costVector))])
    log.info("----------------------------")
    results.append(min(costVector))
log.info("Cost Vector of outputs:")
log.info(''.join(str(e)+', ' for e in results))
log.info("BEST OF ALL")
log.info(min(results))
log.info("OF ID:")
log.info((results.index(min(results))))



