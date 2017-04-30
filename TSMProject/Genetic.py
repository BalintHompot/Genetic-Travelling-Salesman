import geopy
import csv
import heapq
import copy
import random
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6367 * c
    return km


class version:

    def __init__(self, genom, cities_coor, m, color):
        self.genom = copy.deepcopy(genom)
        #self.genom.append(self.genom[0])
        self.cities_coor = cities_coor
        self.fitness = self.fitness_function(color)
        #self.fitness_function()

    def __lt__(self, other):
        return self.fitness<other.fitness

    def fitness_function(self, color):
        fit = 0
        #print(self.genom)
        for x in range(len(self.genom)-1):
            fit += haversine(self.cities_coor[self.genom[x]][1],self.cities_coor[self.genom[x]][0], self.cities_coor[self.genom[x+1]][1], self.cities_coor[self.genom[x+1]][0])
            #m.drawgreatcircle(cities_coor[self.genom[x]][1], cities_coor[self.genom[x]][0], cities_coor[self.genom[x + 1]][1],cities_coor[self.genom[x + 1]][0], linewidth=2, color= color)
        return fit


class population:
    def __init__(self,versions, size, mutation, m):
        self.versions = versions
        self.color = ("#%06x" % random.randint(0, 0xFFFFFF))

    def average(self):
        sum = 0
        count = 0
        for ver in self.versions:
            sum += ver.fitness
            count += 1

        return sum/count

    def crossover(self, size):
        newpop = []
        for sons in range(size - 1):
            pmax = len(self.versions)
            divider = pmax * (pmax +1) / 2
            probability = []
            for x in  range(len(self.versions)):
                probability.append(pmax / divider)
                pmax -= 1
            #print(probability)
            father = np.random.choice(self.versions, p = probability)
            mother = father
            while mother == father:
                mother = np.random.choice(self.versions, p = probability)
            #print("father is " + str(father.genom) + "\nmother is " + str(mother.genom))

            son = []
            #position = []
            pos1 = random.choice(father.genom)
            pos = int(father.genom.index(pos1)/2)

            for p in range(pos-1):
                son.append(-1)

            #for h in range(len(genom) - 1):
                #son.append(-1)
                #position.append(h)
            #son.append(-1)
            for chromosome in range(int(len(father.genom)/3)):
                #print("chromosome is " + str(father.genom[chromosome]))
                #while pos == -1:
                #    pos = np.random.choice(position)
                #print("pos is " + str(pos))
                son.append(father.genom[chromosome])
                #son[pos+1] = (father.genom[chromosome+1])
                #position[pos] = -1
                #try:
                #    position[pos + 1] = -1
                #except:
                #    pass
                #pos = -1


            s = set(son)


            for x in range(len(mother.genom)):
                if not mother.genom[x] in s:
                    placed = False
                    for j in range(len(son)):
                        if son[j]== -1:
                            son[j] = mother.genom[x]
                            placed = True
                            break
                    if not placed:
                        son.append(mother.genom[x])

            #print("new son: " + str(son))
            #print(str(len(mother.genom)) + " " + str(len(father.genom)))
            #print(len(son))
            sonversion = version(son,father.cities_coor, m, self.color)
            heapq.heappush(newpop, sonversion)
        heapq.heappush(newpop, self.versions[0])
        return newpop

def draw(version, cities_coor):
    m = Basemap(llcrnrlon=-180., llcrnrlat=-80., urcrnrlon=180., urcrnrlat=80., rsphere=(6378137.00, 6356752.3142),resolution='l', projection='mill', lat_0=0., lon_0=-0., lat_ts=30.)
    m.drawcoastlines()
    m.fillcontinents()
    for x in range(len(version.genom)-1):
        m.drawgreatcircle(cities_coor[version.genom[x]][1], cities_coor[version.genom[x]][0], cities_coor[version.genom[x + 1]][1],cities_coor[version.genom[x + 1]][0], linewidth=2, color= "b")
    return m
    
path = "C:\\Users\Bálint\Desktop\Bálint\program\Python\GeneticCities\\testcities.csv"
with open(path , 'r') as csvfile:
    fieldnames = ['Country', 'City', 'Latitude', 'Longitude']
    cities_reader = csv.DictReader(csvfile, fieldnames=fieldnames,  delimiter=';')
    cities = list(cities_reader)
    #print(cities)
    cities_coor = []
    for row in cities:
        try:
            cities_coor.append([float(row['Latitude']), float(row['Longitude'])])
        except:
            print("nem float")
    #print(cities_coor[1][0])
    #print(haversine(float(cities_coor[3][0]), float(cities_coor[3][1]), float(cities_coor[2][0]), float(cities_coor[2][1])))

    #m = Basemap(llcrnrlon=-180., llcrnrlat=-80., urcrnrlon=180., urcrnrlat=80., rsphere=(6378137.00, 6356752.3142),resolution='l', projection='mill', lat_0=0., lon_0=-0., lat_ts=30.)
    #m.drawcoastlines()
    #m.fillcontinents()

    population_size = 500
    genom = []
    for i in range(len(cities_coor)):
        genom.append(i)

    #print(genom)
    
    mutation = 0.1
    size = 50
    versions = []
    generations = 20000
    #for i in range(100):

    m = 1
    graphdata = []

    for i in range(size):
        random.shuffle(genom)
        heapq.heappush(versions, version(genom, cities_coor, m, 'b'))
    new_population = population(versions, size, mutation, m)

    for g in range(generations):
        newversions = new_population.crossover(size)
        new_population = population(newversions, size, mutation, m)
        #graphdata.append(np.mean(version.fitness for version in new_population.versions))
        #graphdata.append(new_population.average())
        graphdata.append((new_population.versions[0]).fitness)
        print((new_population.versions[0]).fitness)

    plt.plot(graphdata)
    plt.show()
    m = draw(new_population.versions[0], cities_coor)
    plt.show(m)
    #print(cities_coor)

    #draw(cities_coor)

# set up orthographic map projection with
# perspective of satellite looking down at 50N, 100W.
# use low resolution coastlines.

