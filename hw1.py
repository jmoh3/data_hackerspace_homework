#
# CS 196 Data Hackerspace
# Assignment 1: Data Parsing and NumPy
# Due September 24th, 2018
#

import json
import csv
import numpy as np
import math
from PIL import Image

def histogram_times(filename):
    airplane_data = []
    
    with open(filename) as f:
        csv_reader = csv.reader(f)
        airplane_data = list(csv_reader)

    hourBuckets = [0] * 24
    hours = []

    for i in range(1, len(airplane_data)):
        if airplane_data[i][1] != '':
            hours.append(airplane_data[i][1].partition(':')[0])

    for hour in hours:
        # fix
        try:
            int(hour)
            if int(hour) < 24 and int(hour) >= 0:
                hourBuckets[int(hour)] += 1
        except ValueError:
            pass

    return hourBuckets

# print(histogram_times('airplane_crashes.csv'))

def weigh_pokemons(filename, weight):
    with open(filename) as data_file:
        pokemon = json.load(data_file)
    
    weightIndexes = []
    
    for i in range(0, len(pokemon['pokemon'])):
        pokeWeight = float(pokemon['pokemon'][i]['weight'].partition(" ")[0])
        print(pokeWeight)
        if pokeWeight == weight:
            weightIndexes.append(i)

    names = []

    for index in weightIndexes:
        name = pokemon['pokemon'][index]['name']
        names.append(name)

    return names

# print(weigh_pokemons('pokedex.json', 10.0))

def single_type_candy_count(filename):
    with open(filename) as data_file:
        pokemon = json.load(data_file)
    
    candy_sum = 0

    for i in range(0, len(pokemon['pokemon'])):
        if len(pokemon['pokemon'][i]['type']) == 1:
            try:
                num_candy = int(pokemon['pokemon'][i]['candy_count'])
                candy_sum += num_candy
            except KeyError:
                pass

    return candy_sum

# print(single_type_candy_count('pokedex.json'))

def reflections_and_projections(points):
    dimensions = points.shape
    output = []

    for x in range(0, dimensions[1]):
        vector = np.matrix(points[:,x])
        vector = np.transpose(vector)

        vector[1] = 2 - vector[1]

        rotationMatrix = np.array([[0, -1], [1, 0]])
        vector = np.matmul(rotationMatrix, vector)

        projectionMatrix = np.array([[1, 3], [3, 9]])
        projectionMatrix = (1/10) * projectionMatrix
        vector = np.matmul(projectionMatrix, vector)
        vector = np.transpose(vector)
        
        output.append(vector)

    return np.transpose(np.concatenate(output))

#x = np.array([[1, 3, 5, 7], [2, 4, 6, 8]])
#print(reflections_and_projections(x))


def normalize(image):
    max = np.amax(image)
    min = np.amin(image)

    minImage = np.full((32, 32), -min)
    newImage = np.add(image, minImage)
    
    normal = 255/(max - min)

    newImage = np.round(normal * image)

    return newImage

testImage = np.random.rand(32, 32) * 200
testIm = Image.fromarray(testImage)
testIm.show()

normalized = Image.fromarray(normalize(testImage))
normalized.show()
print(normalized)

def sigmoid_normalize(image, variance):

    def sigmoid(pixel, a):
        return 255(1 + math.e**((-a ** (-1))*(pixel - 128)))**(-1)

    return np.round(sigmoid(image, variance))

#sig_normalized = Image.fromarray(sigmoid_normalize(testImage, 10))
#sig_normalized.show()
