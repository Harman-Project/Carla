#!/usr/bin/env python

#INIT
import glob
import os
from tkinter import *
import threading
import sys
import numpy as np
import random
import os
import keras
import time
import traceback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.datasets import mnist
from keras.utils import to_categorical
import tensorflow as tf
#END OF INIT

#CONST

FILEPATH="nets/"

population = 10
generations = 9999999

mute_weight1, mute_weight2 = 0.7, 0.3 #determine how much from one and from another parent
a_mute, b_mute = -0.02, 0.02 #determine mutation rate
best_slice = 2
mutate_best = False
input_dim = 3
network_layers = [3, 6, 1]


def get_random_matrix(a_range, b_range, low, high=None):
    if high is None:
        res = (b_range - a_range) * np.random.random_sample((low,)) + a_range    
    else:
        res = (b_range - a_range) * np.random.random_sample((low, high)) + a_range           
    
    return res

class Network:
    def __init__(self):
        self._id=np.random.randint(1,1000)

        self._model = None
        self._accuracy = 0

        self._weights = [None for _ in range(len(network_layers))]
        self._bias = [None for _ in range(len(network_layers))]

    def clone(self):
        new_net = Network()

        new_net._weights = self._weights
        new_net._bias = self._bias

        return new_net
    
    def random_weights(self):
        self._weights = []        
        self._weights.append(get_random_matrix(-1, 1, input_dim, network_layers[0]))
        for i in range(1, len(network_layers)):
            self._weights.append(get_random_matrix(-1, 1, network_layers[i-1], network_layers[i]))
            
        self._bias = []
        for i in range(len(network_layers)):
            self._bias.append(get_random_matrix(-1, 1, network_layers[i]))
    

    def serve_model(self):            
        self._model = Sequential()
        
        input_layer = Dense(network_layers[0], input_shape=[input_dim,])
        self._model.add(input_layer)
        input_layer.set_weights([self._weights[0], self._bias[0]])
        
        for i in range(1, len(network_layers)):
            layer = Dense(network_layers[i])
            self._model.add(layer)
            layer.set_weights([self._weights[i], self._bias[i]])
            
        self._model.compile(loss='mean_squared_error', optimizer='nadam', metrics=['acc'])

def init_networks(population):
    networks = [Network() for _ in range(population)]
    for network in networks:
        network.random_weights()
    return networks

def fitness(networks, start_race_func):
    i=0
    for network in networks:
        i += 1
        try:
            network.serve_model()
            network._accuracy = start_race_func(network._model)
            print ("Accuracy of {} genom with id {}:      {}".format(i, network._id, network._accuracy))
        except:
            network._accuracy = 0
            print ('Build failed.')
            traceback.print_exc()
            os.system("PAUSE")

    return networks

def selection(networks):
    networks = sorted(networks, key=lambda network: network._accuracy, reverse=True)
    
    for i in range(len(networks)):
        filename="Network "+str(i)+".h5"
        
        if os.path.exists(FILEPATH+filename):
            os.unlink(FILEPATH+filename)
        networks[i]._model.save(FILEPATH+filename)

    for network in networks:
        network._model = None
    keras.backend.clear_session()    
    networks = networks[:best_slice]
    return networks

def crossover(networks):
    offspring = []
    for i in range(len(networks)):
        offspring.append(networks[i].clone())
    networks.extend(offspring)
    
    offspring = []
    for _ in range(int((population - len(networks)) / 2)):
        parent1 = random.choice(networks)
        parent2 = random.choice(networks)
        child1 = Network()
        child2 = Network()
        
        for i in range(len(network_layers)):
            child1._weights[i] = (mute_weight1 * parent1._weights[i] + mute_weight2 * parent2._weights[i])
            child2._weights[i] = (mute_weight2 * parent1._weights[i] + mute_weight1 * parent2._weights[i])
        
            child1._bias[i] = (mute_weight1 * parent1._bias[i] + mute_weight2 * parent2._bias[i])
            child2._bias[i] = (mute_weight2 * parent1._bias[i] + mute_weight1 * parent2._bias[i])

        offspring.append(child1)
        offspring.append(child2)

    networks.extend(offspring)
    return networks

def mutate(networks):
    if mutate_best:
        index_range = range(len(networks))
    else:
        index_range = range(best_slice, len(networks))

    for i in index_range:
        for j in range(len(network_layers)):
            sizes = networks[i]._weights[j].shape
            mutation = get_random_matrix(a_mute, b_mute, sizes[0], sizes[1])
            networks[i]._weights[j] += mutation
                
            sizes = networks[i]._bias[j].shape
            mutation = get_random_matrix(a_mute, b_mute, sizes[0])
            networks[i]._bias[j] += mutation

    return networks

def evolution(start_race_func):
    networks = init_networks(population)
    
    for gen in range(generations):
        print ('Generation {}'.format(gen+1))
       
        networks = fitness(networks, start_race_func)
        networks = selection(networks)
        networks = crossover(networks)
        networks = mutate(networks)
                


















