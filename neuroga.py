import copy
import json
import math
import random
import os
from threading import Thread

import numpy as np

DEBUG = True

def sigmoid(x):
    return 1/(1+np.exp(-x))

@np.vectorize
def relu(x):
    """
    Rectifier activation function.
    """
    return np.maximum(0, x)

class Network:
    def __init__(self, shape, activation=sigmoid, saved=None):
        """
        :param shape: list of num neurons in each layer. (Can make `None` if loading from `saved`)
        :param activation: activation function
        :param saved: saved representation of Network (from `serialise`)
        """

        self.activf = activation

        if saved is not None:
            data = json.loads(saved)
            self.shape = data['shape']
            self.weights = [np.array(x) for x in data['weights']]
            self.biases = [np.array(x) for x in data['biases']]
        else:
            self.shape = shape

            # init (random inital weights/biases)
            self.weights = [np.random.randn(j, i) for i, j in zip(
                self.shape[:-1], self.shape[1:])] # matrix for each layer-gap
            self.biases = [np.random.randn(i, 1) for i in self.shape[1:]] # single column matrix for each layer-gap

        if len(shape)<2:
            raise ValueError

    def forward(self, data):
        """
        run input through network and get output
        :param data: list of inputs
        :return: list of outputs
        """
        result = (np.array([data]).T if isinstance(data, list) else data)

        for w, b in zip(self.weights, self.biases):
            result = self.activf(np.dot(w,result) + b)

        return result.flatten()

    def serialise(self):
        return json.dumps({
            'shape':self.shape,
            'weights':[weights.tolist() for weights in self.weights],
            'biases':[biases.tolist() for biases in self.biases]
        })

class Agent:
    """
    A member of a `Genetic` class' population
    """
    def __init__(self, net:Network, fitf):
        """
        :param net: underlying Neural Network
        :param fitf: fitness function. Network will be passed as parameter.
        """
        self.net:Network = net
        self.__fitf = fitf
        self.fitness = 0

    def evaluate(self):
        """
        update fitness by executing fitness function
        """
        fitness = self.__fitf(self.net)
        self.fitness = fitness
        return fitness

class Genetic:
    def __init__(self,
                 shape,
                 pop_size,
                 fitf,
                 save=None,
                 sel_top=0.5,
                 sel_rand=0.3,
                 sel_mut=0.6,
                 prob_cross=0.5,
                 prob_mut=0.7,
                 mut_range=(-1,1),
                 activf=sigmoid,
                 opt_max=True,
                 parallelise=False):
        """
        Evolve Neural Networks to optimise a given function ('fitness function')
        :param shape: shape of Neural Networks (list of num neurons in each layer). Can be `None` if loading.
        :param pop_size: number of NNs in population Can be `None` if loading.
        :param fitf: fitness function. Corresponding NN will be passed as parameter. Should output fitness.
        :param save: model saving location
        :param sel_top: Num top agents to be selected for the next generation per step. Fraction of pop_size.
        :param sel_rand: Num agents to be randomly selected for the next generation per step. Fraction of pop_size
        :param sel_mut: Num agents to be mutated per step. Fraction of pop_size
        :param prob_cross: During crossing, prob for a given gene to be crossed. 0-1 (Cross frequency)
        :param prob_mut: During mutation, prob for a given gene to be mutated. 0-1 (Mutation frequency)
        :param mut_range: (Tuple) Mutation range (Mutation amount)
        :param activf: activation function for underlying NNs (default: sigmoid)
        :param opt_max: whether to maximise fitf or to minimise it
        :param parallelise: whether to parallelise (multithread) evaluation of NNs (execution of fitfs)
        """
        # todo saving to file

        if sel_top+sel_rand>=1.0:
            raise ValueError('sel_top + sel_rand cannot sum to 1.0 because otherwise no children agents will be '
                             'in next generation')

        self.shape = shape
        self.pop_size = pop_size
        self.fitf = fitf
        self.save = save
        self.sel_top = sel_top
        self.sel_rand = sel_rand
        self.sel_mut = sel_mut
        self.prob_cross = prob_cross
        self.prob_mut = prob_mut
        self.mut_range = mut_range
        self.activf = activf
        self.opt_max = opt_max
        self.parallelise = parallelise

        self.gen_num = 0
        self.population = []

        if self.save is not None:
            if not os.path.isdir(self.save):
                if DEBUG: print('Save does not exist')
                os.makedirs(self.save)

                # init population
                for i in range(self.pop_size):
                    self.population.append(Agent(
                        Network(self.shape, self.activf),
                        self.fitf))

                with open(os.path.join(self.save,'ga.json'),'w') as f:
                    json.dump({
                        'shape':self.shape,
                        'pop_size':self.pop_size,
                        'sel_top':self.sel_top,
                        'sel_rand':self.sel_rand,
                        'sel_mut':self.sel_mut,
                        'prob_cross':self.prob_cross,
                        'prob_mut':self.prob_mut,
                        'mut_range':self.mut_range,
                        'opt_max':self.opt_max,
                        'parallelise':self.parallelise,
                        'population':[agent.net.serialise() for agent in self.population],
                        'gen_num':self.gen_num
                    },f)
            else:
                with open(os.path.join(self.save,'ga.json'),'r') as f:
                    data = json.load(f)
                    self.shape=data['shape']
                    self.pop_size=data['pop_size']
                    self.sel_top=data['sel_top']
                    self.sel_rand=data['sel_rand']
                    self.sel_mut=data['sel_mut']
                    self.prob_cross=data['prob_cross']
                    self.prob_mut=data['prob_mut']
                    self.mut_range=data['mut_range']
                    self.opt_max=data['opt_max']
                    self.parallelise=data['parallelise']
                    self.population=[Agent(Network(self.shape,self.activf,saved=x),self.fitf)
                                     for x in data['population']]
                    self.gen_num=data['gen_num']
                if DEBUG: print('Loaded from save')

    def next_pop(self):
        """
        select agents to survive to next generation
        :return:
        """
        next_pop = []

        # select specified no of top agents
        for i in range(math.floor(self.pop_size*self.sel_top)):
            next_pop.append(copy.deepcopy(self.population[i]))

        # randomly select specified no of agents
        for i in range(math.floor(self.pop_size*self.sel_rand)):
            next_pop.append(copy.deepcopy(random.choice(self.population)))

        return next_pop

    def cross(self, parent1, parent2):
        """
        create child agent from parent agents
        """
        child = copy.deepcopy(parent1)

        # cross weights
        for i, weights in enumerate(parent2.net.weights):
            for j, weight in enumerate(parent2.net.weights[i]):
                if random.random() < self.prob_cross:
                    child.net.weights[i][j] = weight

        # cross biases
        for i, biases in enumerate(parent2.net.biases):
            for j, bias in enumerate(parent2.net.biases[i]):
                if random.random() < self.prob_cross:
                    child.net.biases[i][j] = bias

        return child

    def __evaluate(self):
        """
        evaluate and sort agents
        """
        if self.parallelise:
            threads=[]

            for agent in self.population:
                t = Thread(target=agent.evaluate)
                t.start()
                threads.append(t)

            for t in threads:
                t.join()

        else:
            for agent in self.population:
                agent.evaluate()

        self.population.sort(key=lambda agent: agent.fitness, reverse=self.opt_max)

    def step(self):
        self.__evaluate()
        if DEBUG: print('['+str(self.gen_num)+'] Fit: '+str(self.population[0].fitness))

        self.population=self.next_pop()

        # children generation
        for i in range(self.pop_size - len(self.population)):
            self.population.insert(0,self.cross(self.population[0],random.choice(self.population)))

        # weights mutation
        for no in range(math.floor(self.pop_size*self.sel_mut)):
            mutant = random.choice(self.population)
            for i, weights in enumerate(mutant.net.weights):
                for j, weight in enumerate(mutant.net.weights[i]):
                    if random.random() < self.prob_mut:
                        mutant.net.weights[i][j]+=random.uniform(*self.mut_range)

        # biases mutation
        for no in range(math.floor(self.pop_size * self.sel_mut)):
            mutant = random.choice(self.population)
            for i, biases in enumerate(mutant.net.biases):
                for j, bias in enumerate(mutant.net.biases[i]):
                    if random.random() < self.prob_mut:
                        mutant.net.biases[i][j]+=random.uniform(*self.mut_range)

        self.gen_num+=1


random.seed(3)

# n = Network([2,2,1],saved='{"shape": [2, 2, 1], "weights": [[[-17.984009797282607, -17.51685086787261], [22.012925023447604, 22.25461461520413]], [[-290.63351218454557, -287.90345525718317]]], "biases": [[[9.716006094964486], [-33.44022550410801]], [[37.21915278409331]]]}')
# print(n)

def xor_error(net):
    # error from XOR
    return (0 - net.forward([0, 0])[0]) ** 2 +\
           (1 - net.forward([0, 1])[0]) ** 2 +\
           (1 - net.forward([1, 0])[0]) ** 2 +\
           (0 - net.forward([1, 1])[0]) ** 2

g = Genetic([2,2,1],
            10,
            xor_error,
            save='models/xor/',
            opt_max=False,
            )
for b in range(10000):
    g.step()
    # if g.population[0].fitness<1e-100:
    #     # print('wow')
    #     inpts = [[0, 0], [0, 1], [1, 0], [1, 1]]
    #     for inpt in inpts:
    #         outpt = g.population[0].net.forward(inpt)[0]
    #         print(str(inpt) + '--> ' + str(outpt))
    #     exit()
