import json
import math
import random
import os
import statistics
import warnings
from threading import Thread
from typing import List
import numpy as np

DEBUG = True
PLOT = True

if PLOT:
    import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

@np.vectorize
def relu(x):
    """
    Rectifier activation function.
    """
    return np.maximum(0, x)

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth[int(box_pts*0.5):int(-0.5*box_pts)]

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

            # init (random inital weights/biases). Glorot initialisation
            self.weights = [np.random.normal(0,(2.0/(self.shape[0]+self.shape[-1]))**0.5,(j,i)) for i, j in zip(
                self.shape[:-1], self.shape[1:])] # matrix for each layer-gap
            # single column matrix for each layer-gap.
            self.biases = [np.random.normal(0,(2.0/(self.shape[0]+self.shape[-1]))**0.5,(i,1)) for i in self.shape[1:]]

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
        :param fitf: fitness function. Network will be passed as parameter. Do not call directly.
        """
        self.net:Network = net
        self.fitf = fitf
        self.fitness = 0

    def evaluate(self):
        """
        update fitness by executing fitness function
        """
        fitness = self.fitf(self.net)
        self.fitness = fitness
        return fitness

class GeneticWarning(UserWarning):
    pass

class Genetic:
    def __init__(self,
                 shape,
                 pop_size,
                 fitf,

                 save=None,
                 save_interval=50,
                 save_hist=True,

                 selection_args=None,
                 cross_args=None,
                 mutate_args=None,

                 activf=sigmoid,
                 opt_max=True,
                 parallelise=False):
        """
        Evolve Neural Networks to optimise a given function ('fitness function').
        Can override exposed methods.

        :param shape: shape of Neural Networks (list of num neurons in each layer). Can be `None` if loading.
        :param pop_size: number of NNs in population Can be `None` if loading.
        :param fitf: fitness function. Corresponding NN will be passed as parameter. Should output fitness.

        :param save: model saving location
        :param save_interval: amount of generations between saves
        :param save_hist: whether to store best NNs for each key generation

        :param selection_args: arguments passed to selection method
        :param cross_args: arguments passed to cross method
        :param mutate_args: arguments passed to mutate method

        :param activf: activation function for underlying NNs (default: sigmoid)
        :param opt_max: whether to maximise fitf or to minimise it
        :param parallelise: whether to parallelise (multithread) evaluation of NNs (execution of agents)
        """

        if selection_args is None:
            selection_args = {
                'ptop': 0.1,
                'prand': 0.1
            }
        if cross_args is None:
            cross_args = {}
        if mutate_args is None:
            mutate_args = {}

        self.shape = shape
        self.pop_size = pop_size
        self.fitf = fitf

        self.save = save
        self.save_interval = save_interval
        self.save_hist = save_hist

        self.selection_args = selection_args
        self.cross_args = cross_args
        self.mutate_args = mutate_args

        self.activf = activf
        self.opt_max = opt_max
        self.parallelise = parallelise

        self.population = []
        self.gen_num = 0
        self.__fit_hist = []

        # init population
        for i in range(self.pop_size):
            self.population.append(Agent(
                Network(self.shape, self.activf),
                self.fitf))

        if self.save is not None:
            if not os.path.isdir(self.save):
                if DEBUG: print('Save does not exist')
                os.makedirs(self.save)

                self.__save()
            else:
                self.__load()

    def __save(self):
        with open(os.path.join(self.save, 'ga.json'), 'w') as f:
            json.dump({
                'shape': self.shape,
                'pop_size': self.pop_size,
                'gen_num': self.gen_num,
                'population': [agent.net.serialise() for agent in self.population]
            }, f)
        if DEBUG: print('Saved.')

    def __load(self):
        with open(os.path.join(self.save, 'ga.json'), 'r') as f:
            data = json.load(f)
            self.shape = data['shape']
            self.pop_size = data['pop_size']
            self.gen_num = data['gen_num']
            self.population = [Agent(Network(self.shape, self.activf, saved=x), self.fitf)
                               for x in data['population']]
        if DEBUG: print('Loaded from save')

    def __sort(self):
        self.population.sort(key=lambda agent: agent.fitness, reverse=self.opt_max)

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

        self.__sort()

    @staticmethod
    def select(pop:List[Agent],**kwargs)->List[Agent]:
        """
        select agents to survive to next generation
        :param pop: ordered current population
        :return: intermediate population
        """
        num_top = 0
        num_rand = 0

        if 'ptop' in kwargs:
            num_top = math.floor(len(pop)*float(kwargs['ptop']))
        if 'prand' in kwargs:
            num_rand = math.floor(len(pop)*float(kwargs['prand']))
        if 'top' in kwargs:
            if num_top != 0:
                warnings.warn('Overriding ptop parameter', GeneticWarning)
            num_top = kwargs['top']
        if 'rand' in kwargs:
            if num_rand != 0:
                warnings.warn('Overriding prand parameter', GeneticWarning)
            num_rand = kwargs['rand']

        if num_top+num_rand<2:
            raise ValueError('Cannot select less than 2 agents for intermediary population')
        if num_top+num_rand==len(pop):
            warnings.warn('With the current selection params, recombination will be skipped as'
                          ' the intermediary population size will be equal to the target population size',
                          GeneticWarning)
        if num_rand+num_rand>len(pop):
            raise ValueError('Population not big enough for selection with current selection params')

        ipop = []

        # top agents
        for i in range(num_top):
            ipop.append(pop.pop(0))

        # rand agents
        for i in range(num_rand):
            ipop.append(pop.pop(random.randint(0,len(pop)-1)))

        return ipop

    @staticmethod
    def cross(parent1:Agent, parent2:Agent, **kwargs)->List[Agent]:
        """
        create children agents from parent agents
        :return: list of children agents (pref: 2)
        """
        prob = 0.6
        bias_prob = prob
        cross_biases = True

        if 'prob' in kwargs:
            prob = kwargs['prob']
            bias_prob = prob
        if 'bias_prob' in kwargs:
            bias_prob = kwargs['bias_prob']
        if 'cross_biases' in kwargs:
            cross_biases = kwargs['cross_biases']

        child1 = Network(parent1.net.shape,parent1.net.activf)
        child2 = Network(parent1.net.shape,parent1.net.activf)

        # cross weights
        for i, weights in enumerate(parent1.net.weights):
            for j in range(len(weights)):
                if random.random()<prob:
                    # swap
                    child1.weights[i][j] = parent2.net.weights[i][j]
                    child2.weights[i][j] = parent1.net.weights[i][j]
                else:
                    # don't swap
                    child1.weights[i][j] = parent1.net.weights[i][j]
                    child2.weights[i][j] = parent2.net.weights[i][j]

        if cross_biases:
            # cross biases
            for i, biases in enumerate(parent1.net.biases):
                for j in range(len(biases)):
                    if random.random()<bias_prob:
                        # swap
                        child1.biases[i][j] = parent2.net.biases[i][j]
                        child2.biases[i][j] = parent1.net.biases[i][j]
                    else:
                        # don't swap
                        child1.biases[i][j] = parent1.net.biases[i][j]
                        child2.biases[i][j] = parent2.net.biases[i][j]

        return [Agent(child1,parent1.fitf), Agent(child2,parent1.fitf)]

    @staticmethod
    def recombine(ipop:List[Agent], pop_size, cross_args=None)->List[Agent]:
        """
        recombine the intermediary population until population of specified size is formed
        :return: next population
        """
        if len(ipop)<2:
            raise ValueError('Intermediary population must be bigger than 2 for recombination')

        next_pop = []

        while len(next_pop) < pop_size:
            children = Genetic.cross(ipop[0],ipop[1],**cross_args)

            while len(next_pop) < pop_size and len(children)>0:
                next_pop.append(children.pop())

        return next_pop

    @staticmethod
    def mutate(pop:List[Agent], **kwargs)->List[Agent]:
        """
        mutate the given population
        """
        selp = 1.0
        prob = 0.3
        bias_prob = prob
        mutate_biases = True
        amount = 2
        bias_amount = amount

        if 'selp' in kwargs:
            selp = kwargs['selp']
        if 'prob' in kwargs:
            prob = kwargs['prob']
            bias_prob = prob
        if 'bias_prob' in kwargs:
            bias_prob = kwargs['bias_prob']
        if 'mutate_biases' in kwargs:
            mutate_biases = kwargs['mutate_biases']
        if 'amount' in kwargs:
            amount = kwargs['amount']
            bias_prob = amount
        if 'bias_amount' in kwargs:
            bias_amount = kwargs['bias_amount']

        ret_pop = []
        num = math.floor(len(pop)*selp)

        for _ in range(num):
            agent = pop.pop(random.randint(0,len(pop)-1))

            # mutate weights
            for i in range(len(agent.net.weights)):
                for j in range(len(agent.net.weights[i])):
                    if random.random() < prob:
                        # mutate
                        agent.net.weights[i][j]+=random.uniform(-1*amount,amount)

            if mutate_biases:
                # mutate biases
                for i in range(len(agent.net.biases)):
                    for j in range(len(agent.net.biases[i])):
                        if random.random() < bias_prob:
                            # mutate
                            agent.net.biases[i][j] += random.uniform(-1*bias_amount, bias_amount)

            ret_pop.append(agent)

        return ret_pop

    def step(self):
        self.__evaluate()

        if DEBUG:
            try:
                print('['+str(self.gen_num)+'] Fit: '+str(self.population[0].fitness)+
                      ' Stdv: '+str(statistics.stdev([x.fitness for x in self.population])))
            except AssertionError: pass

            if PLOT:
                self.__fit_hist.append(self.population[0].fitness)

                if len(self.__fit_hist) > 50:
                    del self.__fit_hist[0]

                plt.clf()
                plt.xlabel('generation')
                plt.ylabel('fitness')
                # plt.plot(smooth(self.__fit_hist,2), color='red')
                plt.plot(self.__fit_hist, color='red')
                plt.pause(0.05)
                plt.draw()

        self.population = self.mutate(
                self.recombine(
                    self.select(self.population,**self.selection_args),
                    self.pop_size,
                    self.cross_args), **self.mutate_args)

        if self.save is not None:
            if self.gen_num % self.save_interval == 0:
                self.__save()
                if self.save_hist:
                    # save hist
                    with open(os.path.join(self.save,str(self.gen_num)+'.json'),'w') as f:
                        f.write(self.population[0].net.serialise())

        self.gen_num+=1
