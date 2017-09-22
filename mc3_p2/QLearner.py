"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand

#Each time goal is not reached returns -1
#Ignore Dyna at first

class QLearner(object):

    def __init__(self, num_states=100,num_actions = 4, alpha = 0.2, \
        gamma = 0.9, rar = 0.5, radr = 0.99, dyna = 0, \
        verbose = False):
        self.numStates = num_states
        self.numActions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.dyna = dyna
        self.verbose = verbose
        #self.num_actions = num_actions
        self.rar = rar
        self.radr = radr
        self.lastState = 0
        self.lastAction = 0
        self.Qtable = np.random.rand(self.numStates,self.numActions)
        if dyna>0:
            self.TC = np.empty((num_states,num_actions,num_states))
            self.TC[:,:,:]=0.00001
            self.R = np.empty((num_states,num_actions))
            self.T = np.empty((num_states, num_actions, num_states))

    def querysetstate(self, s):
        """
        roughly 20 lines or so
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.lastState = s
        #action = np.argmax(self.Qtable[s])
        action = rand.randint(0, self.numActions-1)
        self.lastAction=action
        if self.verbose: print "s =", s,"a =",action
        return action

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
        if(rand.random()<=self.rar):
            action = rand.randint(0, self.numActions - 1)
        else:
            iEstimate = r + self.gamma * self.Qtable[s_prime,np.argmax(self.Qtable[s_prime])]
            Qprime = ((1-self.alpha) * self.Qtable[self.lastState,self.lastAction]) + self.alpha * iEstimate
            #Qprime = (1-self.alpha)+self.alpha
            self.Qtable[self.lastState,self.lastAction]= Qprime
            action = np.argmax(self.Qtable[s_prime])
        if self.verbose: print "s =", s_prime,"a =",action,"r =",r
        self.rar = self.rar * self.radr
        if self.dyna>0:
            state = self.lastState
            a = self.lastAction
            sd_prime=s_prime
            for i in range(self.dyna):
                self.TC[state, a, sd_prime] += 1
                self.R[state, a] = (1 - self.alpha) + self.alpha * r
                state = rand.randint(0, self.numStates - 1)
                a = rand.randint(0, self.numActions - 1)
                self.T[state,a,sd_prime] = self.TC[state,a,sd_prime]/sum(self.TC[state,a])


        self.lastState=s_prime
        self.lastAction=action
        return action

    def updateRule(self,s,a,sPrime,reward):
        iEstimate = reward + self.gamma * self.Qtable[sPrime,np.argmax(self.Qtable[sPrime])]
        Qprime = ((1-self.alpha) * self.Qtable[s,a]) + self.alpha * iEstimate
        #Qprime = (1-self.alpha)+self.alpha
        return Qprime

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
