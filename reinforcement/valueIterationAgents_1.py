# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        valuesPrevious = self.values.copy()

        for i in range(0, self.iterations):
          for st in states:
            actionsAvailable = self.mdp.getPossibleActions(st)
            QVals = util.Counter()
            for act in actionsAvailable:
              QValue = 0
              succStatesAndProbs = self.mdp.getTransitionStatesAndProbs(st, act)
              for possibleSucc in succStatesAndProbs:
                succReward = self.mdp.getReward(st, act, possibleSucc[0])
                QValue += possibleSucc[1] * (succReward + self.discount * valuesPrevious[possibleSucc[0]])
              QVals[(st, act)] = QValue
            self.values[st] = QVals[QVals.argMax()]
          valuesPrevious = self.values.copy()


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        QValue = 0
        succStatesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
        for possibleSucc in succStatesAndProbs:
          succReward = self.mdp.getReward(state, action, possibleSucc[0])
          QValue += possibleSucc[1] * (succReward + self.discount * self.values[possibleSucc[0]])
        return QValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
          return None

        actionsAvailable = self.mdp.getPossibleActions(state)
        actionAndQVals = util.Counter()
        for act in actionsAvailable:
          QValue = self.computeQValueFromValues(state, act)
          actionAndQVals[act] = QValue
        return actionAndQVals.argMax()


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()

        for i in range(0, self.iterations):
          stateNum = i % len(states)
          st = states[stateNum]
          actionsAvailable = self.mdp.getPossibleActions(st)
          QVals = util.Counter()
          for act in actionsAvailable:
            QValue = 0
            succStatesAndProbs = self.mdp.getTransitionStatesAndProbs(st, act)
            for possibleSucc in succStatesAndProbs:
              succReward = self.mdp.getReward(st, act, possibleSucc[0])
              QValue += possibleSucc[1] * (succReward + self.discount * self.values[possibleSucc[0]])
            QVals[(st, act)] = QValue
          self.values[st] = QVals[QVals.argMax()]

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        print "*******"
        states = self.mdp.getStates()

        # compute predecessors of all states
        predecessors = dict()
        for st in states:
          actionsAvailable = self.mdp.getPossibleActions(st)
          for act in actionsAvailable:
            succStatesAndProbs = self.mdp.getTransitionStatesAndProbs(st, act)
            for possibleSucc in succStatesAndProbs:
              if possibleSucc[1] > 0: # if any probability

                # if state not in dictionary yet, make new key-value pair with for predecessor
                if possibleSucc[0] not in predecessors.keys():
                  predecessors[possibleSucc[0]] = {st}
                # if state in dictionary already, add state to values of predecessor key
                else:
                  predecessors[possibleSucc[0]].add(st)

        print "Predecessors ", predecessors

        # initialize empty priority queue
        queuedStates = util.PriorityQueue()

        # for each non-terminal state, do:
        for st in states:
          if not self.mdp.isTerminal(st):
            act = self.computeActionFromValues(st)
            maxQVal = self.computeQValueFromValues(st, act)
            diff = abs(maxQVal - st.getValue())
            print "Adding to queue state", st, " with priority ", -1*diff
            queuedStates.update(st, -1*diff)

        for i in xrange(self.iterations):
          if queuedStates.isEmpty():
            print "Queue is empty. Terminating runValueIteration."
            return None
          prioritizedState = queuedStates.pop()
          prioritizedVal = prioritizedState.getValue()
          print "updating state ", prioritizedState, "with value", prioritizedVal
          self.values[prioritizedState] = prioritizedVal

          for pred in predecessors[prioritizedState]:
            if not self.mdp.isTerminal(pred):
              act = self.computeActionFromValues(pred)
              maxQVal = self.computeQValueFromValues(pred, act)
              diff = abs(maxQVal - self.values[pred])
              if diff > self.theta: 
                print "Adding to queue pred ", pred, " with priority ", -1*diff
                queuedStates.update(pred, -1*diff)

              # QValue = 0
              # succStatesAndProbs = self.mdp.getTransitionStatesAndProbs(st, act)
              # for possibleSucc in succStatesAndProbs:
              #   succReward = self.mdp.getReward(st, act, possibleSucc[0])
              #   QValue += possibleSucc[1] * (succReward + self.discount * self.values[possibleSucc[0]])
              # QVals[(st, act)] = QValue
            # maxQVal = QVals[QVals.argMax()]








