# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        newNumGhosts = successorGameState.getNumAgents()
        newAgentPos = []
        newAgentPos.append(newPos)

        for ghostNum in range(1,newNumGhosts):
          newAgentPos.append(successorGameState.getGhostPosition(ghostNum))

        newGhostDistSum = 0
        nearestGhostDist = 1000
        for ghostNum in range(1,len(newAgentPos)):
          newGhostDistSum += manhattanDistance(newPos, newAgentPos[ghostNum])
          nearestGhostDist = min(nearestGhostDist, manhattanDistance(newPos, newAgentPos[ghostNum]))

        newFoodCount = successorGameState.getNumFood()
        sumFoodDist = 0
        nearestFood = 100
        for i in range(0,newFoodCount):
            sumFoodDist += manhattanDistance(newFood.asList()[i], newPos)
            nearestFood = min(nearestFood, manhattanDistance(newFood.asList()[i], newPos))

        foodDistFactor = 10.0/(sumFoodDist+1.0)
        foodCountFactor = 30.0/(newFoodCount+1.0)

        if nearestGhostDist < 4:
          ghostDistFactor = nearestGhostDist
        else: 
          ghostDistFactor = 1000

        # print "New Successor:"
        # print "foodDist factor=", foodDistFactor
        # print "foodCount factor=", foodCountFactor
        # print "ghostDist factor=", ghostDistFactor

        evalFunc =  foodDistFactor + foodCountFactor + ghostDistFactor
        # print "total eval=", evalFunc

        return evalFunc

        # return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()
        return self.maxValue(gameState, 0, numAgents, 0)[1]

    def maxValue(self, state, agentIndex, numAgents, currentDepth):
      val = -10000
      legalActions = state.getLegalActions(agentIndex)
      valActionCouple = (val, "Error")

      for action in legalActions:
        successorState = state.generateSuccessor(agentIndex, action)
        if successorState.isWin() or successorState.isLose():
          successorVal = self.evaluationFunction(successorState)
        else:
          successorVal = self.minValue(successorState, agentIndex+1, numAgents, currentDepth)[0]

        if successorVal > val:
          valActionCouple = (successorVal, action)
          val = successorVal

      return valActionCouple

    def minValue(self, state, agentIndex, numAgents, currentDepth):
      val = 10000
      legalActions = state.getLegalActions(agentIndex)
      valActionCouple = (val, "Error")

      # combining the cases
      for action in legalActions:
        successorState = state.generateSuccessor(agentIndex, action)
        if successorState.isWin() or successorState.isLose() or (agentIndex == numAgents-1 and currentDepth == self.depth-1):
          successorVal = self.evaluationFunction(successorState)
        else:
          if agentIndex == numAgents-1:
            successorVal = self.maxValue(successorState, 0, numAgents, currentDepth+1)[0]
          else: 
            successorVal = self.minValue(successorState, agentIndex+1, numAgents, currentDepth)[0]

        if successorVal < val:
          valActionCouple = (successorVal, action)
          val = successorVal

      return valActionCouple


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()
        return self.maxValue(gameState, 0, numAgents, 0, -10000, 10000)[1]

    def maxValue(self, state, agentIndex, numAgents, currentDepth, alpha, beta):
      val = -10000
      legalActions = state.getLegalActions(agentIndex)
      valActionCouple = (val, "Error")

      for action in legalActions:
        successorState = state.generateSuccessor(agentIndex, action)
        if successorState.isWin() or successorState.isLose():
          successorVal = self.evaluationFunction(successorState)
        else:
          successorVal = self.minValue(successorState, agentIndex+1, numAgents, currentDepth, alpha, beta)[0]

        if successorVal > beta:
          return (successorVal, action)

        if successorVal > val:
          valActionCouple = (successorVal, action)
          val = successorVal
          alpha = max(successorVal, alpha)

      return valActionCouple

    def minValue(self, state, agentIndex, numAgents, currentDepth, alpha, beta):
      val = 10000
      legalActions = state.getLegalActions(agentIndex)
      valActionCouple = (val, "Error")

      # combining the cases
      for action in legalActions:
        successorState = state.generateSuccessor(agentIndex, action)
        if successorState.isWin() or successorState.isLose() or (agentIndex == numAgents-1 and currentDepth == self.depth-1):
          successorVal = self.evaluationFunction(successorState)
        else:
          if agentIndex == numAgents-1:
            successorVal = self.maxValue(successorState, 0, numAgents, currentDepth+1, alpha, beta)[0]
          else: 
            successorVal = self.minValue(successorState, agentIndex+1, numAgents, currentDepth, alpha, beta)[0]

        if successorVal < alpha:
          return (successorVal, action)

        if successorVal < val:
          valActionCouple = (successorVal, action)
          val = successorVal
          beta = min(successorVal, beta)

      return valActionCouple

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()
        return self.maxValue(gameState, 0, numAgents, 0)[1]

    def maxValue(self, state, agentIndex, numAgents, currentDepth):
      val = -10000
      legalActions = state.getLegalActions(agentIndex)
      valActionCouple = (val, "Error")

      for action in legalActions:
        successorState = state.generateSuccessor(agentIndex, action)
        if successorState.isWin() or successorState.isLose():
          successorVal = self.evaluationFunction(successorState)
        else:
          successorVal = self.expectedValue(successorState, agentIndex+1, numAgents, currentDepth)[0]

        if successorVal > val:
          valActionCouple = (successorVal, action)
          val = successorVal

      return valActionCouple

    def expectedValue(self, state, agentIndex, numAgents, currentDepth):
      val = 10000
      legalActions = state.getLegalActions(agentIndex)
      valActionCouple = (val, "Error")
      successorVals = []

      # combining the cases
      for action in legalActions:
        successorState = state.generateSuccessor(agentIndex, action)
        if successorState.isWin() or successorState.isLose() or (agentIndex == numAgents-1 and currentDepth == self.depth-1):
          successorVals.append(self.evaluationFunction(successorState))
        else:
          if agentIndex == numAgents-1:
            successorVals.append(self.maxValue(successorState, 0, numAgents, currentDepth+1)[0])
          else: 
            successorVals.append(self.expectedValue(successorState, agentIndex+1, numAgents, currentDepth)[0])

      expectedVal = float(sum(successorVals))/float(len(successorVals))
      valActionCouple = (expectedVal, "None")
      return valActionCouple

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    numGhosts = currentGameState.getNumAgents()
    agentPos = []
    agentPos.append(currentGameState.getPacmanPosition())
    foodPos = currentGameState.getFood()
    foodCount = currentGameState.getNumFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    for ghostNum in range(1,numGhosts):
      agentPos.append(currentGameState.getGhostPosition(ghostNum))

    ghostDistSum = 0
    nearestGhostDist = 1000
    nearestGhostIndex = 0
    for ghostNum in range(1,len(agentPos)):
      ghostDistSum += manhattanDistance(agentPos[0], agentPos[ghostNum])
      ghostDist = manhattanDistance(agentPos[0], agentPos[ghostNum])
      if ghostDist < nearestGhostDist:
        nearestGhostDist = ghostDist
        nearestGhostIndex = ghostNum

    sumFoodDist = 0
    nearestFood = 100
    for i in range(0,foodCount):
        sumFoodDist += manhattanDistance(foodPos.asList()[i], agentPos[0])
        nearestFood = min(nearestFood, manhattanDistance(foodPos.asList()[i], agentPos[0]))

    foodDistFactor = -sumFoodDist/1000.0
    foodCountFactor = 30.0/(foodCount+1.0)

    if scaredTimes[nearestGhostIndex-1] > 3:
      ghostDistFactor = 100.0
    elif nearestGhostDist < 4.0:
      ghostDistFactor = nearestGhostDist
    else: 
      ghostDistFactor = 100.0

    capsuleFactor = -20.0*len(currentGameState.getCapsules())

    # print "New eval:"
    # print "foodDist factor=", foodDistFactor
    # print "foodCount factor=", foodCountFactor
    # print "ghostDist factor=", ghostDistFactor

    evalFunc =  foodDistFactor + foodCountFactor + ghostDistFactor + capsuleFactor

    return evalFunc

    ###### over it

        # foodDistFactor = 5.0/(sumFoodDist+1.0)
    foodDistFactor = -sumFoodDist/1000.0
    # foodCountFactor = 50.0/(foodCount+1.0)
    if foodCount == 0:
      foodCountFactor = 10000
    else: 
      foodCountFactor = -10.0*foodCount

    if scaredTimes[nearestGhostIndex-1] > 3:
      ghostDistFactor = 20.0
    elif nearestGhostDist < 5:
      ghostDistFactor = nearestGhostDist/1000.0
    else: 
      ghostDistFactor = 10.0

# Abbreviation
better = betterEvaluationFunction

