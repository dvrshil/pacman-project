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
import math

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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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
        
        """
        We previously tried:
        -----
        deltaFoodDist = nearbyFoodDist - newNearbyFoodDist            
        if deltaFoodDist > 0: heuristic += 10
        heuristic -= len(newFood) #less food left is better
        if nearbyGhost <= newNearbyGhost: heuristic += 100
        else: heuristic -= 50
        -----
        if scaredTime > 0 and deltaNearbyGhostDist > 0: heuristic += 20
        deltaNearbyGhostDist = newNearbyGhostDist - nearbyGhostDist
        scaredTime = sum(newScaredTimes)
        """
        
        #we return +/- infinity if the given action leads to win/lose states
        if successorGameState.isWin(): return math.inf
        if successorGameState.isLose(): return -math.inf
    
        #Simply derived the distance of the nearest food pellet in the new state
        newFoodDistances = [util.manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()]
        newNearbyFoodDist = min(newFoodDistances)
        
        #initialized heuristic and defined a variable that keeps track 
        #of the score difference of the current and successor game states
        heuristic = 0
        deltaScore = successorGameState.getScore() - currentGameState.getScore()
        
        #we set heuristic values using basic assumptions and trial-error of watching the pacman play
        #we tried many complex parameters here, but this is the most simple version of our implementation.
        heuristic += 1/newNearbyFoodDist
        heuristic += deltaScore
    
        #simply return the final heuristic value to the evaluation call
        return heuristic

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
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        
        #simply saved the number of ghosts
        numGhosts = gameState.getNumAgents() - 1
        
        #implemented a min-value evaluator that we will use for analyzing ghost's moves
        def minValue(state, depth, agentIndex):
            value = math.inf #initialized a variable to track the minimum value
            if state.isWin() or state.isLose(): return self.evaluationFunction(state) #return evaluation if terminal states
            actions = state.getLegalActions(agentIndex)
            for action in actions: #find the minimum score from the successor states that a ghost could optimize for
                successor = state.generateSuccessor(agentIndex, action)
                if agentIndex == numGhosts: value = min(value, maxValue(successor, depth))
                else: value = min(value, minValue(successor, depth, agentIndex+1))
            return value #return the minimum of the scores
        
        #implemented a max-value evaluator that we will use for analyzing pacman's moves, thus ahentIndex would default to 0
        def maxValue(state, depth):
            if state.isWin() or state.isLose() or depth + 1 == self.depth: return self.evaluationFunction(state)
            value = -math.inf #initialized a variable to track the maximum value
            actions = state.getLegalActions(0)
            for action in actions: #find the maximum score from the successor states that our pacman should optimize for
                successor = state.generateSuccessor(0, action)
                value = max(value, minValue(successor, depth + 1, 1))
            return value #return the maximum of the scores

        #The evaluator that actually returns the best possibble move for pacman, from a given state
        myActions = gameState.getLegalActions(0)
        score = -math.inf #initialized a variable to track the best score to be achieved
        finalAction = '' #initialized a variable to track the best action that leads to the best score
        for newAction in myActions:
            successor = gameState.generateSuccessor(0, newAction)
            #we need to find the ghost's moves for our root action, and further analyse more moves in the later states, 
            #these further moves are automatically handled by the min-max-values functions recursing on each other
            newScore = minValue(successor, 0, 1)
            #now we optimize for the best action (achieving maximum possible score),
            #we simply return the action that leads to the highest score after analyzing ghosts' movements
            if newScore > score: finalAction, score = newAction, newScore
        return finalAction #return the best action for the current state
        
        #util.raiseNotDefined()
        
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def maxValue(self,depth,state):
        legalAct = state.getLegalActions(0)
        if (depth > self.depth) or state.isWin() or state.isLose(): #Return the state evaluation if the required depth is exceeded or the game is over.
            return self.evaluationFunction(state)
        
        maxEval = -float("inf")

        for action in legalAct:  #After Pacman has completed all of the available actions, calculate the maximum value of ghost actions
            successor = state.generateSuccessor(0,action)
            maxEval = max(maxEval , self.chanceValue(depth,successor,1))
        return maxEval

    def chanceValue(self,depth,state,agentIndex):
        legalAct = state.getLegalActions(agentIndex)
        if state.isWin() or state.isLose():  #there's no need to verify the depth because it only changes when Max is playing.
            return self.evaluationFunction(state)
        
        chanceEval = 0
        agentCount = state.getNumAgents()
        if agentIndex <= agentCount-2:  #If there are another ghosts after this ghost
            for action in legalAct:   
                successorState = state.generateSuccessor(agentIndex,action) #calculate the chance of the current ghost action and the next one
                chanceEval = chanceEval + self.chanceValue(depth,successorState,agentIndex+1)
                
        else:    #If there are no ghosts after this ghost then it is pacman's (max) turn
            for action in legalAct:  
                successorState = state.generateSuccessor(agentIndex,action)
                chanceEval = chanceEval + self.maxValue(depth+1,successorState)/len(legalAct)  #probability of a ghost taking each action is 1/len(legalAct) 
        
        return chanceEval

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        bestActionValue = -float("inf")
        bestAction = None
        for action in gameState.getLegalActions(0): #for pacman's every action 
            successorState = gameState.generateSuccessor(0,action)   #the next state if pacman do the current action
            if self.chanceValue(1,successorState,1) > bestActionValue:  #find the max value of the ghosts' actions 
                bestActionValue = self.chanceValue(1,successorState,1)
                bestAction = action
        return bestAction
       # return self.maxVal(gameState, 0, 0)[0] 
        
    def maxVal(self, gameState, agentIndex, depth):
            best = ("max", -float("inf"))
            for action in gameState.getLegalActions(agentIndex):
                nGameStateState = gameState.generateSuccessor(agentIndex, action)
                nAgentIndex = (depth+1)%gameState.getNumAgents()
                val = (action, self.expectiMax(nGameState, nAgentIndex, depth+1))
                
                best = max(best, val, key=lambda x:x[1])
            return best

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
  
    currentscore = currentGameState.getScore()  #because the starting evaluation represents the current score, while Pacman does nothing, the evaluation decreases.

    pacmanPosition = currentGameState.getPacmanPosition()
    remainfood = currentGameState.getFood().asList()
    ghostposition = currentGameState.getGhostStates()
    remaincapsule = currentGameState.getCapsules()
    
    if len(remaincapsule) > 0:  #to get close capsule is better
        capsuleDistances = []
        for capsule in remaincapsule:
            capsuleDistances.append(util.manhattanDistance(capsule,pacmanPosition))
        currentscore = currentscore - min(capsuleDistances)
    
    if len(remainfood) > 0:   #to get close closest food is better
        foodDistances = []
        for food in remainfood:
            foodDistances.append(util.manhattanDistance(food,pacmanPosition))
        currentscore = currentscore - min(foodDistances)
    
    if currentGameState.hasFood(pacmanPosition[0],pacmanPosition[1]): currentscore = currentscore + 50  #will increase score if it can reach the food 
    if currentGameState.isLose(): currentscore = currentscore - 100000 #significant decrease in score if state is loss
    if currentGameState.isWin(): currentscore = currentscore + 100000 #significant increase in score if state is loss
    for capsule in remaincapsule:     #will increase score if it can reach the capsule 
        if pacmanPosition == capsule : currentscore = currentscore + 100


    for ghost in ghostposition: 
        if ghost.scaredTimer > 0:   #if ghost is scared then being closeto them is better
            currentscore = currentscore + util.manhattanDistance(ghost.getPosition(),pacmanPosition)
        else:   
            currentscore = currentscore - util.manhattanDistance(ghost.getPosition(),pacmanPosition)

    currentscore = currentscore - len(remainfood)  

    return currentscore
# Abbreviation
better = betterEvaluationFunction
