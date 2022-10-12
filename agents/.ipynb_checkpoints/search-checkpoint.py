# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Grid

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    
    #Initialized frontier to be a stack
    #path is the variable used to keep track of  the final list of path actions which is dfs' output.
    #visited used to keep track of the visited/expanded nodes.
    frontier = util.Stack()
    path = []
    visited = []
    
    #Initially, push the start state with an empty path and zero cost to frontier.
    #livePath is a variable that saves current node's last path
    #liveCost is a variable that saves the total cost for reaching the current state.
    frontier.push((problem.getStartState(), [], 0))
    liveNode, path, liveCost = frontier.pop()
    
    #Unconditional for loop, that returns the final list of actions.
    while True:
        
        #This if block simply returns the path variable when the current node is the final goal node.
        if problem.isGoalState(liveNode):
            return path
        
        #If the frontier is empty when the goal hasn't been met, print error string.
        if frontier.isEmpty is True:
            print("Failure: The frotier is empty before the retrieval of the goal state.")    
        
        #Whenever the current node hasn't been expanded,...
        #... we save the successor states to temporary variable similar to livePath and liveCost...
        #... and push them to the frontier.
        if liveNode not in visited:
            for nextNode, nextPath, nextCost in problem.getSuccessors(liveNode):
                    frontier.push((nextNode, path + [nextPath], nextCost))
        
        #We simply add the current node, which has been expanded now, to the visited list
        #And we pop the last element from the frontier to the appropriate variables for the next loop. 
        visited.append(liveNode)
        liveNode, path, liveCost = frontier.pop()
        
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    my_queue = util.Queue()  #initialize empty queue
    visited = set()  #Keep track of visited nodes using a set
    path = []  #Keep track of the final path using a list of moves (north,south etc.)
    parentMap = {(problem.getStartState(),'Stop',0):None}  #Keep track of how to reach a node using a dictionary where key is the current node and value is the parent node
    my_queue.push((problem.getStartState(),'Stop',0))
    
    #Please note that bfs is implemented just like dfs with the frontier as a queue instead of a stack.
    frontier = util.Queue()
    path = []
    visited = []
    
    frontier.push((problem.getStartState(), [], 0))
    liveNode, path, liveCost = frontier.pop()
    
    while True:
                
        if problem.isGoalState(liveNode):
            return path
        
        if frontier.isEmpty is True:
            print("Failure: The frotier is empty before the retrieval of the goal state.")    

        if liveNode not in visited:    
            for nextNode, nextPath, nextCost in problem.getSuccessors(liveNode):
                    frontier.push((nextNode, path + [nextPath], nextCost))
        
        #The only differende from bfs here,...
        #... where the first elment is popped instead of the last element.
        visited.append(liveNode)
        liveNode, path, liveCost = frontier.pop()
        
    return False
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    priorityQueue = util.PriorityQueue()  #create empty priority queue
    visited = set()  #be able to keep track of visited nodes using a set
    path = []  #be able to keep track of the final path using a list of directions
    parentMap = {(problem.getStartState(),'Stop',0):None}  #be able to keep track of how to reach a node using a dictionary which...
                                                            #... key is current node and value is the parent node
    priorityQueue.push((problem.getStartState(),'Stop',0),0)
    costToReach = {problem.getStartState():0}  #be able to keep track of the cost for reaching each node using a dictionary
    

    while not priorityQueue.isEmpty():  #as long as queue has value inside, it will keep going
        currentNode = priorityQueue.pop()
        currentState = currentNode[0]
        if currentState in visited : continue    #in case of queue contain the same node twice,...
                                                    # ...it helps to skip that same node to add visited
        visited.add(currentState)
       
        
        if problem.isGoalState(currentState):   #if it reached goal
           
            while parentMap[currentNode] != None:
                path.append(currentNode[1])
                currentNode = parentMap[currentNode]
           
            path.reverse()
            
            return path  #this list is the final version of the path

        for successor in problem.getSuccessors(currentState):   #if it didn't reached goal add succesors
            if successor[0] not in visited:
                parentMap[successor] = currentNode
                costToReach[successor[0]] = successor[2] + costToReach[currentNode[0]]
                priorityQueue.push(successor, costToReach[successor[0]])  #push cost of going to the successor + cost to reach the...
                                                                            #... current node to the priority queue
        
        
    return False
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    priorityQueue = util.PriorityQueue()  #create empty priority queue
    visited = set()  #Keep track of visited nodes using a set
    path = []   #be able to keep track of the final path using a list of directions
    parentMap = {(problem.getStartState(),'Stop',0):None}  #be able to keep track of how to reach a node using a dictionary...
                                                            #... which key is current node and value is the parent node
    priorityQueue.push((problem.getStartState(),'Stop',0),0)
    costToReach = {problem.getStartState():0}   #be able to keep track of the cost for reaching each node using a dictionary
    

    while not priorityQueue.isEmpty():
        currentNode = priorityQueue.pop()
        currentState = currentNode[0]
        if currentState in visited : continue     #in case of queue contain the same node twice,...
                                                    #... it helps to skip that same node to add visited
        visited.add(currentState)
     
        
        if problem.isGoalState(currentState):  #if it reached goal
            while parentMap[currentNode] != None:
                path.append(currentNode[1])
                currentNode = parentMap[currentNode]
            path.reverse()
           
            return path  #this list is the final version of the path

        for successor in problem.getSuccessors(currentState):  #if it didn't reached goal add succesors
            if successor[0] not in visited:
                parentMap[successor] = currentNode
                costToReach[successor[0]] = costToReach[currentNode[0]] + successor[2]
                priorityQueue.push(successor, costToReach[successor[0]] + heuristic(successor[0],problem))  
                #push successor with priority successor[2] -the cost of going to the successor- to the priority queue
        
    return False
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
