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

"*** YOUR CODE HERE ***"
class pacman_node:
    problem_ = None
    explored = dict()
    def __init__(self, state, parent):
        self.state = state
        self.parent = parent

    def __str__(self):
        return str((self.state, self.parent))

    @staticmethod
    def set(problem):
        pacman_node.problem_ = problem

    @staticmethod
    def unset():
        pacman_node.problem_ = None
        pacman_node.explored = dict()

    def getSuccessors(self):
        child = pacman_node.problem_.getSuccessors(self.state[0])
        child = [ pacman_node(_, self.state[0]) for _ in child if _[0] not in pacman_node.explored]

        return child

    def add_node_to_explored(self):
        pacman_node.explored[self.state[0]] = (self.parent, self.state[1])

    def back_trace(self):
        # find all ancestor backward from this node
        state = self.state
        parent = pacman_node.explored[state[0]]
        actions = list()
        while parent != (None, None):
            actions.insert(0, parent[1])
            state = parent
            parent = pacman_node.explored[state[0]]

        return actions

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

class pacman_node:
    problem_ = None
    explored = dict()
    heuristic_ = None
    def __init__(self, state, parent):
        self.state = state
        self.parent = parent

    def __str__(self):
        return str((self.state, self.parent))

    @staticmethod
    def set(problem, heuristic = nullHeuristic):
        pacman_node.problem_ = problem
        pacman_node.heuristic_ = staticmethod(heuristic)

    @staticmethod
    def unset():
        pacman_node.problem_ = None
        pacman_node.explored = dict()
        pacman_node.heuristic_ = None


    # for ucs
    @classmethod
    def cost(cls, node):
        if node.parent is not None:
            g = node.state[2] + node.parent[2]
            h = pacman_node.heuristic_(node.state[0], cls.problem_)
            node.updateState((node.state[0], node.state[1], g+h ))
            return node.state[2]
        else:
            return 0 + (pacman_node.heuristic_)(node.state[0], cls.problem_)
      
    def updateState(self, newState):
        del self.state
        self.state = newState

    def getSuccessors(self):
        child = pacman_node.problem_.getSuccessors(self.state[0])
        child = [ pacman_node(_, self.state) for _ in child if _[0] not in pacman_node.explored]

        return child

    def add_node_to_explored(self):
        pacman_node.explored[self.state[0]] = (self.parent, self.state[1])

    def back_trace(self):
        # find all ancestor backward from this node
        state = self.state
        parent = pacman_node.explored[state[0]]
        actions = list()
        while parent != (None, None):
            actions.insert(0, parent[1])
            state = parent[0]
            parent = pacman_node.explored[state[0]]

        return actions

def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch_v1(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    from game import Directions
    stack = util.Stack()
    trace = util.Stack()
    blocked = set()
    action = list()

    x = 0
    state = (problem.getStartState(), None, None, None)
    trace.push(state) # some times fork happen at start point

    while not problem.isGoalState(state[0]):
        node, act, cost, parent = state
        child = problem.getSuccessors(node)

        child = [ _ + (node, ) for _ in child ] # add parent information
        child = [ _ for _ in child if _[0] != parent and (_[0], node) not in blocked and _ not in trace.list ] # remove parent(repeated) node in child

        if not child: # if child is empty, back trace
            fork = stack.list[-1]
            back = trace.list[-1]
            while fork[3] != back[0]: # fork's parent is the start point of fork
                back = trace.pop()
                blocked.add((back[0],  back[3]))
                back = trace.list[-1]

        else:
            [ stack.push(_) for _ in child ] # push child into stack

        # explore one step ahead 
        state = stack.pop()
        trace.push(state)

    else:
        action = [ _[1] for _ in trace.list]
        # dump first acton None of start point
        action = action[1:]

    #util.raiseNotDefined()
    return action


def depthFirstSearch_v2(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    stack = util.Stack()
    trace = util.Stack()
    explored = set()


    state = (problem.getStartState(), None, 0, None)

    while not problem.isGoalState(state[0]):
        node, act, cost, parent = state
        child = problem.getSuccessors(node)
        ancestor = [ _[0] for _ in trace.list]

        child = [ _ + (node,) for _ in child ] # add parent 
        child = [ _ for _ in child if _[0] not in ancestor and _[0] not in explored] # remove parent(repeated) node in child

        if not child: # if child is empty, back trace
            explored.add(node)
            state = trace.pop()
        else:
            trace.push(state)
            [ stack.push(_) for _ in child ] # push child into stack
            state = stack.pop()


    else:
        trace.push(state)
        action = [ _[1] for _ in trace.list]
        # dump first acton None of start point
        action = action[1:]

    #util.raiseNotDefined()
    return action

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    pacman_node.set(problem)
    stack = util.Stack()
    node = pacman_node((problem.getStartState(), None, 0), None)

    while not problem.isGoalState(node.state[0]):
        node.add_node_to_explored()
        child = node.getSuccessors()
        #[stack.push(c) for c in child if c.state[0] not in [_.state[0] for _ in stack.list] ] # BFS
        [stack.push(c) for c in child] # DFS
        node = stack.pop()
    else:
        node.add_node_to_explored()
        action = node.back_trace()
        pacman_node.unset()

    #util.raiseNotDefined()
    return action

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    pacman_node.set(problem)
    stack = util.Queue()
    node = pacman_node((problem.getStartState(), None, 0), None)

    x = 0
    while not problem.isGoalState(node.state[0]):
        node.add_node_to_explored()
        child = node.getSuccessors()
        [stack.push(c) for c in child if c.state[0] not in [_.state[0] for _ in stack.list] ] # BFS
        # [stack.push(c) for c in child] # DFS
        node = stack.pop()

    else:
        node.add_node_to_explored()
        action = node.back_trace()
        pacman_node.unset()

    #util.raiseNotDefined()
    return action


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    pacman_node.set(problem)
    stack = util.PriorityQueueWithFunction(pacman_node.cost)
    node = pacman_node((problem.getStartState(), None, 0), None)

    x = 0
    while not problem.isGoalState(node.state[0]):
        node.add_node_to_explored()
        child = node.getSuccessors()
        [stack.push(c) for c in child if c.state[0] not in [_[2].state[0] for _ in stack.heap] ] # BFS
        # [stack.push(c) for c in child] # DFS
        node = stack.pop()

    else:
        node.add_node_to_explored()
        action = node.back_trace()
        pacman_node.unset()

    return action

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    pacman_node.set(problem, heuristic)
    stack = util.PriorityQueueWithFunction(pacman_node.cost)
    node = pacman_node((problem.getStartState(), None, 0), None)

    x = 0
    while not problem.isGoalState(node.state[0]):
        node.add_node_to_explored()
        child = node.getSuccessors()
        [stack.push(c) for c in child if c.state[0] not in [_[2].state[0] for _ in stack.heap] ] # BFS
        # [stack.push(c) for c in child] # DFS
        node = stack.pop()

    else:
        node.add_node_to_explored()
        action = node.back_trace()
        pacman_node.unset()

    return action


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
