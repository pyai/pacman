# searchAgents.py
# ---------------
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
This file contains all of the agents that can be selected to control Pacman.  To
select an agent, use the '-p' option when running pacman.py.  Arguments can be
passed to your agent using '-a'.  For example, to load a SearchAgent that uses
depth first search (dfs), run the following command:

> python pacman.py -p SearchAgent -a fn=depthFirstSearch

Commands to invoke other search strategies can be found in the project
description.

Please only change the parts of the file you are asked to.  Look for the lines
that say

"*** YOUR CODE HERE ***"

The parts you fill in start about 3/4 of the way down.  Follow the project
description for details.

Good luck and happy searching!
"""
from __future__ import division

from game import Directions
from game import Agent
from game import Actions
import util
import time
import search

class GoWestAgent(Agent):
    "An agent that goes West until it can't."

    def getAction(self, state):
        "The agent receives a GameState (defined in pacman.py)."
        if Directions.WEST in state.getLegalPacmanActions():
            return Directions.WEST
        else:
            return Directions.STOP

#######################################################
# This portion is written for you, but will only work #
#       after you fill in parts of search.py          #
#######################################################

class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search
    algorithm for a supplied search problem, then returns actions to follow that
    path.

    As a default, this agent runs DFS on a PositionSearchProblem to find
    location (1,1)

    Options for fn include:
      depthFirstSearch or dfs
      breadthFirstSearch or bfs


    Note: You should NOT change any code in SearchAgent
    """

    def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError, fn + ' is not a search function in search.py.'
        func = getattr(search, fn)
        if 'heuristic' not in func.func_code.co_varnames:
            print('[SearchAgent] using function ' + fn)
            self.searchFunction = func
        else:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError, heuristic + ' is not a function in searchAgents.py or search.py.'
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError, prob + ' is not a search problem type in SearchAgents.py.'
        self.searchType = globals()[prob]
        print('[SearchAgent] using problem type ' + prob)

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.searchFunction == None: raise Exception, "No search function provided for SearchAgent"
        starttime = time.time()
        problem = self.searchType(state) # Makes a new search problem
        self.actions  = self.searchFunction(problem) # Find a path
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP

class PositionSearchProblem(search.SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print 'Warning: this does not look like a regular search maze'

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

class StayEastSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the West side of the board.

    The cost function for stepping into a position (x,y) is 1/2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: .5 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn, (1, 1), None, False)

class StayWestSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the East side of the board.

    The cost function for stepping into a position (x,y) is 2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: 2 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn)

def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def euclideanHeuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################

class CornersProblem(search.SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function
    """

    def __init__(self, startingGameState, costFn = lambda x: 1, visualize=True):
        """
        Stores the walls, pacman's starting position and corners.
        """
        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top, right = self.walls.height-2, self.walls.width-2
        self.corners = ((1,1), (1,top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                print 'Warning: no food in corner ' + str(corner)
        self._expanded = 0 # DO NOT CHANGE; Number of search nodes expanded
        # Please add any code here which you would like to use
        # in initializing the problem
        "*** YOUR CODE HERE ***"
        self.costFn = costFn
        self.visualize = visualize
        self.width = right
        self.height = top
        #self.food = {(1, 1):False, (1,top):False, (right, 1):False, (right, top):False}
        #self.abce={'a':(1,top), 'b':(right, top), 'c':(right, 1), 'd':(1,1)} # 'a':left top, 'b':right top, 'c':right down, 'd':left down 
        #self.bite = (False, False, False, False)

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    #def get_abcd(self):
    #    return (self.food.get(self.abce.get('a')), self.food.get(self.abce.get('b')), self.food.get(self.abce.get('c')), self.food.get(self.abce.get('d')))

    def getStartState(self):
        """
        Returns the start state (in your state space, not the full Pacman state
        space)
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        return self.startingPosition + (False, False, False, False)

    def isGoalState(self, state):
        """
        Returns whether this search state is a goal state of the problem.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        isGoal = all(state[2:])

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state[0:2])
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
            For a given state, this should return a list of triples, (successor,
            action, stepCost), where 'successor' is a successor to the current
            state, 'action' is the action required to get there, and 'stepCost'
            is the incremental cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            # Add a successor state to the successor list if the action is legal
            # Here's a code snippet for figuring out whether a new position hits a wall:
            #   x,y = currentPosition
            #   dx, dy = Actions.directionToVector(action)
            #   nextx, nexty = int(x + dx), int(y + dy)
            #   hitsWall = self.walls[nextx][nexty]

            "*** YOUR CODE HERE ***"
            x, y, a, b, c, d = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)

            if not self.walls[nextx][nexty]:
                nextState_ = (nextx, nexty)
                if nextState_ == self.corners[0]:
                    a = True
                elif nextState_ == self.corners[1]:
                    b = True
                elif nextState_ == self.corners[2]:
                    c = True
                elif nextState_ == self.corners[3]:
                    d = True
                else:
                    pass

                nextState = nextState_ + (a, b, c, d)
                cost = self.costFn(nextState)
                successors.append( (nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state[0:2])

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999.  This is implemented for you.
        """
        if actions == None: return 999999
        x,y= self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
        return len(actions)


def cornersHeuristic(state, problem):
    """
    A heuristic for the CornersProblem that you defined.

      state:   The current search state
               (a data structure you chose in your search problem)

      problem: The CornersProblem instance for this layout.

    This function should always return a number that is a lower bound on the
    shortest path from the state to a goal of the problem; i.e.  it should be
    admissible (as well as consistent).
    """
    corners = problem.corners # These are the corner coordinates
    width = problem.width
    height = problem.height
    walls = problem.walls # These are the walls of the maze, as a Grid (game.py)

    "*** YOUR CODE HERE ***"
    """
    1          3     b          d
                  =  

    0          2     a          c
    """
    # euclidean distance
    x, y, a, b, c, d = state
    foods = sum([ 1 for _ in (a, b, c, d) if not _])

    distance = dict([(_, abs(x - _[0]) + abs(y - _[1])) for _ in corners]) 
    abcd = dict([ (_1, _2) for _1, _2 in zip(corners, (a, b, c, d))]) 

    h = 0
    if foods == 4:
        h += min(distance.values())
        if width < height:
            h += 2 * (width-1) + (height-1)
        else:
            h += 2 * (height-1) + (width-1)
    elif foods == 3:
        if a or d:
            h += min((distance[corners[1]], distance[corners[2]])) + (width-1) + (height-1)
        elif b or c:
            h += min((distance[corners[0]], distance[corners[3]])) + (width-1) + (height-1)
    elif foods == 2:
        _1, _2 = [ _ for _ in corners if not abcd[_]]
        h += min([distance[_] for _ in corners if not abcd[_]])
        h += abs(_1[0] - _2[0]) + abs(_1[1] - _2[1])
    elif foods == 1:
        h += min([distance[_] for _ in corners if not abcd[_]])

    return h

class AStarCornersAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, cornersHeuristic)
        self.searchType = CornersProblem

class FoodSearchProblem:
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """
    def __init__(self, startingGameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0 # DO NOT CHANGE
        self.heuristicInfo = {} # A dictionary for the heuristic to store information

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1 # DO NOT CHANGE
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()
                nextFood[nextx][nexty] = False
                successors.append( ( ((nextx, nexty), nextFood), direction, 1) )
        return successors

    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        x,y= self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost

class AStarFoodSearchAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, foodHeuristic)
        self.searchType = FoodSearchProblem

def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state
    "*** YOUR CODE HERE ***"
    import copy
    def manhattan(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def adjacent(a, b):
        # if manhattan adjacent then return True, otherwise False
        return manhattan(a, b) == 1

    foods = foodGrid.asList()

    # get connections
    food_edge = list()
    for i, f1 in enumerate(foods):
        for j, f2 in enumerate(foods):
            if adjacent(f1, f2):
                food_edge.append((f1, f2))

    # get adjacentlist
    food_adj = dict()
    for f in foods:
        a = set()
        for e in food_edge:
            if f == e[0]:
                a.add(e[1])
            elif f == e[1]:
                a.add(e[0])
        food_adj[f] = copy.deepcopy(list(a))

    # get connected components
    food_cc = list()
    for f in foods:
        explored = set()
        stack = util.Stack()
        stack.push(f)
        while not stack.isEmpty():
            node = stack.pop()
            explored.add(node)
            [stack.push(_) for _ in food_adj[node] if _ not in explored]
        else:
            if explored not in food_cc:
                food_cc.append(explored)


    # get manhattan distance between connected components
    pacman_cloest = 0
    min_manhatan = dict()
    for i, cc1 in enumerate(food_cc):
        m = float("Inf") 
        for j, cc2 in enumerate(food_cc):
            if i != j:
                for _1 in cc1:
                    for _2 in cc2:
                        if manhattan(_1, _2) < m:
                            m = manhattan(_1, _2)
        min_manhatan[tuple(cc1)] = m







    #print "foods:", foods
    #print "food_edge:", food_edge
    #print "food_adj:", food_adj
    #print "food_cc:", food_cc
    #print "min_manhatan:", min_manhatan
    #print "len(foods) + sum(min_manhatan.values()):", len(foods) + sum(min_manhatan.values())

    #print "========="
    #print problem.walls
    #print "........"
    #print foodGrid
    #print "========="

    #while True:
    #    pass
    if min_manhatan:   
        return len(foods) + sum(list(min_manhatan.values()) ) - min(list(min_manhatan.values())) - len(min_manhatan)
    else:
        return len(foods) + sum(list(min_manhatan.values()) )

def foodHeuristic(state, problem):
    position, foodGrid = state
    "*** YOUR CODE HERE ***"
    import copy
    def manhattan(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def adjacent(a, b):
        # if manhattan adjacent then return True, otherwise False
        return manhattan(a, b) == 1

    foods = list()
    foods = foodGrid.asList()

    return len(foods)

def foodHeuristic(state, problem):
    position, foodGrid = state
    "*** YOUR CODE HERE ***"
    import copy
    def manhattan(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def adjacent(a, b):
        # if manhattan adjacent then return True, otherwise False
        return manhattan(a, b) == 1

    foods = foodGrid.asList()
    foods.append(position)
    foodn = len(foods)

    # adjacent matrix
    adj = [[None,]*foodn for i in range(foodn)]
    for i, f1 in enumerate(foods):
        for j, f2 in enumerate(foods):
            if i ==j:
                adj[i][j] = 0
            else:
                adj[i][j] = manhattan(f1, f2)


    # start from 0
    walked = list()
    i = 0
    walked.append(i)

    cost = 0
    while len(walked) < foodn:
        row = [ 9999999999999 if j in walked else f for j, f in enumerate(adj[i])  ]
        cost += min(row)
        i = row.index(min(row))
        walked.append(i)

    #print "adj:", adj
    #print "cost:", cost

    return cost


def foodHeuristic(state, problem):
    position, foodGrid = state
    "*** YOUR CODE HERE ***"
    #print "foodGrid.asList():", foodGrid.asList()
    source = position
    import copy
    def manhattan(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def euclidean(a, b):
        return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5

    def distance(a, b):
        return manhattan(a, b)

    def adjacent(a, b):
        # if manhattan adjacent then return True, otherwise False
        return manhattan(a, b) == 1

    foods = list()
    
    #height = foodGrid.height
    #width = foodGrid.width
    q1, q2, q3, q4 = list(), list(), list(), list()
    foods = foodGrid.asList()
    foods.append(source)
    for f in foods:
        x, y = f
        if (x, y) == source:
            continue
        elif x >= position[0] and y > position[1]:
            q1.append((x, y))
        elif x < position[0] and y >= position[1]:
            q2.append((x, y))
        elif x <= position[0] and y < position[1]:
            q3.append((x,y))
        else:
            q4.append((x,y))

    
    foodn = len(foods)


    if foodn == 1:
        return 0

    adj = dict()
    for f1 in foods:
        for f2 in foods:
            if f1 not in adj:
                adj[f1] = dict()

            # ignore
            if f1 == f2:
                continue
            else:
                (adj[f1])[f2] = distance(f1, f2)

    #nbr = dict()
    #for f1 in foods:
    #    for f2 in foods:
    #        if f1 not in nbr:
    #            nbr[f1] = dict()
#
    #        # ignore
    #        if manhattan(f1, f2) == 1:
    #            if f2 not in nbr[f1]:
    #                (nbr[f1])[f2] = dict()
    #            else:
    #                (nbr[f1])[f2] = (nbr[f1])[f2] + 1

    qcost = list()
    qwalk = list()
    for q in (q1, q2, q3, q4):
        this = source
        cost = 0
        walked = [source]
        while len(walked) < len(q)+1:
            mk = None
            mv = float('inf')

            for k, v in adj[this].items():
                if k in q and k not in walked and v < mv:
                    mk, mv = k, v

            # break even
            ks = [k for k, v in adj[this].items() if v == mv and k in q ]

            print "ks:", ks
            #n = float('inf')
            #for k in ks:
            #    if k in nbr[this]:
            #        if (nbr[this])[k] < n:
            #            mk, n = k, (nbr[this])[k]
            

            cost += mv
            this = mk
            walked.append(this)

        if cost == 0:
            cost = float('inf')

        qwalk.append(walked)
        qcost.append(cost)


    this = qwalk[ qcost.index(min(qcost)) ][1]
    #print "this:", this
    cost = (adj[source])[this]
    walked = [source, this]
    while len(walked) < len(foods):
        mk = None
        mv = float('inf')
        for k, v in adj[this].items():
            if k not in walked and v < mv:
                mk, mv = k, v

        cost += mv
        this = mk
        walked.append(this)

    print "source:", source
    print "walked:", walked
    print "cost:", cost

    return cost


def foodHeuristic(state, problem):
    position, foodGrid = state
    "*** YOUR CODE HERE ***"
    #print "foodGrid.asList():", foodGrid.asList()
    source = (1, position, (position, ))
    import copy
    def manhattan(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def euclidean(a, b):
        return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5

    def distance(a, b):
        return manhattan(a, b)

    def adjacent(a, b):
        # if manhattan adjacent then return True, otherwise False
        return manhattan(a, b) == 1

    foods = list()
    
    q1, q2, q3, q4 = list(), list(), list(), list()
    foods = foodGrid.asList()
    foodn = len(foods)

    if foodn == 0:
        return 0

    food_edge = list()

    if len(foods) >1:
        for f1 in foods:
            for f2 in foods:
                if adjacent(f1, f2):
                    food_edge.append((f1, f2))
    else:
        food_edge.append((foods[0], foods[0]))

    # get adjacentlist
    food_adj = dict()
    for f in foods:
        if f not in food_adj: 
            a = set()
        else:
            a = set(food_adj[f])

        for e in food_edge:
            if f in e:
                a.add(e[0])
                a.add(e[1])

        food_adj[f] = copy.deepcopy(list(a))

    # get connected components
    def center(l):
        xsum, ysum = 0, 0
        for x, y in l:
            xsum += x
            ysum += y
        else:
            xmean = xsum/len(l)
            ymean = ysum/len(l)
        
        return (xmean, ymean)


    # get connected components
    food_cc = list()
    for f in foods:
        explored = set()
        stack = util.Stack()
        stack.push(f)
        while not stack.isEmpty():
            node = stack.pop()
            explored.add(node)
            [stack.push(_) for _ in food_adj[node] if _ not in explored]
        else:
            if explored not in food_cc:
                food_cc.append(explored)

    food_cc2 = [source, ]
    for fc in food_cc:
        food_cc2.append((len(fc), center(fc), tuple(fc) ))


    adj = dict()
    for f1 in food_cc2:
        for f2 in food_cc2:
            if f1 not in adj:
                adj[f1] = dict()

            # ignore
            if f1 == f2:
                continue
            else:
                (adj[f1])[f2] = distance(f1[1], f2[1])


    for f in food_cc2:
        x, y = f[1]
        sourcex, sourcey = source[1]
        if (x, y) == position:
            continue
        elif x >= sourcex and y > sourcey:
            q1.append(f)
        elif x < sourcex and y >= sourcey:
            q2.append(f)
        elif x <= sourcex and y < sourcey:
            q3.append(f)
        else:
            q4.append(f)

    def direction(a, b):
        # a is news related to b
        news = set()
        if a[0] < b[0]:
            news.add('w')
        elif a[0] > b[0]:
            news.add('e')

        if a[1] < b[1]:
            news.add('s')
        elif a[1] > b[1]:
            news.add('n')

        return news

    def estimate_cc_cost(start, end):
        news = direction(end, start)
        center = end[1]
        cost = 0
        for p in end[2]:
            if news & direction(p, center):
                cost += 1 if distance(p, center) > 1 else 0
        return cost


    qcost = list()
    qwalk = list()
    for q in (q1, q2, q3, q4):
        this = source
        cost = 0
        walked = [source, ]
        while len(walked) < len(q)+1:
            mk = None
            mv = float('inf')

            for k, v in adj[this].items():
                if k in q and k not in walked:
                    if v + estimate_cc_cost(this, k) < mv:
                        mk = k
                        mv = v + estimate_cc_cost(this, k)


            cost += mv
            this = mk
            walked.append(this)

        if cost == 0:
            cost = float('inf')

        qwalk.append(walked)
        qcost.append(cost)


    this = qwalk[ qcost.index(min(qcost)) ][1]
    cost = (adj[source ])[this] 
    walked = [source, this]
    while len(walked) < len(food_cc2):
        mk = None
        mv = float('inf')
        for k, v in adj[this].items():
            if k not in walked:
                if v + estimate_cc_cost(this, k) < mv:
                    mk = k
                    mv = v + estimate_cc_cost(this, k)


        cost = cost + mv
        this = mk
        walked.append(this)


    return cost




mask_array = None
def gen_mask_array(n):
    print "start gen_mask_array with", n, "bits"
    global mask_array

    if mask_array is None:
        mask_array = dict()
        for i in range(2**n):
            pattern = ("{:0>"+str(n)+"}").format( str(bin(i))[2:] )
            if pattern.count('1') not in mask_array:
                mask_array[pattern.count('1')] = list()

            mask_array[pattern.count('1')].append(pattern)
    print "end gen_mask_array"  


def foodHeuristic_bellman_karp(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state
    "*** YOUR CODE HERE ***"
    source = position
    import copy
    def manhattan(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def adjacent(a, b):
        # if manhattan adjacent then return True, otherwise False
        return manhattan(a, b) == 1

    def mask_select(item, mask):
        return [ i for i, m in zip(item, mask) if m]

    foods = list()
    #foods.append(source)
    height = foodGrid.height
    width = foodGrid.width

    for x in range(width):
        for y in range(height):
            if foodGrid[x][y]:
                foods.append((x, y))

    foodn = len(foods)
    import time
    sta = time.time()
    gen_mask_array(foodn)
    end = time.time()
    print "time:", end - sta

    # adjacent matrix
    adj = [[None,]*foodn for i in range(foodn)]
    for i, f1 in enumerate(foods):
        for j, f2 in enumerate(foods):
            if i ==j:
                adj[i][j] = 0
            else:
                adj[i][j] = manhattan(f1, f2)

    #source_to_adj = [manhattan(source, f) for f in foods] # position to 0, 1, ... foodn -1

    print "adj:", adj

    memory = list() # a list of dict

    print 'source:', source, 'foods:', foods
    print 'foodn:', foodn
    #import itertools


    # print "mask_array:", mask_array
    for i in range(foodn):
        masks = mask_array[i]
        print "i =", i
        #print "masks:", masks

        memory.append(dict())
        for m in masks:
            mask = [ True if _ =='1' else False for _ in list(m)]
            food_subset = mask_select(foods, mask)

            mint = int(''.join(m), base=2)

            food_rest = set(foods) - set(food_subset)
            for r in food_rest:
                if mint == 0:
                    (memory[i])[(r, mint)] = manhattan(source, r)
                else:
                    (memory[i])[(r, mint)] = min([ manhattan(r, key[0]) + value for key, value in (memory[i-1]).items() if key[0] != r])
                     



    end = time.time()
    print "time:", end - sta

    return 0 if foodn == 0 else min(memory[foodn-1].values())





         



class ClosestDotSearchAgent(SearchAgent):
    "Search for all food using a sequence of searches"
    def registerInitialState(self, state):
        self.actions = []
        currentState = state
        while(currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState) # The missing piece
            self.actions += nextPathSegment
            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    t = (str(action), str(currentState))
                    raise Exception, 'findPathToClosestDot returned an illegal move: %s!\n%s' % t
                currentState = currentState.generateSuccessor(0, action)
        self.actionIndex = 0
        print 'Path found with cost %d.' % len(self.actions)

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition()
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState)

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    successor function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """

    def __init__(self, gameState):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        x,y = state

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(search.bfs(prob))
