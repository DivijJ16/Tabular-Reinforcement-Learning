import itertools
class Agent:

    def get_region(self, stack, visited):
      ans = []
      while len(stack) > 0:
            cell = stack.pop(-1)
            ans.append(cell)
            for [dx,dy] in self.actions.values():
                ncell = (cell[0] + dx, cell[1] + dy)
                if(ncell in visited and not visited[ncell]):
                    visited[ncell] = True
                    stack.append(ncell)
      return ans 
          
      
    def get_pirate_region(self):
        visited = {cell: False for cell in self.pirate_cells}
        stack = [self.p1]
        visited[self.p1] = True
        p1_region = self.get_region(stack,visited)
        stack = [self.p2]
        visited[self.p2] = True
        p2_region = self.get_region(stack,visited)
        return [p1_region,p2_region]
        
        
    def __init__(self, layout_file, prob_file):
        """
        Initialize the agent.

        Args:
            layout_file: Path to the grid layout file.
            prob_file: Path to the file containing environment probabilities.

        You may use this function to:
            - Read and store the grid layout.
            - Read and store transition probabilities.
            - Identify important locations such as the fort.
            - Construct the state space and transition model.
            - Initialize any data structures required for learning.
        """

        with open(layout_file,'r') as matrix:
          rows = [row.strip() for row in matrix.readlines()]
        self.n = len(rows)
        self.grid = rows
        self.start = None
        self.goal = None
        self.treasures = []
        self.land_cells = set()
        self.p1 = None
        self.p2 = None
        self.pirate_cells = set()

        for i in range(self.n):
            for j in range(self.n):
                curr = self.grid[i][j]
                if curr == 'S': self.start = (i, j)
                elif curr == 'L': self.land_cells.add((i, j))
                elif curr == 'F': self.goal = (i, j)
                elif curr == 'T': self.treasures.append((i, j))
                elif curr == '1': 
                    self.p1 = (i, j)
                    self.pirate_cells.add((i, j))
                elif curr == '2': 
                    self.p2 = (i, j)
                    self.pirate_cells.add((i, j))
                elif curr == '!': self.pirate_cells.add((i, j))
        self.treasures.sort(key=lambda x: (x[0], x[1])) # treasures[0] is the one with lesser i, or same i and lesser j
        
        with open(prob_file, 'r') as probs:
            p_rows = probs.readlines()
            
        self.prob_s = float(p_rows[0].strip())
        self.prob1 = list(map(float, p_rows[1].strip().split()))
        self.prob2 = list(map(float, p_rows[2].strip().split()))
        
        rewards = list(map(float, p_rows[3].strip().split()))
        self.r_s, self.r_t, self.r_f, self.r_p = rewards
        self.gamma = float(p_rows[4].strip())
        
        self.actions = {0: (1, 0), 1: (-1, 0), 2: (0, -1), 3: (0, 1)}
        
        # self.visited = {cell: False for cell in self.pirate_cells}
        
        self.p1_area,self.p2_area = self.get_pirate_region()
      
        self.states = []
        self.transitions = {}
        self.predecessors = {}
        self.util = {}
        self.build_states()
            

    def get_action(self, ship_location, pirate_locations, treasure_locations) -> int:
        """
        Choose an action for the current state.

        Args:
            ship_location: Tuple (x, y) representing the current
                location of the ship.

            pirate_locations: List containing the locations of all
                pirates. There can be at most 2 pirates.

            treasure_locations: List containing the locations of all
                treasures. There can be at most 2 treasures.

        Returns:
            An integer representing the selected action:

                0 -> UP
                1 -> DOWN
                2 -> LEFT
                3 -> RIGHT

        Note:
            This function may be called multiple times after
            `learn_policy()` has been executed.
        """

        # TODO
        return 0


    def learn_policy(self, time):
        """
        Learn a policy for navigating the Treasure Hunt environment.

        Args:
            time: Maximum time (in seconds) allowed for learning.

        The learned policy should be stored internally by the agent
        and subsequently used by `get_action()`.

        Returns:
            None
        """

        # TODO