import itertools
import numpy as np
import time

class Agent:
    
    def get_region(self, stack, visited):
        ans = []
        while len(stack) > 0:
            cell = stack.pop(-1)
            ans.append(cell)
            for dx, dy in self.actions.values():
                ncell = (cell[0] + dx, cell[1] + dy)
                if ncell in visited and not visited[ncell]:
                    visited[ncell] = True
                    stack.append(ncell)
        return ans 
    
    def get_pirate_region(self):
        visited = {cell: False for cell in self.pirate_cells}
        p1_region = []
        if self.p1:
            visited[self.p1] = True
            p1_region = self.get_region([self.p1], visited)
            
        p2_region = []
        if self.p2:
            visited[self.p2] = True
            p2_region = self.get_region([self.p2], visited)
            
        return [p1_region, p2_region]
    
    def fill_pirate_tm(self,p_area,p_cell_idx,tm,prob) :
        for cell in p_area :
            i = p_cell_idx[cell]
            for action, (dx,dy) in self.actions.items() :
                ncell = (cell[0]+dx, cell[1]+dy)
                if ncell not in p_area :
                    ncell = cell
                j = p_cell_idx[ncell]
                tm[i,j] += prob[action]
                
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
        self.ship_cells = []
        self.pirate_cells = []
        self.p1 = None
        self.p2 = None

        for i in range(self.n):
            for j in range(self.n):
                curr = self.grid[i][j]
                if curr == 'S': 
                    self.start = (i, j)
                    self.ship_cells.append((i,j))
                elif curr == 'W': self.ship_cells.append((i,j))
                elif curr == 'L': pass
                elif curr == 'F': 
                    self.goal = (i, j)
                    self.ship_cells.append((i,j))
                elif curr == 'T': 
                    self.treasures.append((i, j))
                    self.ship_cells.append((i,j))
                elif curr == '1': 
                    self.p1 = (i, j)
                    self.pirate_cells.append((i, j))
                    self.ship_cells.append((i,j))
                elif curr == '2': 
                    self.p2 = (i, j)
                    self.pirate_cells.append((i, j))
                    self.ship_cells.append((i,j))
                elif curr == '!': 
                    self.pirate_cells.append((i, j))
                    self.ship_cells.append((i,j))
                    
        self.actions = {0: (1,0), 1: (-1,0), 2: (0,-1), 3: (0,1)} #UDLR
        
        self.p1_area, self.p2_area = self.get_pirate_region()
        
        self.treasures.sort(key=lambda x: (x[0], x[1])) # treasures[0] is the one with lesser i, or same i and lesser j
        
        with open(prob_file, 'r') as probs:
            p_rows = probs.readlines()
            
        self.prob_s = float(p_rows[0].strip())
        self.prob1 = list(map(float, p_rows[1].strip().split()))
        self.prob2 = list(map(float, p_rows[2].strip().split()))
            
        rewards = list(map(float, p_rows[3].strip().split()))
        self.r_s, self.r_t, self.r_f, self.r_p = rewards
        self.gamma = float(p_rows[4].strip())
        
        self.p1_cell_idx = {cell : idx for idx,cell in enumerate(self.p1_area)}
        self.p2_cell_idx = {cell : idx for idx,cell in enumerate(self.p2_area)}

        self.p1_trans = np.zeros((len(self.p1_area),len(self.p1_area)),dtype=np.float32)
        self.p2_trans = np.zeros((len(self.p2_area),len(self.p2_area)),dtype=np.float32)
        
        self.fill_pirate_tm(self.p1_area,self.p1_cell_idx,self.p1_trans,self.prob1)
        self.fill_pirate_tm(self.p2_area,self.p2_cell_idx,self.p2_trans,self.prob2)

        self.ship_cell_idx = {cell : idx for idx,cell in enumerate(self.ship_cells)}
        self.next_ship_cell = np.zeros((len(self.ship_cells),4),dtype=int)
        
        for cell in self.ship_cells :
            i = self.ship_cell_idx[cell]
            for action,(dx,dy) in self.actions.items() :
                ncell = (cell[0]+dx,cell[1]+dy)
                if ncell not in self.ship_cells :
                    ncell = cell
                self.next_ship_cell[i,action] = self.ship_cell_idx[ncell]
                
        self.invalid = np.zeros((len(self.ship_cells),len(self.p1_area),len(self.p2_area)),dtype=np.float32)
        for c1 in self.ship_cells :
            for c2 in self.p1_area :
                for c3 in self.p2_area :
                    if c1 == c2 or c1 == c3 :
                        self.invalid[self.ship_cell_idx[c1],self.p1_cell_idx[c2],self.p2_cell_idx[c3]] = 1.0
        
        self.bad_pen = np.einsum('ij,sjk,lk->sil', self.p1_trans, self.invalid, self.p2_trans)
        
        self.v = np.zeros((len(self.ship_cells),len(self.p1_area),len(self.p2_area),4),dtype=np.float32)
        self.policy = np.zeros((len(self.ship_cells),len(self.p1_area),len(self.p2_area),4),dtype=np.int8)

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
        
        where = self.ship_cell_idx[ship_location]
        p1 = self.p1_cell_idx[pirate_locations[0]]
        p2 = self.p2_cell_idx[pirate_locations[1]]
        
        treas = 0
        if self.treasures[0] in treasure_locations : treas += 1
        if self.treasures[1] in treasure_locations : treas += 2

        # TODO
        return self.policy[where,p1,p2,treas]


    def learn_policy(self, time_l):
        """
        Learn a policy for navigating the Treasure Hunt environment.

        Args:
            time: Maximum time (in seconds) allowed for learning.

        The learned policy should be stored internally by the agent
        and subsequently used by `get_action()`.

        Returns:
            None
        """
        
        start = time.time()
        while time.time() - start < time_l-0.5 :
            v_state_action = np.zeros((4,len(self.ship_cells),len(self.p1_area),len(self.p2_area),4),dtype=np.float32)
            
            for dir in range(4) :
                next_ship_cells = self.next_ship_cell[:,dir]
                for treasures in range(4) :
                    rewards = np.full(len(self.ship_cells),self.r_s,dtype=np.float32)
                    
                    new_v = np.zeros((len(self.ship_cells),len(self.p1_area),len(self.p2_area)),dtype=np.float32)
                    for i in range(len(self.ship_cells)) :
                        next_t = treasures
                        ncell = self.ship_cells[next_ship_cells[i]]
                        if ncell == self.goal :
                            rewards[i] += self.r_f
                            new_v[i] = 0.0
                            continue
                        elif ncell == self.treasures[0] and (treasures&1) :
                            rewards[i] += self.r_t
                            next_t -= 1
                        elif ncell == self.treasures[1] and (treasures&2) :
                            rewards[i] += self.r_t        
                            next_t -= 2
                            
                        new_v[i] = self.v[next_ship_cells[i],:,:,next_t]
                    
                    new_v[self.invalid[next_ship_cells] != 0.0] = 0.0
                    
                    exp_v = np.einsum('ij,sjk,lk->sil', self.p1_trans, new_v, self.p2_trans)
                    exp_r = rewards[:,None,None] + (self.r_p)*(self.bad_pen[next_ship_cells])
                    
                    exp_v = exp_r + self.gamma*exp_v
                    
                    for act_dir in range(4) :
                        p = self.prob_s if act_dir == dir else (1-self.prob_s)/3.0
                        v_state_action[act_dir,:,:,:,treasures] += p*exp_v
            
            next_v = np.max(v_state_action,axis=0)
            self.policy = np.argmax(v_state_action,axis=0)
            
            if np.max(np.abs(self.v-next_v)) < 1e-4 :
                self.v = next_v
                break
            
            self.v = next_v

        # TODO