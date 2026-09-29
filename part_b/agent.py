from env import HighwayEnv
import time as tm
import random as rnd

class Agent:

    def state_to_idx(self,speed, lane, min_distances): #convert the state tuple {speed,lane,min_distances} to a 720 state index
      # speed belongs to {0,1,2,3} , d_curr in {0,1,2,3,4} , d_left/d_right in {0,1,2,3,4,5}
      if lane:
        d_left = min_distances[lane-1]
      else:
        d_left = 5 # indicates a wall ,ie no left lane exists!
      if lane<3:
        d_right = min_distances[lane+1]
      else:
        d_right = 5 # indicates a wall ,ie no right lane exists!
      d_curr = min_distances[lane]
      
      return speed*180 + d_curr*36 + d_left*6 + d_right # unique hash for each of the 4-tuples {speed,d_curr,d_left,d_right}
    
    def get_mirror_idx(self, origidx): # returns the index of the mirror state of the curr state represented by origidx
      d_right = origidx%6
      rem = origidx//6
      d_left = rem%6
      rem = rem//6
      d_curr = rem%5
      rem = rem//5
      speed = rem
      return speed*180 + d_curr*36 + d_right*6 + d_left # unique hash for the swapped d_left, d_right mirror state

      
    def __init__(self, env: HighwayEnv, discount_factor = 0.99):
        """
        Initialize the agent.

        Args:
            env: The HighwayEnv environment. Students may use the
                 environment to access its parameters and dynamics.
        """

        self.env = env
        self.df = discount_factor
        self.cnt_s = 720 # 4 speeds * 6 d_left [state 5 means no left lane] * 6 d_right * 5 d_curr
        self.cnt_a = 5
        self.q_table = [[3.0] * self.cnt_a for state in range(self.cnt_s)] # stores q[s,a] values
        self.fr = [[0] * self.cnt_a for state in range(self.cnt_s)] # fr[s,a] = cnt of how many times has agent taken action a from state s

        self.pi = [0] * self.cnt_s
        self.mirror_a = {0: 0, 1: 1, 2: 3, 3: 2, 4: 4}
        
    def learn_policy(self, time_limit):
        """
        Learn a policy for controlling the car.

        Args:
            time: Maximum time in seconds allowed for learning.

        The learned policy should be stored internally and used by
        `get_action()`.

        Returns:
            None
        """
        start_t = tm.time()
        tot_time = time_limit - 1.0
        elapsed_time = 0.0
        step_cnt = 0
        epsilon = 0.3
        
        Q = self.q_table
        N = self.fr
        gamma = self.df
        mirror_map = self.mirror_a
        
        while elapsed_time < tot_time:
          obs = self.env.reset()
          speed,lane,min_distances = obs
          state_idx = self.state_to_idx(speed,lane,min_distances)
          done = False
          
          while not done:
            step_cnt += 1
            
            if not step_cnt & 255:
              elapsed_time = tm.time() - start_t
              if elapsed_time >= tot_time:
                break
              epsilon = max(0.02, 0.3 * (1 - elapsed_time / (0.7 * tot_time)))
            q_values = Q[state_idx]
            
            if rnd.random() < epsilon:
              a = rnd.randrange(self.cnt_a)
            else:
              a = q_values.index(max(q_values))
            
            next_obs , r, done = self.env.step(a)
            new_speed , new_lane, new_mindist = next_obs
            state_idx2 = self.state_to_idx(new_speed,new_lane,new_mindist)
            
            new_q_val = r
            if not done or r>=0: # this means simulation hasnt ended yet!
              new_q_val += gamma*max(Q[state_idx2])
            
            n = N[state_idx][a]
            alpha = max(0.02, 0.5 / (1 + n / 10.0))  
            N[state_idx][a] += 1
            q_values[a] = q_values[a] + alpha * (new_q_val - q_values[a])
            
            state_idx_m = self.get_mirror_idx(state_idx)
            state_idx2_m = self.get_mirror_idx(state_idx2)
            a_m = mirror_map[a]
            q_values_m = Q[state_idx_m]
            new_q_val_m = r
            if not done or r>=0 :
              new_q_val_m += gamma * max(Q[state_idx2_m])
            n_m = N[state_idx_m][a_m]
            alpha = max(0.02, 0.5 / (1 + n_m / 10.0))  
            N[state_idx_m][a_m] += 1
            q_values_m[a_m] += alpha * (new_q_val_m - q_values_m[a_m])
            
            state_idx = state_idx2
        self.pi = [q.index(max(q)) for q in Q]
            


    def get_action(self, speed, lane, min_dist):
        """
        Select an action for the current state.

        Args:
            speed: Current speed of the controlled car.
            lane: Current lane of the controlled car.
            min_dist: A list where min_dist[i] represents the minimum
                distance between the controlled car and another car
                in lane i.

        Returns:
            int: An action from the following set:

                ACTION_INCREASE_SPEED = 0
                ACTION_DECREASE_SPEED = 1
                ACTION_INCREASE_LANE = 2
                ACTION_DECREASE_LANE = 3
                ACTION_NO_OP = 4
        """
        state_idx = self.state_to_idx(speed, lane, min_dist)
        return self.pi[state_idx]