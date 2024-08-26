import numpy as np
import tensorflow as tf
from tf_agents.trajectories import time_step as ts
import tf_agents as tfa
from tf_agents.environments import tf_py_environment
from tf_agents.environments import py_environment
from tf_agents import specs
import math
from termcolor import colored
from copy import deepcopy
#from tf_agents.specs.array_spec import array_spec
 

SURVIVAL_WEIGHT = 1
TIMEOUT_WEIGHT = 1
EXIT_WEIGHT = 100
POWERUP_WEIGHT = 10
ENEMY_WEIGHT = 10         
NEGATIVE_ACTION_WEIGHT = 200
max_powerups = 10
max_enemies = 9
max_walls =800
max_bombs=5
class Bomberman_Environment(py_environment.PyEnvironment):
    def __init__(self, state):
        
        self.detonation_step =None
        self._state = state
        self.inital_state = deepcopy(state)
        self._lives =self._state[8]
        self._original_enemy_positions=list(self._state[4].numpy())

        self._action_spec = specs.array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=5, name='action'
        )
        
        # Define the shape and data types for fixed-size elements
        self._observation_spec = [
            specs.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=4000, name='step'),  # step
            specs.BoundedArraySpec(shape=(2,), dtype=np.int32, minimum=0, maximum=3000, name='bomberman'),  # bomberman
            specs.BoundedArraySpec(shape=(2,), dtype=np.int32, minimum=0, maximum=3000, name='exit'),  # exit
            specs.BoundedArraySpec(shape=(max_powerups, 2), dtype=np.int32, minimum=0, maximum=3000, name='powerups'),  # powerups
            specs.BoundedArraySpec(shape=(max_enemies, 3), dtype=np.int32, minimum=0, maximum=3000, name='enemies'),  # enemies
            specs.BoundedArraySpec(shape=(max_walls, 2), dtype=np.int32, minimum=0, maximum=3000, name='walls'),  # walls
            specs.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=100, name='bomb_range'),  # bomb_range
            specs.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=1, name='detonator'),  # detonator
            specs.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=3, name='lives'),  # lives
            specs.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=3000, name='timeout'),  # timeout
            specs.BoundedArraySpec(shape=(max_bombs,4), dtype=np.int32, minimum=0, maximum=3000, name='bombs')  # bombs
        ]

        self._episode_ended = False
        print(colored("[I] Environment initialized", "green"))

    def observation_spec(self):
        print(colored("[A] Observation spec","blue"))
        return self._observation_spec

    def action_spec(self):
        print(colored("[A] Action spec","blue"))
        return self._action_spec

    def _reset(self):
        print(colored("[I] Resetting environment","green"))
        self._episode_ended = False
        # Reset bomb range and detonator based on initial state
        initial_bomb_range = 3  # Retrieve the initial bomb range from the state
        initial_detonator = False  # Retrieve the initial detonator value from the state
        self.enemy_count = len(self._state[4])
        self._state[6] = initial_bomb_range
        self._state[7] = initial_detonator
        

        # create a list of tensors for observation
        observation = [
            tf.constant(self.inital_state[0], dtype=tf.int32),
            tf.constant(self.inital_state[1], dtype=tf.int32),
            tf.constant(self.inital_state[2], dtype=tf.int32),
            tf.constant(self.inital_state[3], dtype=tf.int32),
            tf.constant(self.inital_state[4], dtype=tf.int32),
            tf.constant(self.inital_state[5], dtype=tf.int32),
            tf.constant(self.inital_state[6], dtype=tf.int32),
            tf.constant(self.inital_state[7], dtype=tf.int32),
            tf.constant(self.inital_state[8], dtype=tf.int32),
            tf.constant(self.inital_state[9], dtype=tf.int32),
            tf.constant(self.inital_state[10], dtype=tf.int32)
        ]
        print("Reseted state: ", self.inital_state)
        print(colored("[F] Finished resetting environment","green"))
        return ts.restart(observation)

    
    def _step(self, action):
        print(colored("[A] Stepping through environment","blue"))
        action=action.item()
        if self._episode_ended:
            return self.reset()



        # Take action and update the game state
        # TODO: Update the game state based on the action
        step=self._state[0].numpy()+1
        bomberman=list(self._state[1].numpy())
        exit=list(self._state[2].numpy())
        powerups=list(self._state[3].numpy())
        enemies=list(self._state[4].numpy())
        walls=list(self._state[5].numpy())
        bomb_range=self._state[6]
        detonator=self._state[7]
        lives=self._state[8]
        timeout=self._state[9]
        bombs=list(self._state[10].numpy())
        
        consequence=None

        #DONE ACHO EU 
        
        print("Bomberman position: ", bomberman)
        

        if action == 0 and bomberman[0]%2 and bomberman[1]-1>1: 
            print("Bomberman moved up")
            # For "w" input
            # Update next_state accordingly
            bomberman[1]-=1
            
        elif action == 1 and  bomberman[0]%2 and bomberman[1]-1<29:
            print("Bomberman moved down")  
            # For "s" input
            # Update next_state accordingly
            bomberman[1]+=1
        #on moving horizontaly even rows will always have a wall to each side
        elif action == 2 and  bomberman[1]%2 and bomberman[1]-1>1:
            print("Bomberman moved left")
            # For "a" input
            # Update next_state accordingly
            bomberman[0]-=1
            
        elif action == 3 and  bomberman[1]%2 and bomberman[1]-1<49:
            # For "d" input
            # Update next_state accordingly
            bomberman[0]+=1
            print("Bomberman moved right")
            
        elif action == 4:
            # For "B" input
            # Update next_state accordingly
            #according to the game.py file bomb timer is given by radius+1


            #bombs.append([bomberman[0],bomberman[1],bomb_range+1,bomb_range])
            for i in range(len(bombs)):
                if bombs[i][2]==-1:
                
                    bombs[i] = np.array([bomberman[0], bomberman[1], 4, 3], dtype=np.int32)
                    break
            print("--------------------------Bomberman dropped a bomb--------------------------")

            self.detonation_step=step+3+1
            
        elif action == 5 and detonator:
            print("--------------------------Bomberman detonated a bomb--------------------------")
            # For "A" input
            # Update next_state accordingly
            #removes the oldest bomb as it is exploded
            bombs.pop(0)
        else:
            #Empty action
            print("-------------------------------------DEBUG-------------------------------------")
            if action == 0 or action == 1 or action == 2 or action == 3:
                consequence=0
                print("Bomberman hit a wall")
            elif action == 4:
                consequence=1

                print("Bomberman tried to drop a bomb but failed")
            elif action == 5:
                print("Bomberman tried to detonate a bomb but failed (no detonator)")
                consequence=2
            elif any(bomberman[0] == enemy[0]  and bomberman[1] == enemy[1] for enemy in enemies):
                print("Bomberman was killed by an enemy")
                consequence=3
            else:
                print("Bomberman did nothing (possible problem with action generation")

        #Time to detonate        
        if self.detonation_step ==step:
            bombs.pop(0)
            self.detonation_step=None
        if lives<self._lives:
            self._reset_on_life_lost()
            self._lives = lives  # Update the number of lives   
        # Check if the episode has ended
        if bomberman == exit:
            self._episode_ended = True
        elif lives == 0:
            self._episode_ended = True
        elif step == timeout:
            self._episode_ended = True
        self._episode_ended = False  # Set it to True if the episode has ended

        # Calculate the reward based on the game state
        reward = self.calculate_reward(self._state,consequence)
        # return a list of tensors for observation instead of dictionary
        observation = [
            tf.constant(step, dtype=tf.int32),
            tf.constant(bomberman, dtype=tf.int32),
            tf.constant(exit, dtype=tf.int32),
            tf.constant(powerups, dtype=tf.int32),
            tf.constant(enemies, dtype=tf.int32),
            tf.constant(walls, dtype=tf.int32),
            tf.constant(bomb_range, dtype=tf.int32),
            tf.constant(detonator, dtype=tf.int32),
            tf.constant(lives, dtype=tf.int32),
            tf.constant(timeout, dtype=tf.int32),
            tf.constant(bombs,dtype=tf.int32)
        ]
        self._state = observation
        print("Reward has been calculated"+str(reward))
        print(colored("[F] Finished stepping through environment","green"))
        return ts.transition(observation, reward=tf.squeeze(reward), discount=tf.squeeze(tf.constant([1.0])))

    def calculate_reward(self,state,consequence):
        survival_reward = self.calculate_survival_reward(state)
        goal_reward = self.calculate_goal_reward(state)
        negative_action_reward = self.calculate_negative_action_reward(consequence)
        powerup_reward = self.calculate_powerup_reward(state)
        time_reward = self.calculate_time_reward(state)
        enemy_reward= self.calculate_enemy_reward(state)

        overall_reward = (
            SURVIVAL_WEIGHT * survival_reward +
            EXIT_WEIGHT * goal_reward +
            NEGATIVE_ACTION_WEIGHT * negative_action_reward +
            POWERUP_WEIGHT * powerup_reward +
            TIMEOUT_WEIGHT * time_reward +
            ENEMY_WEIGHT * enemy_reward
        )
        return tf.constant([overall_reward.numpy()],dtype=tf.float32)

    def calculate_survival_reward(self,state):

        if state[8]!=0:
            # The Bomberman is alive, assign a positive reward
            survival_reward = 1.0
        else:
            # The Bomberman is dead, assign a negative reward
            survival_reward = -1.0

        return survival_reward

    def calculate_goal_reward(self,state):
        if list(state[1].numpy())==list(state[2].numpy()):
            # The goal is achieved, assign a positive reward
            goal_reward = 100.0
        else:
            # The goal is not yet achieved
            goal_reward = 0.0

        return goal_reward


    def calculate_negative_action_reward(self,consequence):
        if consequence == 0:
            # The Bomberman walked into a wall, assign a negative reward
            negative_action_reward = -0.5
        elif consequence == 1 or consequence == 2:
            # The Bomberman failed to deploy a bomb
            negative_action_reward = -0.2
        elif consequence == 3:
            # The Bomberman died
            negative_action_reward = -5.0
        else:
            # No negative action occurred
            negative_action_reward = 0.0

        return negative_action_reward

    def calculate_time_reward(self,state):
        # Calculate the time-based reward
        time_ratio = (state[9] - state[0].numpy()) / state[9]
        time_reward = time_ratio * -0.1  # Decreasing reward as time progresses

        return time_reward
    def calculate_powerup_reward(self,state):
        

        if any(list(state[1].numpy())==list(p) for p in state[3].numpy()):
            # A power-up was obtained, assign a positive reward
            powerup_reward = 5.0
        else:
            # No power-up obtained
            powerup_reward = 0.0

        return powerup_reward

    def calculate_enemy_reward(self,state):
        
        enemy_reward_list={
        0:1,
        1:2,
        2:4,
        3:8,
        4:16,
        5:32,
        6:64


        }
        if self.enemy_count != len(list(state[4].numpy())):
            for enemy in list(state[4].numpy()):
               enemy_reward = enemy_reward_list[enemy[2]]
        else:
            enemy_reward = 0
        self.enemy_count = len(list(state[4].numpy()))

        return enemy_reward 
    def _reset_on_life_lost(self):
        """Reset specific elements of the environment when the agent loses a life."""
        # Reset bomberman's position
        print(colored("[I] Resetting on life lost","green"))
        self._state[1] = tf.constant([1, 1], dtype=tf.int32)

        # Teleport enemies away that are closer than a certain distance
        for i, enemy in enumerate(self._state[4]):
            if self.distance(self._state[1], enemy) < 3:
                # Assuming self._original_enemy_positions contains the original positions of the enemies
                self._state[4][i] = tf.constant(self._original_enemy_positions[i], dtype=tf.int32)

        # Add any other reset rules here
        print(colored("[F] Finished resetting on life lost","green"))

    def distance(self, p1, p2):
        """Calculate the Euclidean distance between two points."""
        x1, y1 = p1
        x2, y2 = p2
        return math.hypot(x1 - x2, y1 - y2)

