import numpy as np
import tf_agents as tfa
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.specs import tensor_spec
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.policies import random_tf_policy
from tf_agents.trajectories.time_step import TimeStep
from state_tuple import StateTuple
from termcolor import colored
from tf_agents.trajectories import Trajectory
from tf_agents.specs import TensorSpec, BoundedTensorSpec
from tf_agents.trajectories.trajectory import to_transition,Transition
from tf_agents.trajectories.time_step import TimeStep,StepType
from collections import namedtuple
from tf_agents.policies import q_policy
from Bomberman_Environment import Bomberman_Environment as Bomber_env

class RLAgent:
    def __init__(self, init_state):

        # Convert the custom environment to TensorFlow environment
        env = tf_py_environment.TFPyEnvironment(Bomber_env(init_state))
        self.env=env
        observation_spec = env.observation_spec()
        self.next_reward = 0.0
        # Define the QNetwork architecture
        num_neurons = 100
        fc_layer_params = (100,)
        # Define the preprocessing layers
        preprocessing_layers = [tf.keras.layers.Flatten() for spec in observation_spec]

        # Create the encoding network

        preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)

        # Create the QNetwork
        q_net = q_network.QNetwork(
            input_tensor_spec=observation_spec,
            action_spec=env.action_spec(),
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            fc_layer_params=fc_layer_params
        )



        # Define the optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        # Define the agent
        self.agent = tfa.agents.dqn.dqn_agent.DqnAgent(
            time_step_spec=env.time_step_spec(),
            action_spec=env.action_spec(),
            q_network=q_net,
            optimizer=optimizer,
            
            )


        # Define the policy
        self.policy = self.agent.policy
        print(self.policy)
        # Set up the replay buffer
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=self.agent.collect_data_spec,
        batch_size=self.env.batch_size,
        max_length=100000)


        # Initialize the agent
        self.agent.initialize()
        print(colored("[I] Agent initialized", "green"))

    def collect_step(self):
        time_step = self.env.current_time_step()
        action_step = self.policy.action(time_step)
        next_time_step = self.env.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        self.replay_buffer.add_batch(traj)
        print(colored("[A] Trajectory added to replay buffer","blue"))

   
            
    def get_time_step(self, state_tuple):
        observation = [
            tf.expand_dims(state_tuple[0], axis=0),  # step
            tf.expand_dims(state_tuple[1], axis=0),  # bomberman
            tf.expand_dims(state_tuple[2], axis=0),  # exit
            tf.expand_dims(state_tuple[3], axis=0),  # powerups
            tf.expand_dims(state_tuple[4], axis=0),  # enemies
            tf.expand_dims(state_tuple[5], axis=0),  # walls
            tf.expand_dims(state_tuple[6], axis=0),  # bomb_range
            tf.expand_dims(state_tuple[7], axis=0),  # detonator
            tf.expand_dims(state_tuple[8], axis=0),  # lives
            tf.expand_dims(state_tuple[9], axis=0),   # timeout
            tf.expand_dims(state_tuple[10], axis=0)  # bombs
        ]
        
        discount = tf.constant([1.0])
        step_type = tf.constant([0])

        time_step = TimeStep(
            step_type=step_type,
            reward=self.next_reward,
            discount=discount,
            observation=observation
        )

        return time_step

    def get_next_time_step(self, action):
        #action_tensor = tf.constant([action], dtype=tf.int32)
        #print(action_tensor)
        # Call the environment's _step function
        next_time_step = self.env.step(action)
        print(colored("[A] Next state has been created", "blue"))
        return next_time_step

    def helper(self, state):

        """This function is used to collect trajectories from the environment and add them to the replay buffer as well as acting like
        a act function for the agent."""
        
        time_step = self.get_time_step(state)
        action_step = self.policy.action(time_step)
        next_time_step = self.get_next_time_step(action_step.action)
        
        print("------------------I am current time_step------------------")
        print(time_step)
        print("------------------I am next_step------------------")
        print(next_time_step)
        print("------------------I am action_step------------------")
        print(action_step)
        
        #With the following code, we can store the time_step in the replay buffer
        self.next_reward = next_time_step.reward.numpy()[0] 

        
        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        self.replay_buffer.add_batch(traj)
        time_step = next_time_step
        
        print(colored("[A] Added to replay buffer", "blue"))
        return action_step.action.numpy()[0]

        
    def train(self, num_iterations=1000, log_interval=200):
        """
        num_iterations : int
            number of batch iterations
        log_interval : int
            how many batches before logging training status
        """
        dataset = self.replay_buffer.as_dataset(
        num_parallel_calls=3, 
        sample_batch_size=64, 
        num_steps=2).prefetch(3)

        iterator = iter(dataset)

        total_loss = 0.0

        for i in range(num_iterations):
            # Sample a batch of data from buffer and update the agent's network.
            experience, _ = next(iterator)
            train_loss = self.agent.train(experience)

            total_loss += train_loss.loss

            if i % log_interval == 0:
                print(f"step = {i}: loss = {train_loss.loss}")

        return total_loss / num_iterations  # Return average loss over training.


    def evaluate(self, num_episodes=10):
        """
        num_episodes : int
            number of episodes to run agent for evaluation
        """

        total_return = 0.0  # Total return over num_episodes
        for _ in range(num_episodes):

            time_step = self.env.reset()
            episode_return = 0.0  # Return for a single episode

            while not time_step.is_last():
                action_step = self.policy.action(time_step)
                time_step = self.env.step(action_step.action)
                episode_return += time_step.reward

            total_return += episode_return

        avg_return = total_return / num_episodes
        print(f"Average return over {num_episodes} episodes: {avg_return}")

        return avg_return
