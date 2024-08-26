import numpy as np
import tensorflow as tf

from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.policies import epsilon_greedy_policy

# Hyperparameters
num_iterations = 20000
initial_collect_steps = 1000
collect_steps_per_iteration = 1
replay_buffer_max_length = 100000
batch_size = 64
learning_rate = 1e-3
log_interval = 200
epsilon = 0.1

# Environment
env_name = 'Blackjack-v1'
env = suite_gym.load(env_name)
tf_env = tf_py_environment.TFPyEnvironment(env)

# Preprocessing layer
preprocessing_layers = (
    tf.keras.layers.Flatten(),  # or any other preprocessing layer
    tf.keras.layers.Flatten(),  
    tf.keras.layers.Flatten()   
)
preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)
preprocessing_layer = tf.keras.layers.Dense(8, activation='relu')

# QNetwork
q_net = q_network.QNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    preprocessing_layers=preprocessing_layers,
    preprocessing_combiner=preprocessing_combiner,
    fc_layer_params=(100,)
)

# Agent
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
    tf_env.time_step_spec(),
    tf_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)

agent.initialize()

# Policies
eval_policy = agent.policy
collect_policy = agent.collect_policy

# Replay buffer
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_env.batch_size,
    max_length=replay_buffer_max_length)

# Data collection
def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    buffer.add_batch(traj)



def collect_data(env, policy, buffer, steps):
    for _ in range(steps):
        collect_step(env, policy, buffer)

collect_data(tf_env, epsilon_greedy_policy.EpsilonGreedyPolicy(policy=agent.policy,
                                                      epsilon=epsilon),
             replay_buffer, initial_collect_steps)

# Dataset
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, 
    sample_batch_size=batch_size, 
    num_steps=2).prefetch(3)

iterator = iter(dataset)

# Training loop
agent.train = common.function(agent.train)

agent.train_step_counter.assign(0)
overall_reward = 0
n_wins = 0
n_draws = 0
n_losses = 0
# Collect a few steps using collect_policy and save to the replay buffer.
for _ in range(num_iterations):
    # Collect a few steps using collect_policy and save to the replay buffer.
    for _ in range(collect_steps_per_iteration):
        collect_step(tf_env, agent.collect_policy, replay_buffer)

    # Sample a batch of data from the buffer and update the agent's network.
    experience, unused_info = next(iterator)
    train_loss = agent.train(experience).loss

    step = agent.train_step_counter.numpy()

    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss))


# Testing the agent
num_eval_episodes = 100 # Number of episodes to evaluate the agent
eval_interval = 1000  # Interval at which to eva2luate the agent

for episode in range(num_eval_episodes):
    time_step = tf_env.reset()
    episode_reward = 0

    print(f"\n====== EPISODE {episode} ======")

    while not time_step.is_last():
        
        action_step = agent.policy.action(time_step)
        time_step = tf_env.step(action_step.action)
        intention  = action_step.action.numpy()[0]

        player_pts = time_step.observation[0].numpy()[0]
        dealer_pts = time_step.observation[1].numpy()[0]
        reward     = time_step.reward.numpy()[0]
        if player_pts > 21:
            print("\033[41m\033[37m[DEALER]: YOU LOST\033[0m\n")
        else:
            print("[DEALER]: YOU HAVE", player_pts , "POINTS")
            print("[DEALER]: I HAVE", dealer_pts, "POINTS")
            print("[DEALER]: DO YOU WANT TO HIT OR STAND?")

            if intention == 1:
                action = "HIT"
            else:
                action = "STAND"

            print("[PLAYER]: I WANT TO", action)

            if reward == 1:
                print("\033[92mREWARD RECEIVED:", reward)
            elif reward == -1:
                print("\033[91mREWARD RECEIVED:", reward)
            else:
                print("\033[93mREWARD RECEIVED:", reward)

            print('\033[0m')
        
    if reward == 1:
        print("\033[42m\033[37m[DEALER]: YOU WON\033[0m\n")
        n_wins += 1
    elif reward == -1:
        print("\033[41m\033[37m[DEALER]: YOU LOST\033[0m\n")
        n_losses += 1
    else:
        print("\033[43m\033[37m[DEALER]: IT'S A DRAW\033[0m\n")
        n_draws += 1

    overall_reward += reward

print("Final reward:", overall_reward)
print("Average reward:", overall_reward / num_eval_episodes)
print("Number of Wins:", n_wins)
print("Number of Draws:", n_draws)
print("Number of Losses:", n_losses)