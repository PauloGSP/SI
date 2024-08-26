import tensorflow as tf
from tf_agents.environments import suite_gym
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.drivers import dynamic_step_driver, dynamic_episode_driver
from tf_agents.policies import random_tf_policy

""" 
Este é um exemplo de treinamento de um agente com TensorFlow Agents no ambiente CartPole-v1 do OpenAI Gym. 
O agente usa a rede neural Q para estimar valores de Q e é treinado com a estratégia DQN. O loop de treinamento é semelhante ao exemplo anterior, 
mas com ajustes para o ambiente do CartPole-v1.

Espero que isso ajude a entender como o TensorFlow Agents pode ser usado para treinar agentes de aprendizado por reforço em diferentes ambientes!
 """

# Define a função para criar o ambiente
def create_environment():
    return suite_gym.load('CartPole-v1')

# Cria o ambiente
env = create_environment()

# Define o número de episódios a serem executados durante o treinamento
num_iterations = 20000

# Define o tamanho do lote para atualizações de modelo
batch_size = 64

# Define a taxa de aprendizado
learning_rate = 1e-3

# Define a taxa de desconto para recompensas futuras
gamma = 0.99

# Define a rede neural que o agente usará para estimar valores de Q
q_net = q_network.QNetwork(
    env.observation_spec(),
    env.action_spec(),
    fc_layer_params=(100,)
)

# Define o agente
agent = dqn_agent.DqnAgent(
    env.time_step_spec(),
    env.action_spec(),
    q_network=q_net,
    optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate),
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=tf.Variable(0)
)

# Define o coletor de dados
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=env.batch_size,
    max_length=100000
)
collect_policy = agent.collect_policy
collect_driver = dynamic_step_driver.DynamicStepDriver(
    env,
    collect_policy,
    observers=[replay_buffer.add_batch],
    num_steps=1
)

# Define a política de avaliação
eval_policy = agent.policy
eval_driver = dynamic_episode_driver.DynamicEpisodeDriver(
    env,
    eval_policy,
    observers=[replay_buffer.add_batch]
)

# Define a estratégia de coleta
random_policy = random_tf_policy.RandomTFPolicy(env.time_step_spec(), env.action_spec())
initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
    env,
    random_policy,
    observers=[replay_buffer.add_batch],
    num_steps=100
)

# Cria o loop de treinamento
dataset = replay_buffer.as_dataset(num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).prefetch(3)
iterator = iter(dataset)
agent.train = common.function(agent.train)
agent.train_step_counter.assign(0)
avg_return = compute_avg_return(env, eval_policy, num_eval_episodes=10)
returns = [avg_return]
for _ in range(num_iterations):
    collect_driver.run()
    experience, unused_info = next(iterator)
    train_loss = agent.train(experience)
    replay_buffer.clear()
    
    if agent.train_step_counter.numpy() % 200 == 0:
        avg_return = compute_avg_return(env, eval_policy, num_eval_episodes=10)
        print('step = {0}: Average Return = {1}'.format(agent.train_step_counter.numpy(), avg_return))
        returns.append(avg_return)
        
# Define a função para calcular o retorno médio
def compute_avg_return(environment, policy, num_eval_episodes=10):
    total_return = 0.0
    for _ in range(num_eval_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return
    avg_return = total_return / num_eval_episodes
    return avg_return