
import tensorflow as tf


class StateTuple:
    def get_state_tuple( state):
        # Unpack the state_tuple
        if not isinstance(state, dict):
            print("VOU MORRER")
            return state.observation

        else:
            step = state["step"]
            bomberman = state["bomberman"]
            exit = state["exit"]
            powerups = state["powerups"]
            enemies = [enemy['pos'] + [enemy['tag']] for enemy in state['enemies']]
            walls = state["walls"]
            bomb_range = state["bomb_range"]
            detonator = state["detonator"]
            lives = state["lives"]
            timeout = state["timeout"]
            bombs =state["bombs"]

            # Add phantom elements for missing elements
            max_powerups = 10
            max_enemies = 9
            max_walls = 800
            max_bombs=5

            powerups += [[-1, -1]] * (max_powerups - len(powerups))
            enemies += [[-1, -1, -1]] * (max_enemies - len(enemies))
            walls += [[-1, -1]] * (max_walls - len(walls))
            bombs += [[-1, -1,-1,-1]] * (max_bombs - len(bombs))
            exit = exit if exit else [-1, -1]

            # Convert individual components to match observation spec
            step_observation = tf.convert_to_tensor(step, dtype=tf.int32)
            bomberman_observation = tf.convert_to_tensor(bomberman, dtype=tf.int32)
            exit_observation = tf.convert_to_tensor(exit, dtype=tf.int32)
            powerups_observation = tf.convert_to_tensor(powerups, dtype=tf.int32)
            enemies_observation = tf.convert_to_tensor(enemies, dtype=tf.int32)
            walls_observation = tf.convert_to_tensor(walls, dtype=tf.int32)
            bomb_range_observation = tf.convert_to_tensor(bomb_range, dtype=tf.int32)
            detonator_observation = tf.convert_to_tensor(detonator, dtype=tf.int32)
            lives_observation = tf.convert_to_tensor(lives, dtype=tf.int32)
            timeout_observation = tf.convert_to_tensor(timeout, dtype=tf.int32)
            bombs_observation =tf.convert_to_tensor(bombs, dtype=tf.int32)

            # Create the observation tuple
            observation = [
                step_observation,
                bomberman_observation,
                exit_observation,
                powerups_observation,
                enemies_observation,
                walls_observation,
                bomb_range_observation,
                detonator_observation,
                lives_observation,
                timeout_observation,
                bombs_observation
            ]
            return observation
