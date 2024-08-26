import math
import sys
import json
import asyncio
import websockets
import getpass
import os
import time
import random

from destroy_wall import dest
from clients import Client
from agentai import AgentAI
from destroy_enemies import Destroy_enemies

enemy_to_id={

    "Balloom":0,
    "Oneal":1,
    "Doll":2,
    "Minvo":3,
    "Kondoria":4,
    "Ovapi":5,
    "Pass":6
}

async def agent_loop(server_address="localhost:8000", agent_name="84730"):
    async with websockets.connect(f"ws://{server_address}/player") as websocket:
        msg, game_properties = await Client.recv_game_properties(websocket, agent_name)

        current_level = None

        # Powerups
        got_powerup = False
        bomb_range = 3
        bombs = 1
        detonator = False
        speed = 0
        powerup_last_pos = []

        # Bomberman
        lives = 3

        # Bombs
        bomb_drop_location = []
        can_drop_bomb = False

        # Dodge
        safe_position = []
        last_pos_with_bomb = []
        last_pos_with_bomb_counter = 0
        dodging_suicide = False

        # Enemy
        enemy_id = ''
        last_enemy_direction = [0, 0]
        last_enemy_position = []
        enemy_is_boring_counter = 0

        # Pathing
        pathing = []
        last_pos = []

        game_map = AgentAI.create_empty_2d_array(game_properties["size"][0], game_properties["size"][1])
        AgentAI.array_to_empty_map(game_map)
        
        

        # Create an instance of the RLAgent class


        while True:

            state = await Client.recv_game_state(websocket)

            # Stop the cycle if loose all lives or if stp receive server messages.
            if not state or lives == 0:
                break
                
            # Fill the map with walls and enemies.
            try:
                AgentAI.clear_map(game_map)
                AgentAI.fill_empty_map_with_walls(game_map, state["walls"])
                AgentAI.fill_empty_map_with_enemies(game_map, state["enemies"])
            except:
                break

            if current_level is None:
                current_level = state["level"]
            elif current_level < state["level"]:
                current_level = state["level"]
                got_powerup = False
                bomb_drop_location = []
                can_drop_bomb = False
                safe_position = []
                enemy_id = ''
                last_enemy_direction = [0, 0]
                last_enemy_position = []
                pathing = []
                last_pos = []
                powerup_last_pos = []
                enemy_is_boring_counter = 0
                last_pos_with_bomb = []
                last_pos_with_bomb_counter = 0
                dodging_suicide = False

            if state["level"] in [2, 3, 5, 6, 8, 11, 12, 13]:
                got_powerup = True
                if state["powerups"]:
                    game_map[state["powerups"][0][0][1]][state["powerups"][0][0][0]] = 4

            if not state["enemies"] and not got_powerup and state["exit"]:
                game_map[state["exit"][1]][state["exit"][0]] = 4

            # Update internal state if lose 1 live.
            if state["lives"] < lives:

                enemy_id = ''
                last_enemy_position = []
                last_enemy_direction = [0, 0]
                bomb_drop_location = []
                target = []
                can_drop_bomb = False
                safe_position = []
                pathing = []
                target_enemy = False
                last_pos_with_bomb = []
                last_pos_with_bomb_counter = 0
                dodging_suicide = False

            # Update variable lives
            lives = state["lives"]

            # Update internal state if the state of bombs change.
            if state['bombs']:
                exist_bomb = True

            else:
                exist_bomb = False
                safe_position = []

            closest_destructive_wall = Destroy_enemies.closest_wall(state["bomberman"], state["walls"])

            if state["bomberman"] == last_pos_with_bomb and exist_bomb:
                last_pos_with_bomb_counter += 1

            if last_pos_with_bomb_counter >= 2 and exist_bomb:
                target = AgentAI.get_safe_spot(state["bomberman"], state["bombs"][0],
                                                          state["bomberman"], bomb_range - last_pos_with_bomb_counter,
                                                          game_map,
                                                          safes=[])

                can_drop_bomb = False
                pathing = AgentAI.a_star_pathing(state["bomberman"], target, game_map, False)
                dodging_suicide = True

            if not dodging_suicide:
                if powerup_last_pos == state["bomberman"]:
                    got_powerup = True
                    if powerup == "Flames":
                        bomb_range += 1
                    elif powerup == "Bombs":
                        bombs += 1
                    elif powerup == "Detonator":
                        detonator = True
                    elif powerup == "Speed":
                        speed += 1

                # Catch powerup.
                if state["powerups"] and not got_powerup:
                    powerup = state["powerups"][0][1]
                    powerup_last_pos = state["powerups"][0][0]
                    target = powerup_last_pos

                elif len(state["enemies"]) > 0 and not exist_bomb:

                    dist = [dest.distance_points(state['bomberman'], enemy["pos"]) for enemy in state["enemies"]]
                    index = dest.close_position(dist)
                    target_enemy = state["enemies"][index]["pos"]

                    if enemy_id == state["enemies"][index]["id"]:
                        enemy_is_boring_counter += 1
                    else:
                        enemy_is_boring_counter = 0

                    if enemy_id != state["enemies"][index]["id"]:
                        last_enemy_position = []
                        last_enemy_direction = [0, 0]

                    if last_enemy_position:

                        direction = [target_enemy[0] - last_enemy_position[0],
                                     target_enemy[1] - last_enemy_position[1]]

                        if direction == [0, 0] or -1 <= direction[0] + direction[1] >= 1:  # remendo
                            if -1 <= last_enemy_direction[0] + last_enemy_direction[1] >= 1:
                                direction = [0, 0]
                            else:
                                direction = last_enemy_direction

                        bomb_drop_location, last_enemy_direction = Destroy_enemies.place_drop_bomb(target_enemy, game_map,
                                                                                                   direction, tmp=random.randint(2,3))

                    else:

                        bomb_drop_location, last_enemy_direction = Destroy_enemies.place_drop_bomb(target_enemy, game_map,
                                                                                                   [0, 0], tmp=random.randint(2,3))

                    if game_map[bomb_drop_location[1]][bomb_drop_location[0]] == 3:
                        if game_map[bomb_drop_location[1] + 1][bomb_drop_location[0]] == 2:
                            bomb_drop_location = [bomb_drop_location[0], bomb_drop_location[1] + 1]
                        elif game_map[bomb_drop_location[1]][bomb_drop_location[0] + 1] == 2:
                            bomb_drop_location = [bomb_drop_location[0] + 1, bomb_drop_location[1]]
                        elif game_map[bomb_drop_location[1] - 1][bomb_drop_location[0]] == 2:
                            bomb_drop_location = [bomb_drop_location[0], bomb_drop_location[1] - 1]
                        else:
                            bomb_drop_location = [bomb_drop_location[0] - 1, bomb_drop_location[1]]
                        can_drop_bomb = True

                    enemy_id = state["enemies"][index]["id"]
                    target = bomb_drop_location
                    last_enemy_position = target_enemy

                elif state["exit"] and not state["enemies"] and not exist_bomb and got_powerup:
                    target = state["exit"]

                    # Destroy walls at distance of 1 from bomberman.
                elif Destroy_enemies.enemy_is_in_range(state["bomberman"], closest_destructive_wall,
                                                       1) and not exist_bomb:

                    tmp_safe_position = AgentAI.get_safe_spot(state["bomberman"], state["bomberman"],
                                                              closest_destructive_wall, bomb_range,
                                                              game_map,
                                                              safes=[])

                    if tmp_safe_position:
                        safe_position = tmp_safe_position
                        can_drop_bomb = True
                    else:
                        can_drop_bomb = False

                elif (not state['exit'] and not exist_bomb) or (
                        not got_powerup and not exist_bomb and not state["enemies"])\
                        or powerup_last_pos is not None\
                        or got_powerup and not state["exit"]:
                    target = closest_destructive_wall

                if enemy_is_boring_counter > 50:
                    target = closest_destructive_wall
                    enemy_is_boring_counter = 0

                if state["bomberman"] != target and not state['bombs']:
                    if len(pathing) == 0:
                        pathing = AgentAI.a_star_pathing(state["bomberman"], target, game_map, True)

                    if game_map[pathing[0][1]][pathing[0][0]] == 2:
                        tmp_safe_position = AgentAI.get_safe_spot(state["bomberman"], state["bomberman"],
                                                                  pathing[0], bomb_range,
                                                                  game_map,
                                                                  safes=[])

                        if tmp_safe_position:
                            safe_position = tmp_safe_position
                            can_drop_bomb = True
                        else:
                            can_drop_bomb = False

                if state['bombs'] and safe_position and state["bomberman"] != safe_position:
                    pathing = AgentAI.a_star_pathing(state["bomberman"], safe_position, game_map, False)

                if last_pos == state["bomberman"] and detonator and state["bombs"]:
                    key = "A"

                last_pos = state["bomberman"]

            dodging_suicide = False
            for e in state["enemies"]:
                
                e["tag"]=enemy_to_id[e["name"]]
            print(state)
                
            
            if len(pathing) == 0:
                if state["bomberman"] == bomb_drop_location and not state["bombs"]:
                    tmp_safe_position = AgentAI.get_safe_spot(state["bomberman"], bomb_drop_location, target_enemy,
                                                              bomb_range, game_map,
                                                              safes=[])

                    if tmp_safe_position:
                        safe_position = tmp_safe_position
                        key = "B"
                        last_enemy_position = []
                        last_enemy_direction = [0, 0]
                    else:
                        key = ""

                elif detonator and (
                        state["bomberman"][0] != state["bombs"][0][0] or state["bomberman"][1] != state["bombs"][0][1]):
                    key = "A"

                else:
                    key = ""

                pathing = []

                await websocket.send(
                    json.dumps({"cmd": "key", "key": key})
                )

            elif len(pathing) > 0:
                next_pos = pathing[0]
                if can_drop_bomb:
                    key = "B"
                    can_drop_bomb = False
                    breaking_walls = True
                elif state["bomberman"][1] - 1 == next_pos[1]:
                    key = "w"
                elif state["bomberman"][1] + 1 == next_pos[1]:
                    key = "s"
                elif state["bomberman"][0] - 1 == next_pos[0]:
                    key = "a"
                elif state["bomberman"][0] + 1 == next_pos[0]:
                    key = "d"
                else:
                    key = ""

                moving_adjacent_enemy = False
                if game_map[next_pos[1] - 1][next_pos[0]] == 3:
                    moving_adjacent_enemy = True
                if game_map[next_pos[1] + 1][next_pos[0]] == 3:
                    moving_adjacent_enemy = True
                if game_map[next_pos[1]][next_pos[0] - 1] == 3:
                    moving_adjacent_enemy = True
                if game_map[next_pos[1]][next_pos[0] + 1] == 3:
                    moving_adjacent_enemy = True

                if moving_adjacent_enemy and not state["bombs"]:
                    key = ''

                pathing = []

                await websocket.send(
                    json.dumps({"cmd": "key", "key": key})
                )
                await websocket.send(
                    json.dumps({"cmd": "key", "key": key}))

                #agent.save_model("trained_model.pth")

# Not to touch...
loop = asyncio.get_event_loop()
SERVER = os.environ.get("SERVER", "localhost")
PORT = os.environ.get("PORT", "8000")
NAME = os.environ.get("NAME", getpass.getuser())
loop.run_until_complete(agent_loop(f"{SERVER}:{PORT}", NAME))
