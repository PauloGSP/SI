import math
import random

from searchnode import SearchNode


class AgentAI:

    # Function to get the closest enemy (diagonal distance, not closest path-wise)
    @staticmethod
    def closest_enemy(bomber_coords, enemies):
        if not enemies:
            return None
        elif len(enemies) == 1:
            return enemies[0]["pos"]
        elif len(enemies) == 2:
            if math.hypot(enemies[0]["pos"][0] - bomber_coords[0], enemies[0]["pos"][1] - bomber_coords[1]) \
                    < math.hypot(enemies[1]["pos"][0] - bomber_coords[0], enemies[1]["pos"][1] - bomber_coords[1]):
                return enemies[0]["pos"]
            else:
                return enemies[1]["pos"]
        else:
            if math.hypot(enemies[0]["pos"][0] - bomber_coords[0], enemies[0]["pos"][1] - bomber_coords[1]) \
                    < math.hypot(enemies[1]["pos"][0] - bomber_coords[0], enemies[1]["pos"][1] - bomber_coords[1]):
                return AgentAI.closest_enemy(bomber_coords, [enemies[0]] + enemies[2:])
            else:
                return AgentAI.closest_enemy(bomber_coords, [enemies[1]] + enemies[2:])

    # Function that generates a recursive cost map - being discarded
    @staticmethod
    def generate_cost_map(bomber_pos, bomb_radius, walls, map_width, map_height):
        cost_map = AgentAI.create_empty_2d_array(map_width, map_height)
        current_map = AgentAI.create_base_map(walls, map_width, map_height)
        cost_map = AgentAI.array_to_empty_cost_array(cost_map, map_width, map_height)

        AgentAI.fill_cost_map_rec(cost_map, current_map, bomb_radius, bomber_pos, move_cost=0)
        return cost_map

    # Function that creates an empty 2d array
    @staticmethod
    def create_empty_2d_array(width, height):
        cost_map = []

        for i in range(0, height):
            tmp = []
            for j in range(0, width):
                tmp.append(None)
            cost_map.append(tmp)

        return cost_map

    # Function that creates an empty game map
    @staticmethod
    def array_to_empty_map(in_map):
        for i in range(0, len(in_map)):
            for j in range(0, len(in_map[0])):
                if i == 0 or i == len(in_map) - 1 or j == 0 or j == len(in_map[0]) - 1:
                    in_map[i][j] = 1
                elif i % 2 == 0 and j % 2 == 0:
                    in_map[i][j] = 1
                else:
                    in_map[i][j] = 0

    # Function that creates an empty cost array - being discarded
    @staticmethod
    def array_to_empty_cost_array(in_array, map_width, map_height):
        for i in range(0, len(in_array)):
            for j in range(0, len(in_array[0])):
                if i == 0 or i == len(in_array) - 1 or j == 0 or j == len(in_array[0]) - 1:
                    in_array[i][j] = (map_width + map_height - 2) * 23 + 1
                elif i % 2 == 0 and j % 2 == 0:
                    in_array[i][j] = (map_width + map_height - 2) * 23 + 1
                else:
                    in_array[i][j] = 0

    # Function that fills a game map array with the walls, given their coordinates
    @staticmethod
    def fill_empty_map_with_walls(in_map, walls):
        for wall in walls:
            in_map[wall[1]][wall[0]] = 2

    # Function that fills a game map array with the enemies, given their coordinates
    @staticmethod
    def fill_empty_map_with_enemies(in_map, enemies):
        for enemy in enemies:
            in_map[enemy["pos"][1]][enemy["pos"][0]] = 3

    # Function that cleans a map to update walls and enemies
    @staticmethod
    def clear_map(in_map):
        for i in range(0, len(in_map)):
            for j in range(0, len(in_map[i])):
                if in_map[i][j] == 2 or in_map[i][j] == 3 or in_map[i][j] == 4:
                    in_map[i][j] = 0

    # Function that creates a game map representation
    @staticmethod
    def create_base_map(walls, enemies, map_width, map_height):
        tmp_map = AgentAI.create_empty_2d_array(map_width, map_height)
        AgentAI.array_to_empty_map(tmp_map)
        AgentAI.fill_empty_map_with_walls(tmp_map, walls)
        AgentAI.fill_empty_map_with_enemies(tmp_map, enemies)
        return tmp_map

    # First algorithmic attempt at creating a cost map
    # Optimal yet REALLY inefficient - being discarded (was unfinished)
    @staticmethod
    def fill_cost_map_rec(cost_map, game_map, bomb_radius, bomber_pos, move_cost):
        cost_map[bomber_pos[0]][bomber_pos[1]] = move_cost

        if game_map[bomber_pos[0] - 1][bomber_pos[1]] != 1:
            if cost_map[bomber_pos[0] - 1][bomber_pos[1]] is None \
                    or cost_map[bomber_pos[0] - 1][bomber_pos[1]] > move_cost:
                if game_map[bomber_pos[0] - 1][bomber_pos[1]] == 2:
                    AgentAI.fill_cost_map_rec(cost_map, game_map, bomb_radius, [bomber_pos[0] - 1, bomber_pos[1]],
                                              move_cost + 2 * bomb_radius + 11)
                elif game_map[bomber_pos[0] - 1][bomber_pos[1]] == 0:
                    AgentAI.fill_cost_map_rec(cost_map, game_map, bomb_radius, [bomber_pos[0] - 1, bomber_pos[1]],
                                              move_cost + 1)

        if bomber_pos[0] + 1 < len(game_map) and game_map[bomber_pos[0] + 1][bomber_pos[1]] != 1:
            if cost_map[bomber_pos[0] + 1][bomber_pos[1]] is None \
                    or cost_map[bomber_pos[0] + 1][bomber_pos[1]] > move_cost:
                if game_map[bomber_pos[0] + 1][bomber_pos[1]] == 2:
                    AgentAI.fill_cost_map_rec(cost_map, game_map, bomb_radius, [bomber_pos[0] + 1, bomber_pos[1]],
                                              move_cost + 2 * bomb_radius + 11)
                elif game_map[bomber_pos[0] + 1][bomber_pos[1]] == 0:
                    AgentAI.fill_cost_map_rec(cost_map, game_map, bomb_radius, [bomber_pos[0] + 1, bomber_pos[1]],
                                              move_cost + 1)

        if game_map[bomber_pos[0]][bomber_pos[1] - 1] != 1:
            if cost_map[bomber_pos[0]][bomber_pos[1] - 1] is None \
                    or cost_map[bomber_pos[0]][bomber_pos[1] - 1] > move_cost:
                if game_map[bomber_pos[0]][bomber_pos[1] - 1] == 2:
                    AgentAI.fill_cost_map_rec(cost_map, game_map, bomb_radius, [bomber_pos[0], bomber_pos[1] - 1],
                                              move_cost + 2 * bomb_radius + 11)
                elif game_map[bomber_pos[0]][bomber_pos[1] - 1] == 0:
                    AgentAI.fill_cost_map_rec(cost_map, game_map, bomb_radius, [bomber_pos[0], bomber_pos[1] - 1],
                                              move_cost + 1)

        if bomber_pos[1] + 1 < len(game_map[0]) and game_map[bomber_pos[0]][bomber_pos[1] + 1] != 1:
            if cost_map[bomber_pos[0]][bomber_pos[1] + 1] is None \
                    or cost_map[bomber_pos[0]][bomber_pos[1] + 1] > move_cost:
                if game_map[bomber_pos[0]][bomber_pos[1] + 1] == 2:
                    AgentAI.fill_cost_map_rec(cost_map, game_map, bomb_radius, [bomber_pos[0], bomber_pos[1] + 1],
                                              move_cost + 2 * bomb_radius + 11)
                elif game_map[bomber_pos[0]][bomber_pos[1] + 1] == 0:
                    AgentAI.fill_cost_map_rec(cost_map, game_map, bomb_radius, [bomber_pos[0], bomber_pos[1] + 1],
                                              move_cost + 1)

    # Function based off of A* - WIP
    @staticmethod
    def a_star_pathing(bomber_pos, objective_pos, game_map, pass_walls):
        open_list = []
        closed_list = []

        if game_map[objective_pos[1]][objective_pos[0]] == 3:
            return [bomber_pos]

        new_objective = AgentAI.adjust_objective(bomber_pos, objective_pos, game_map)

        open_list.append(SearchNode(None, bomber_pos))

        while open_list:
            current_node = AgentAI.get_lowest_f_value_node_in_list(open_list)

            open_list.remove(current_node)
            closed_list.append(current_node)

            if current_node.position == new_objective:
                tmp_node = current_node
                path = []
                while tmp_node.parent is not None:
                    path.append(tmp_node.position)
                    tmp_node = tmp_node.parent

                if path:
                    return [path[-1]]

            child_nodes = []

            if current_node.position[0] > 1 \
                    and game_map[current_node.position[1]][current_node.position[0] - 1] != 1 \
                    and game_map[current_node.position[1]][current_node.position[0] - 1] != 3 \
                    and game_map[current_node.position[1]][current_node.position[0] - 1] != 4 \
                    and current_node.position[0] + 2 >= objective_pos[0]:
                if pass_walls or game_map[current_node.position[1]][current_node.position[0] - 1] != 2 \
                        or [current_node.position[0] - 1, current_node.position[1]] == objective_pos:
                    child_nodes.append(SearchNode(current_node,
                                                  [current_node.position[0] - 1, current_node.position[1]]))

            if current_node.position[0] < len(game_map[0]) - 1 \
                    and game_map[current_node.position[1]][current_node.position[0] + 1] != 1 \
                    and game_map[current_node.position[1]][current_node.position[0] + 1] != 3 \
                    and game_map[current_node.position[1]][current_node.position[0] + 1] != 4 \
                    and current_node.position[0] - 2 <= objective_pos[0]:
                if pass_walls or game_map[current_node.position[1]][current_node.position[0] + 1] != 2 \
                        or [current_node.position[0] + 1, current_node.position[1]] == objective_pos:
                    child_nodes.append(SearchNode(current_node,
                                                  [current_node.position[0] + 1, current_node.position[1]]))

            if current_node.position[1] > 1 \
                    and game_map[current_node.position[1] - 1][current_node.position[0]] != 1 \
                    and game_map[current_node.position[1] - 1][current_node.position[0]] != 3 \
                    and game_map[current_node.position[1] - 1][current_node.position[0]] != 4 \
                    and current_node.position[1] + 2 >= objective_pos[1]:
                if pass_walls or game_map[current_node.position[1] - 1][current_node.position[0]] != 2 \
                        or [current_node.position[0], current_node.position[1] - 1] == objective_pos:
                    child_nodes.append(SearchNode(current_node,
                                                  [current_node.position[0], current_node.position[1] - 1]))

            if current_node.position[1] < len(game_map) - 1 \
                    and game_map[current_node.position[1] + 1][current_node.position[0]] != 1 \
                    and game_map[current_node.position[1] + 1][current_node.position[0]] != 3 \
                    and game_map[current_node.position[1] + 1][current_node.position[0]] != 4 \
                    and current_node.position[1] - 2 <= objective_pos[1]:
                if pass_walls or game_map[current_node.position[1] + 1][current_node.position[0]] != 2 \
                        or [current_node.position[0], current_node.position[1] + 1] == objective_pos:
                    child_nodes.append(SearchNode(current_node,
                                                  [current_node.position[0], current_node.position[1] + 1]))

            for child in child_nodes:

                if closed_list.__contains__(child):
                    continue

                child.gCost = current_node.gCost + 1

                child.hCost = math.hypot(bomber_pos[0] - child.position[0], bomber_pos[1] - child.position[1])
                child.fCost = child.gCost + child.hCost

                if AgentAI.is_node_cheaper_in_open_list(child, open_list):
                    continue

                open_list.append(child)

        return [bomber_pos]

    # Function that gets the lowest fValue node in a list, to assist in a* pathing
    @staticmethod
    def get_lowest_f_value_node_in_list(node_list):
        best_node = None
        for node in node_list:
            if best_node is None:
                best_node = node
            elif node.fCost < best_node.fCost:
                best_node = node

        return best_node

    # Function that checks whether a node should be replaced in the list during the a* method
    @staticmethod
    def is_node_cheaper_in_open_list(node, list):
        for tmp_node in list:
            if tmp_node == node and node.gCost <= tmp_node.gCost:
                return True

        return False

    # Function that adds checkpointing to a* pathing by adjusting the current objective
    @staticmethod
    def adjust_objective(bomber_pos, objective, game_map):
        center = [int((len(game_map[0]) - 2) / 2) + 1, int((len(game_map) - 2) / 2) + 1]
        new_objective = []

        if bomber_pos[0] < objective[0]:
            if bomber_pos[0] < center[0] <= objective[0]:
                new_objective.append(center[0])
            else:
                new_objective.append(objective[0])
        elif bomber_pos[0] > objective[0]:
            if bomber_pos[0] > center[0] >= objective[0]:
                new_objective.append(center[0])
            else:
                new_objective.append(objective[0])
        else:
            new_objective.append(bomber_pos[0])

        if bomber_pos[1] < objective[1]:
            if bomber_pos[1] < center[1] <= objective[1]:
                new_objective.append(center[1])
            else:
                new_objective.append(objective[1])
        elif bomber_pos[1] > objective[1]:
            if bomber_pos[1] > center[1] >= objective[1]:
                new_objective.append(center[1])
            else:
                new_objective.append(objective[1])
        else:
            new_objective.append(bomber_pos[1])

        return new_objective

    @staticmethod
    def get_safe_position_list(bomber_pos, bomb_pos, bomb_radius, in_map, parent=[0, 0], safes=[]):
        tmp_safes = []

        if bomb_radius < 0 or (bomber_pos[0] != bomb_pos[0] and bomber_pos[1] != bomb_pos[1]):
            tmp_safes.append(bomber_pos)
        else:
            if in_map[bomber_pos[1]][bomber_pos[0] - 1] == 0 and [bomber_pos[0] - 1, bomber_pos[1]] != parent:
                AgentAI.get_safe_position_list([bomber_pos[0] - 1, bomber_pos[1]], bomb_pos, bomb_radius - 1,
                                               in_map, bomber_pos, safes)
            if in_map[bomber_pos[1]][bomber_pos[0] + 1] == 0 and [bomber_pos[0] + 1, bomber_pos[1]] != parent:
                AgentAI.get_safe_position_list([bomber_pos[0] + 1, bomber_pos[1]], bomb_pos, bomb_radius - 1,
                                               in_map, bomber_pos, safes)
            if in_map[bomber_pos[1] - 1][bomber_pos[0]] == 0 and [bomber_pos[0], bomber_pos[1] - 1] != parent:
                AgentAI.get_safe_position_list([bomber_pos[0], bomber_pos[1] - 1], bomb_pos, bomb_radius - 1,
                                               in_map, bomber_pos, safes)
            if in_map[bomber_pos[1] + 1][bomber_pos[0]] == 0 and [bomber_pos[0], bomber_pos[1] + 1] != parent:
                AgentAI.get_safe_position_list([bomber_pos[0], bomber_pos[1] + 1], bomb_pos, bomb_radius - 1,
                                               in_map, bomber_pos, safes)

        if tmp_safes:
            for safe in tmp_safes:
                if AgentAI.spot_is_safe(safe, bomb_pos, bomb_radius):
                    safes.append(safe)

    @staticmethod
    def get_safe_spot(bomber_pos, bomb_pos, closest_enemy, bomb_radius, in_map, parent=[0, 0], safes=[]):

        AgentAI.get_safe_position_list(bomber_pos, bomb_pos, bomb_radius, in_map, safes=safes)

        tmp_safes = safes.copy()

        if closest_enemy is not None:
            tmp_safes = AgentAI.filter_safes_by_closest_enemy(bomber_pos, safes, closest_enemy)

        if tmp_safes:
            return random.choice(tmp_safes)
        elif safes:
            return random.choice(safes)
        else:
            return [1, 1]

    @staticmethod
    def spot_is_safe(spot, bomb_pos, bomb_radius):
        if spot[0] < bomb_pos[0] - bomb_radius \
                or spot[0] > bomb_pos[0] + bomb_radius \
                or spot[1] < bomb_pos[1] - bomb_radius \
                or spot[1] > bomb_pos[1] + bomb_radius \
                or (spot[0] != bomb_pos[0] and spot[1] != bomb_pos[1]):
            return True
        else:
            return False

    @staticmethod
    def filter_safes_by_closest_enemy(bomber_pos, safes, closest_enemy):
        tmp_safes = []

        for safe in safes:
            if safe[0] <= bomber_pos[0] < closest_enemy[0] \
                    or safe[1] <= bomber_pos[1] < closest_enemy[1] \
                    or closest_enemy[0] < bomber_pos[0] <= safe[0] \
                    or closest_enemy[1] < bomber_pos[1] <= safe[1]:
                tmp_safes.append(safe)

        return tmp_safes
