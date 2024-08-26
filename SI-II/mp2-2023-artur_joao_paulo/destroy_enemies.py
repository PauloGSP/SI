# Function to determinate density of places based on the number of enemies
import math

class Destroy_enemies:

    @staticmethod
    def dist_cal(x,y):
        return math.hypot(x[0] - y[0], x[1] - y[1])

    @staticmethod
    def closest_wall(player, walls):
        dist = float("inf")
        closest_wall = []
        for wall in walls:
            new_dist = Destroy_enemies.dist_cal(player, wall)
            if new_dist < dist:
                dist = new_dist
                closest_wall = wall

        return closest_wall

    @staticmethod
    def enemy_is_in_range(player, enemy_pos, rang=1):

        if not enemy_pos:
            return False

        if player[1] % 2 != 0:
            if math.fabs(player[0] - enemy_pos[0]) <= rang and player[1] == enemy_pos[1]:
                return True
        if player[0] % 2 != 0:
            if math.fabs(player[1] - enemy_pos[1]) <= rang and player[0] == enemy_pos[0]:
                return True
        else:
            return False


    @staticmethod
    def place_drop_bomb(drop_bomb_place, game_map, direction, tmp=1):

        n_target = drop_bomb_place

        ciclo = 0

        while tmp != 0:
            n_target = [n_target[0] + direction[0], n_target[1] + direction[1]]

            if 31 <= n_target[0] <= 0 or 51 <= n_target[1] <= 0:
                break

            if game_map[n_target[1]][n_target[0]] > 0:
                if direction == [0, -1]:
                    direction = [-1, 0]
                elif direction == [-1, 0]:
                    direction = [0, 1]
                elif direction == [0, 1]:
                    direction = [1, 0]
                else:
                    direction = [0, -1]

                ciclo += 1

                if ciclo == 5:
                    break

                n_target = drop_bomb_place

            else:
                tmp -= 1
                drop_bomb_place = n_target

        return drop_bomb_place, direction






