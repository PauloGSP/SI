import asyncio
import random
from copy import deepcopy
from functools import partial


class State:
    def __init__(self, state, moves): # state is car dict
        self.state = state
        self.moves = moves  # list of moves executed to reach this state from the previous e.g. [('A', 'd'), ('A', 'd')]


class Node:
    def __init__(self, state: State, parent):
        self.parent = parent
        self.state = state
        self.depth = parent.depth + 1 if self.parent else 0
        self.score = float('-inf')

        new_state_dict = { k: frozenset(v['coords']) for k,v in state.state.items() }
        self.hashable_state = tuple(new_state_dict.items())

    def state_in_parent(self, state):
        if (self.state.state == state):
            return True
        
        return False if not self.parent else self.parent.state_in_parent(state)
    
    def get_previous_positions_for_car(self, car_id):
        """Returns a set of given car's previous positions throughout parent nodes"""

        positions = set()
        positions.add(frozenset(self.state.state[car_id]['coords']))

        if self.parent:
            positions.update(self.parent.get_previous_positions_for_car(car_id))
        
        return positions

    def is_new_car_position(self):
        """Returns true if the car moved to reach this node's state is in a position it's never been in before"""

        if not self.state.moves:
            return False

        car_id = self.state.moves[0][0]
        curr_pos = frozenset(self.state.state[car_id]['coords'])

        if self.parent:
            previous_positions = self.parent.get_previous_positions_for_car(car_id)
            return curr_pos not in previous_positions
        else:
            return True

    


class SearchTree:
    def __init__(self, root_state, goal_evaluator, actions, heuristic, sorting_heuristic, depth_lim = float('inf')):
        self.open_nodes = [Node(root_state, None)]
        self.goal_evaluator = goal_evaluator # method: determines if a given state is a goal state or not
        self.actions = actions               # method: returns all the possible actions for a given state
        self.heuristic = heuristic           # method: given state, evaluates its score
        self.depth_lim = depth_lim
        self.best_node = self.open_nodes[0]  # best node so far (starts out as the root node)
        self.sorting_heuristic = sorting_heuristic
        self.previous_states = dict()        # stores previous states as keys, with their shallowest depth as their value

    def path(self, node):
        if node.parent == None:
            return [node.state]

        return self.path(node.parent) + [node.state]

    def node_path(self, node):
        if node.parent == None:
            return [node]
        
        return self.node_path(node.parent) + [node]


    async def search(self):
        try:
            while self.open_nodes:

                await asyncio.sleep(0)

                node = self.open_nodes.pop(0)

                if self.goal_evaluator(node.state):
                    print("found solution")
                    self.best_node = node
                    return self.path(node)

                if node.score > self.best_node.score and node.depth >= self.best_node.depth:  # and node.depth == self.depth_lim-1: # if current node is better than previous best node
                    self.best_node = node
                elif node.score == self.best_node.score:
                    if random.randint(0, 1) == 0:
                        self.best_node = node
                
                #if node.depth > self.best_node.depth:
                #    if node.score + 5 > self.best_node.score:
                #        self.best_node = node
                #elif node.score > self.best_node.score:
                #    self.best_node = node


                if node.depth < self.depth_lim-1:

                    new_nodes = []

                    #TODO: FUNÇAO GET MOVES            
                    for state in self.actions(node.state.state):
                        new_node = Node(state, node)

                        if new_node.hashable_state in self.previous_states:
                            if new_node.depth < self.previous_states[new_node.hashable_state]:
                                self.previous_states[new_node.hashable_state] = new_node.depth
                            else:
                                #print("Avoided repeated state")
                                #print(new_node.hashable_state)
                                #print(new_node.depth)
                                #print(self.previous_states)
                                #print()
                                continue


                        if not node.state_in_parent(state.state):
                            self.previous_states[new_node.hashable_state] = new_node.depth
                            new_node.score = self.heuristic(state.state)
                            new_nodes.append(new_node)

                    await self.add_to_open(new_nodes)
                
            return self.path(self.best_node)
        except asyncio.CancelledError:
            return self.path(self.best_node)


    async def add_to_open(self, new_nodes):
        await asyncio.sleep(0)

        #priority_nodes = None
        # TODO: calculate car chain dynamically
        # open branches on the car chain first
        #if priority_nodes:
        #    for priority in priority_nodes:
        #        for n in range(len(new_nodes)):
        #            node = new_nodes[n]
        #            if node.state.move[0]==priority:
        #                self.open_nodes.extend([node])
        #                new_nodes.pop(n)
        #    self.open_nodes.extend(new_nodes)
        #else:

        #self.open_nodes.extend(new_nodes)
        #self.open_nodes = sorted(self.open_nodes, key=lambda n: self.sorting_heuristic(n.state.state))


        # breadth first
        self.open_nodes.extend(new_nodes)  
        
        # A*
        #self.open_nodes.extend(new_nodes)
        #self.open_nodes.sort(key=lambda n: n.score - n.depth, reverse=True)
        #self.open_nodes.sort(key=lambda n: n.score - n.depth + n.is_new_car_position(), reverse=True)

        # a mix of A* and breadth-first, sort new_nodes with A* and add them to the end of the list
        #new_nodes.sort(key=lambda n: n.score - n.depth, reverse=True)
        #self.open_nodes.extend(new_nodes)

        # don't even know what to call this:
        # keep the first 5 nodes as they are and reorder the rest as follows:
        # from the remaining nodes, first best, worst, second best, second worst, third best, ...
        #self.open_nodes.extend(new_nodes)
        #remaining = sorted(self.open_nodes[5:], key=lambda n: n.score - n.depth, reverse=True)
        #for i in range(1, len(remaining)//2):
        #    remaining.insert(i, remaining.pop())
        #self.open_nodes[5:] = remaining





def calculate_moves(cars: dict, grid):
    """
    Returns list of states containing possible moves given current grid and dict of cars.
    """

    def count_open_spaces(grid, car_id, car):
        """
        Returns a count of consecutive free spaces on the back and front of the car.
        If the car is vertical, returns (up_count, down_count)
        If the car is horizontal, returns (left_count, right_count)
        """

        if car['direction'] == 'H':
            x_start = 0
            x_increment = 1
            y_start = car['coords'][0][1]
            y_increment = 0
        elif car['direction'] == 'V':
            x_start = car['coords'][0][0]
            x_increment = 0
            y_start = 0
            y_increment = 1

        is_pre_car = True
        pre_car_count = 0
        post_car_count = 0

        for i in range(len(grid)):
            
            position = grid[y_start+y_increment*i][x_start+x_increment*i]

            if position == 'o':
                if is_pre_car:
                    pre_car_count += 1
                else:
                    post_car_count += 1
            elif position == car_id:
                is_pre_car = False
            else:
                if is_pre_car:
                    pre_car_count = 0
                else:
                    break

        return pre_car_count, post_car_count


    def add_to_states(states, cars, car_id, direction, count):
        x_inc = count if direction == 'd' else -count if direction == 'a' else 0
        y_inc = count if direction == 's' else -count if direction == 'w' else 0

        new_cars = deepcopy(cars)
        new_cars[car_id]['coords'] = [ (x + x_inc, y + y_inc) for x, y in cars[car_id]['coords'] ]

        states.append( State(new_cars, [(car_id, direction)]*count) )


    states = []

    # determine current grid state
    new_grid = [None]*len(grid)
    for i in range(len(grid)):
        new_grid[i] = ['o']*len(grid)

    for id, car in cars.items():
        car_coords = cars[id]['coords']
        for coord in car_coords:
            new_grid[coord[1]][coord[0]] = id

    # determine possible moves
    for id, car in cars.items():
        if id == 'x': continue

        free_before, free_after = count_open_spaces(new_grid, id, car)

        if free_before > 0:
            add_to_states(states, cars, id, 'w' if car['direction'] == 'V' else 'a', free_before)
        if free_after > 0:
            add_to_states(states, cars, id, 's' if car['direction'] == 'V' else 'd', free_after)

    return states


def calculate_board_heuristic(cars, grid_size):
    # heuristic worthy values:
    # distancia do carro vermelho à meta
    # numero de carros a bloquear o carro vermelho
    # (5, )

    coords = cars['A']['coords']
    max_x = max([t[0] for t in coords])
    distance = 5 - max(coords, key = lambda t: t[0])[0]
    blocking_list = {'a': [], 'b': []}
    blocking_count = 0

    num_blocking_cars = 0
    for k, v in cars.items():
        if k == 'A':
            continue
        

        is_blocking = 0
        car_coords = cars[k]['coords']

        #if min([t[0] for t in car_coords]) >= max_x:
        is_blocking = sum([1 if c[1] == coords[1][1] else 0 for c in car_coords])

        if is_blocking:

            blocking_list = [
                float("inf") if car_coords[0][1] == 0 else 0,
                float("inf") if car_coords[-1][1] == grid_size-1 else 0
            ]

            num_blocking_cars += is_blocking
            car_coords_x = [coord[0] for coord in car_coords]
            for car, values in cars.items():
                if car != k and car != 'A' and car != 'x':
                    other_car_coords_x = [coord[0] for coord in values['coords']]
                    if [x for x in other_car_coords_x if x in car_coords_x]:
                        if max(values['coords'], key=lambda x: x[1])[1] < min(car_coords, key=lambda x: x[1])[1]:
                            #blocking_list['a'].append(car)
                            blocking_list[0] += 1
                        else:
                            #blocking_list['b'].append(car)
                            blocking_list[1] += 1
            blocking_count += min(blocking_list)

    #return - num_blocking_cars - distance - (len(blocking_list['a']) + len(blocking_list['b']))
    #return - num_blocking_cars - distance - min(len(blocking_list['a']), len(blocking_list['b']))
    return - num_blocking_cars - distance - blocking_count
    


def sorting_heuristic(cars, grid_size):
    
    blocking_cars = []
    car_A_coords = cars['A']['coords']

    for car_id, val in cars.items():
        if car_id in 'Ax': continue

        car_coords = val['coords']

        if min([t[0] for t in car_coords]) >= max([t[0] for t in car_A_coords]):
            is_blocking = sum([1 if c[1] == car_A_coords[1][1] else 0 for c in car_coords])
        else:
            is_blocking = 0

        if is_blocking:
            second_blocking_cars = [
                float("inf") if car_coords[0][1] == 0 else 0,
                float("inf") if car_coords[-1][1] == grid_size-1 else 0
            ]

            for other_car, other_val in cars.items():
                if other_car not in "Ax"+car_id:
                    other_car_coords_x = [coord[0] for coord in other_val['coords']]
                    
                    if [x for x in other_car_coords_x if x == car_coords[0][0]]:
                        # if max y of other_car is less than min y of blocking car, other_car is above blocking car
                        if max(other_val['coords'], key=lambda coord: coord[1])[1] < min(car_coords, key=lambda coord: coord[1])[1]:
                            second_blocking_cars[0] += 1
                        else:
                            second_blocking_cars[1] += 1
            
            blocking_cars.append(min(second_blocking_cars))
    
    return min(blocking_cars) if blocking_cars else 0
            



def is_goal_state(state, grid):
    return state.state['A']['coords'][-1][0] == len(grid)-1


class RushHourSearch(SearchTree):
    def __init__(self, grid, root_state, depth_lim=float('inf')):
        super().__init__(
            root_state,
            partial(is_goal_state, grid=grid),
            partial(calculate_moves, grid=grid),
            partial(calculate_board_heuristic, grid_size=len(grid)),
            sorting_heuristic=partial(sorting_heuristic, grid_size=len(grid)),
            depth_lim=depth_lim
        )

