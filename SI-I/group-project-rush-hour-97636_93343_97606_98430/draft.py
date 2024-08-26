import asyncio
from copy import deepcopy
from search import State, RushHourSearch, Node
import time

def get_new_state(grid, cars, move):
    """
    Given current grid, current car dict, and a move returns new grid.
    Move is (car_id, direction) tuple
    Raises exception if move is invalid
    """

    new_grid = [ ['o']*len(grid) for _ in range(len(grid)) ]
    
    for car, data in cars.items():
        if car == move[0]:
            x_inc = 1 if move[1] == 'd' else -1 if move[1] == 'a' else 0
            y_inc = 1 if move[1] == 's' else -1 if move[1] == 'w' else 0
            coords = [ (x+x_inc, y+y_inc) for x,y in data['coords'] ]
        else:
            coords = data['coords']
        
        for x,y in coords:
            if not (0 <= x < len(grid)) or not (0 <= y < len(grid)):
                return grid
                #raise Exception("Invalid move for current grid")
            if new_grid[y][x] != 'o':
                return grid
            new_grid[y][x] = car

    return new_grid

def process_grid(grid_str):
    """
    Takes in the string representing the grid, which is available in the 'state' variable,
    and turns it into an easier-to-use matrix, which can be accessed with (x,y) coordinates.
    The x-axis grows from left to right.
    The y-axis grows downwards.

    Example: for the grid string 'ooooooooooooAAooooooooBoooooBoooooBo', we have the
    following grid matrix:

     x 0 1 2 3 4 5
    y
    0  o o o o o o   -> grid_str[0:6]
    1  o o o o o o   -> grid_str[6:12]
    2  A A o o o o   -> grid_str[12:18]
    3  o o o o B o   -> grid_str[18:24]
    4  o o o o B o   -> grid_str[24:30]
    5  o o o o B o   -> grid_str[30:36]

    This means that the matrix (list of lists) to be returned is:
    
    grid = [
        [o,o,o,o,o,o],   -> grid_str[0:6]
        [o,o,o,o,o,o],   -> grid_str[6:12]
        [A,A,o,o,o,o],   -> grid_str[12:18]
        [o,o,o,o,B,o],   -> grid_str[18:24]
        [o,o,o,o,B,o],   -> grid_str[24:30]
        [o,o,o,o,B,o]    -> grid_str[30:36]
    ]
    """

    grid = []
    positions = dict()

    for i in range(0,31,6):
        newline = grid_str[i:(i+6)]
        newline = [*newline] # split into list of letters
        
        for j in range(len(newline)):
            cell = newline[j]
            if cell == 'o':
                continue
            elif cell not in positions.keys():
                # coords (x, y)
                positions[cell] = { 'coords': [(j,int(i/6))] }
            else:
                positions[cell]['coords'].append((j,int(i/6)))

        # grid access with [y][x]
        grid.append(newline)
    

    # calculate car directions
    for k, val in positions.items():
        if k == 'x':
            val['direction'] = None # circular obstruction
        else:
            val['direction'] = 'V' if val['coords'][0][0] - val['coords'][1][0] == 0 else 'H'


    return grid, positions

def calculate_board_heuristic(cars):
    # heuristic worthy values:
    # distancia do carro vermelho à meta
    # numero de carros a bloquear o carro vermelho
    # (5, )

    coords = cars['A']['coords']
    max_x = max([t[0] for t in coords])
    distance = 5 - max(coords, key = lambda t: t[0])[0]
    blocking_list = {'a': [], 'b': []}

    num_blocking_cars = 0
    for k, v in cars.items():
        if k == 'A':
            continue
        

        is_blocking = 0
        car_coords = cars[k]['coords']

        if min([t[0] for t in car_coords]) >= max_x:
            is_blocking = sum([1 if c[1] == coords[1][1] else 0 for c in car_coords]) 

        if is_blocking:
            num_blocking_cars += is_blocking
            car_coords_x = [coord[0] for coord in car_coords]
            for car, values in cars.items():
                if car != k and car != 'A' and car != 'x':
                    other_car_coords_x = [coord[0] for coord in values['coords']]
                    if [x for x in other_car_coords_x if x in car_coords_x]:
                        if max(values['coords'], key=lambda x: x[1])[1] < min(car_coords, key=lambda x: x[1])[1]:
                            blocking_list['a'].append(car)
                        else:
                            blocking_list['b'].append(car)

    return - num_blocking_cars - distance - min(len(blocking_list['a']), len(blocking_list['b']))


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

        print(f"car {id} open spaces: {free_before} before, {free_after} after")


        if free_before > 0:
            add_to_states(states, cars, id, 'w' if car['direction'] == 'V' else 'a', free_before)
        if free_after > 0:
            add_to_states(states, cars, id, 's' if car['direction'] == 'V' else 'd', free_after)

    return states



grid_str = "ooBBxooDDHJKAAGHJKooGIooooGIEEFFFooo"
grid, cars = process_grid(grid_str)

for row in grid:
    print("".join(row))


async def search_for_solution():
    ts_start = time.time()
    st = RushHourSearch(grid, State(cars, []), depth_lim=15)
    #result = await st.search()
    result = await st.search()
    # combines each state's list of moves into one list
    # e.g. [[('B', 'w')], [('A', 'd'), ('A', 'd')]] --> [('B', 'w'), ('A', 'd'), ('A', 'd')]
    output = [item for sublist in [ s.moves for s in result[1:] ] for item in sublist]

    print("searched in:", time.time() - ts_start)
    print("moves to perform:", output)
    print("path found:")
    for n in st.node_path(st.best_node):
        print("    score:", n.score)

loop = asyncio.get_event_loop()
loop.run_until_complete(search_for_solution())

#new_grid = get_new_state(grid, cars, ('A', 'd'))
#expected_grid = ''.join([item for sublist in new_grid for item in sublist])
#print(grid_str)
#print(expected_grid)


#print(calculate_board_heuristic(cars))


#states = calculate_moves(cars, grid)
#for s in states:
#    print(s)
