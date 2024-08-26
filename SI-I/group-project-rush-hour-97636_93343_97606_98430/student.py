"""Example client."""
import asyncio
from copy import deepcopy
import getpass
import json
import os
import time
import random
import itertools

# Next 4 lines are not needed for AI agents, please remove them from your code!
import pygame
import websockets

from search import RushHourSearch, State

pygame.init()
program_icon = pygame.image.load("data/icon2.png")
pygame.display.set_icon(program_icon)

async def agent_loop(server_address="localhost:8000", agent_name="student"):
    """Example client loop."""
    async with websockets.connect(f"ws://{server_address}/player") as websocket:

        # Receive information about static game properties
        await websocket.send(json.dumps({"cmd": "join", "name": agent_name}))

        # Next 3 lines are not needed for AI agent
        SCREEN = pygame.display.set_mode((299, 123))
        SPRITES = pygame.image.load("data/pad.png").convert_alpha()
        SCREEN.blit(SPRITES, (0, 0))

        DIRECTION_MAP = {
            ('H', 1): 'd',
            ('H', -1): 'a',
            ('V', 1): 's',
            ('V', -1): 'w'
        }


        inputs = []
        expected_grid = None
        global dimensions

        solution = []
        prev_move = None
        curr_move = None
        processing_tree = False
        #loop = asyncio.get_event_loop()

        level = 0
        task = None


        while True:
            try:
                state = json.loads(
                    await websocket.recv()
                )  # receive game update, this must be called timely or your game will get out of sync with the server


                grid_str = state.get("grid").split(" ")[1]
                dimensions = state.get("dimensions")
                grid, positions  = process_grid(grid_str)

                # check if level changed
                if state.get('level') != level:
                    if task:
                        task.cancel()
                        await task
                    print()
                    print("CHANGED LEVEL!")
                    inputs.clear()
                    solution.clear()
                    processing_tree = False
                    level = state.get('level')
                    prev_move = None
                    curr_move = None

                elif expected_grid != grid_str:
                    print("!!! CRAZY DRIVER ALERT !!!")
                    print('actual grid  :', grid_str)
                    print('expected grid:', expected_grid)
                    #print('attempted move:', curr_move)

                    differences = dict()
                    move_fix = None

                    # compare grids to find culprit, saving only the first difference
                    for actual, expected in zip(grid_str, expected_grid):
                        if actual == expected or actual in differences or expected in differences:
                            continue

                        if actual == 'o':   # car moved either down or right
                            differences[expected] = 1
                        else:               # car moved either up or right
                            differences[actual] = -1


                    if len(differences) == 1:     # only one car was out of place
                        car, motion = differences.popitem()

                        if curr_move and car == curr_move[0]:    # crazy driver was the car we were trying to move
                            curr_motion = 1 if curr_move[1] in "sd" else -1
                            if curr_motion == motion:
                                # car moved by itself to the intended location, let it be and get to the next move :stonks:
                                inputs.clear()
                            else:
                                # correct car's position
                                move_fix = (car, -motion)
                        else:
                            # correct car's position
                            move_fix = (car, -motion)

                    else:     # one car's unexpected movement blocked another
                        print("car was blocked")
                        differences.pop(prev_move[0])   # remove the car we were trying to move
                        curr_move = prev_move           # redo last move
                        car, motion = differences.popitem()  # get the other car
                        move_fix = (car, -motion)
                        

                    if move_fix:
                        inputs.clear()
                        if curr_move: solution.insert(0, curr_move)

                        # replace 1/-1 in move_fix with proper input
                        move_fix = (move_fix[0], DIRECTION_MAP[(positions[move_fix[0]]['direction'], move_fix[1])])

                        inputs.extend(get_inputs_to_move_car(
                            state.get("selected"),
                            state.get("cursor"),
                            positions[move_fix[0]]["coords"],
                            move_fix[0],
                            move_fix[1]
                        ))
                        prev_move = curr_move
                        curr_move = move_fix

                        print('fixing with ', move_fix)
                        print()




                #move_scores = {}
                #ts_start = time.time()
                #next_move, _ = calculate_lookahead(grid, positions, size=3, k=20, history=move_scores)
                #print("lookahead in ", time.time() - ts_start)

                #for row in grid:
                #    print(''.join(row))
                #
                #print_move_dict(move_scores)
                #print("chosen move:", next_move)
                #print()

                # ts_start = time.time()

                # st = RushHourSearch(grid, State(positions, None), 5)

                # solution = st.search()
                # print("searched grid in ", time.time() - ts_start)

                # print([s.move for s in solution])
                # print()
                
                if not processing_tree and not solution and not inputs:
                    # cancel previous task if we somehow enter this if while still processing (it's happened)
                    if task:
                        task.cancel()
                        await task

                    processing_tree = True
                    print("searching for grid:")
                    for row in grid:
                        print("  ", "".join(row))

                    task = asyncio.create_task(get_solution(solution, deepcopy(grid), deepcopy(positions), depth_lim=100, timeout=20))


                if solution and not inputs:
                    prev_move = curr_move
                    curr_move = solution.pop(0)

                    selected = state.get('selected')
                    cursor = state.get("cursor")
                    #print('initial selected:', selected)
                    #print('initial cursor:', cursor)

                    inputs.extend(get_inputs_to_move_car(
                        selected,
                        cursor,
                        positions[curr_move[0]]["coords"],
                        curr_move[0],
                        curr_move[1]
                    ))

                    processing_tree = False

                    #print(inputs)
                    #print(curr_move)
                    #print(solution)
                    #print()


                next_input = inputs.pop(0) if inputs else ""

                #print("current grid: ", grid_str)
                if curr_move and not inputs:
                    expected_grid = ''.join([item for sublist in get_new_state(grid, positions, curr_move) for item in sublist])
                    prev_move = curr_move
                    curr_move = None
                else:
                    expected_grid = grid_str
                #print("expected grid:", expected_grid)


                #print(f"{time.ctime(time.time())} - sending ", next_input)
                await websocket.send(
                    json.dumps({"cmd": "key", "key": next_input})
                )  # send key command to server - you must implement this send in the AI age


            except websockets.exceptions.ConnectionClosedOK:
                print("Server has cleanly disconnected us")
        
                print("average -> "+ str(sum(timedic.values())/len(timedic.values())))
                print("total time speant searching -> " + str(sum(timedic.values())))
                print("max -> "+ str(max(timedic.values())))
                print("--------------------------------------Final time table---------------------------------------")
                print(timedic)
                return


async def get_solution(output, grid, positions, depth_lim, timeout=None):
    """Asynchronous processing of the search tree."""

    try:
        ts_start = time.time()
        st = RushHourSearch(grid, State(positions, []), depth_lim=depth_lim)
        #result = await st.search()
        task = asyncio.create_task(st.search())
        await asyncio.wait_for(task, timeout=timeout)
        result = task.result()
        # combines each state's list of moves into one list
        # e.g. [[('B', 'w')], [('A', 'd'), ('A', 'd')]] --> [('B', 'w'), ('A', 'd'), ('A', 'd')]
        output.extend(item for sublist in [ s.moves for s in result[1:] ] for item in sublist)

        print("searched in:", time.time() - ts_start)
        increment()
        timedic[COUNT]=time.time() - ts_start
        
                

        #print("grid:")
        #for line in grid:
        #    print("".join(line))
        #print("output:", [o.move for o in output])
    except asyncio.CancelledError:
        print("Cancelled search")
    except asyncio.TimeoutError:
        result = task.result()
        output.extend(item for sublist in [ s.moves for s in result[1:] ] for item in sublist)
        print(f"timeout at depth {len(result)}, best solution found:", output)


def get_inputs_to_move_car(selected, cursor, car_coords, car_id, direction):
    inputs = []
    if selected != car_id:
        if selected != '':
            inputs.append(' ')
        goal =  closest_car_cell(cursor, car_coords)
        inputs.extend(move_to(cursor, goal))
        inputs.append(' ')

    inputs.append(direction)

    return inputs



# ------------------------------------------------------------------------------------------------------------------------------------

def print_move_dict(moves: dict, indent = 0):
    for key, val in moves.items():
        print(f"{' '*indent*4}↳{key} - local score: {val['base_score']}, cumulative score: {val['cum_score']}")
        if len(val['next']) > 0:
            print_move_dict(val['next'], indent+1)




def move_to(current, goal):
    """
    Calculate the next move based on the current location and the goal location. This
    next move is a move along the
    Make the cursor move from location 'current' to location 'goal'. Both 'current' and
    'goal' are [x,y] coordinates.
    """

    #print(f"moving from {current} to {goal}")

    moves = []

    diffX = goal[0] - current[0]
    diffY = goal[1] - current[1]

    hor = ["a"] if diffX<0 else ["d"]
    ver = ["w"] if diffY<0 else ["s"]

    moves.extend(hor*abs(diffX))
    moves.extend(ver*abs(diffY))

    return moves



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
    for i in range(0,len(grid_str) - (dimensions[0] - 1), dimensions[0]):
        newline = grid_str[i:(i+dimensions[0])]
        newline = [*newline] # split into list of letters
        
        for j in range(len(newline)):
            cell = newline[j]
            if cell == 'o':
                continue
            elif cell not in positions.keys():
                # coords (x, y)
                positions[cell] = { 'coords': [(j,int(i/dimensions[0]))] }
            else:
                positions[cell]['coords'].append((j,int(i/dimensions[0])))

        # grid access with [y][x]
        grid.append(newline)
    # calculate car directions
    for k, val in positions.items():
        if k == 'x':
            val['direction'] = None # circular obstruction
        else:
            val['direction'] = 'V' if val['coords'][0][0] - val['coords'][1][0] == 0 else 'H'

    return grid, positions


def get_cars_coords(cars, car_id, moves=None):
    """
    Given dictionary of cars, car_id and optionally list of moves,
    returns list of car coordinates after given move is applied
    """

    if moves:
        x_inc, y_inc = (0,0)
        for m in moves:
            x_inc += 1 if m == 'd' else -1 if m == 'a' else 0
            y_inc += 1 if m == 's' else -1 if m == 'w' else 0
        return [ (x + x_inc, y + y_inc) for x, y in cars[car_id]['coords'] ]

    return cars[car_id]['coords']
    


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
            elif new_grid[y][x] != 'o':
                return grid
            new_grid[y][x] = car

    return new_grid



# TODO: this could be performed in the `process_grid` function to reduce total grid iterations
def calculate_moves(grid, cars: dict, car_moves: dict = dict()):
    """
    Returns list of possible moves given current grid and dict of cars.
    Each list entry is a tuple containing car_id and movement direction
    """

    #print("CALCULATE MOVES")


    # stores (car_id, move) tuples, move one of {'w', 'a', 's', 'd'}
    moves = []


    # determine current grid state
    new_grid = [None]*len(grid)
    for i in range(len(grid)):
        new_grid[i] = ['o']*len(grid)

    for id, car in cars.items():
        car_coords = get_cars_coords(cars, id, car_moves.get(id, None))

        #print("NEW CAR COORDS:")


        for coord in car_coords:
            new_grid[coord[1]][coord[0]] = id


    for id, car in cars.items():
        # (x, y) coords
        car_coords = get_cars_coords(cars, id, car_moves.get(id, None))
        car_start = car_coords[0]
        car_end = car_coords[-1]

        # grid[y][x] indexation
        if car['direction'] == 'V':
            if car_start[1] > 0 and new_grid[car_start[1]-1][car_start[0]] == 'o':
                #new_grid, new_cars = get_new_state(grid, cars, (id, 'w'))
                #moves.append((id, 'w', new_grid, new_cars))
                moves.append((id, 'w'))
            if car_end[1] < len(grid)-1 and new_grid[car_end[1]+1][car_end[0]] == 'o':
                #new_grid, new_cars = get_new_state(grid, cars, (id, 's'))
                #moves.append((id, 's', new_grid, new_cars))
                moves.append((id, 's'))

        elif car['direction'] == 'H':
            if car_start[0] > 0 and new_grid[car_start[1]][car_start[0]-1] == 'o':
                #new_grid, new_cars = get_new_state(grid, cars, (id, 'a'))
                #moves.append((id, 'a', new_grid, new_cars))
                moves.append((id, 'a'))
            if car_end[0] < len(grid)-1 and new_grid[car_end[1]][car_end[0]+1] == 'o':
                #new_grid, new_cars = get_new_state(grid, cars, (id, 'd'))
                #moves.append((id, 'd', new_grid, new_cars))
                moves.append((id, 'd'))


    #print("new grid:", new_grid)


    return moves



def get_k_best_moves(moves, k, cars, car_moves = None):
    """
    Returns list of k highest rated moves' indexes, in order of highest score to lowest
    """
    # TODO this looks very inefficient, find better way

    best_move_scores = [None]*k
    best_move_indexes = [None]*k

    for i, move in enumerate(moves):
        # move = (car_id, direction)
        # add move to dict, or created if not existent
        if car_moves is None:
            car_moves = {move[0]: [move[1]]}
        elif move[0] in car_moves:
            car_moves[move[0]].append(move[1])
        else:
            car_moves[move[0]] = [move[1]]

        score = calculate_board_heuristic(cars, car_moves)

        # remove added element
        if len(car_moves[move[0]]) == 1:
            car_moves.pop(move[0])
        else:
            car_moves[move[0]].pop()

        for j, l_score in enumerate(best_move_scores):
            if l_score is None or score > l_score:   # > ? nahhhh
                #best_move_scores[j+1:] = best_move_scores[j:-1]
                #best_move_scores[j] = score
                #best_move_indexes[j+1:] = best_move_indexes[j:-1]
                #best_move_indexes[j] = i
                best_move_scores.insert(j, score)
                best_move_indexes.insert(j, i)
                best_move_indexes.pop()
                best_move_scores.pop()
                break
            elif l_score == score:
                if random.randint(0, 1):
                    best_move_scores.insert(j, score)
                    best_move_indexes.insert(j, i)
                else:
                    best_move_scores.insert(j+1, score)
                    best_move_indexes.insert(j+1, i)
                best_move_indexes.pop()
                best_move_scores.pop()
                break
                
    return best_move_indexes, best_move_scores



def closest_car_cell(cursor,car_coords):
    """
    Returns the car cell which is closest to the cursor. The closest car cell is defined as
    the car cell to which the cursor can get too with the smallest number of moves.
    """

    closest_cell = None
    min_moves = None

    for car_cell in car_coords:
        diffX = abs(car_cell[0] - cursor[0])
        diffY = abs(car_cell[1] - cursor[1])
        num_moves = diffX + diffY

        if (not min_moves) or (num_moves<min_moves):
            min_moves = num_moves
            closest_cell = car_cell

    return closest_cell



def get_obstruction_on_side(grid, car, side):
    """
    Returns the adjacent obstacle to the car on the given side ('w', 'a', 's' and 'd').
    Returns None when obstacle is out of bounds.
    Will raise an exception if calling with 'w' or 's' on a horizontal car,
    or when calling with 'a' or 'd' on a vertical car.
    """

    obstacle = None

    # obstacle (x, y)
    if car['direction'] == 'V':
        if side == 'w':
            obstacle = (car[0][0], car[0][1]-1)
        elif side == 's':
            obstacle = (car[-1][0], car[-1][1]+1)
    elif car['direction'] == 'H':
        if side == 'a':
            obstacle = (car[0][0]-1, car[0][1])
        elif side == 'd':
            obstacle = (car[-1][0]+1, car[-1][1])

    
    if not obstacle:
        raise Exception("Called 'get_obstruction_on_side' with invalid car direction and side combination!")

    try:
        # grid access with [y][x]
        return grid[obstacle[1]][obstacle[0]]
    except IndexError:
        return None



def count_open_spaces(grid, car_id, car):
    """
    Returns a count of consecutive free spaces on the back and front of the car.
    If the car is vertical, returns (up_count, down_count)
    If the car is horizontal, returns (left_count, right_count)
    """

    if car['direction'] == 'H':
        x_start = -1
        x_increment = 1
        y_start = car['coords'][0][1]
        y_increment = 0
    elif car['direction'] == 'V':
        x_start = car['coords'][0][0]
        x_increment = 0
        y_start = 1
        y_increment = -1

    
    is_pre_car = True
    pre_car_count = 0
    post_car_count = 0

    for _ in range(len(grid)):
        
        position = grid[y_start+y_increment][x_start+x_increment]

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



def calculate_board_heuristic(cars, car_moves):
    # heuristic worthy values:
    # distancia do carro vermelho à meta
    # numero de carros a bloquear o carro vermelho
    # (5, )

    coords = get_cars_coords(cars, 'A', car_moves.get('A', None))
    distance = dimensions[0] - max(coords, key = lambda t: t[0])[0]
    blocking_list = {'a': [], 'b': []}

    num_blocking_cars = 0
    for k, v in cars.items():
        if k == 'A':
            continue
        
        car_coords = get_cars_coords(cars, k, car_moves.get(k, None))
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

    return - num_blocking_cars*0 - distance - len(min(blocking_list.values(), key=lambda v:len(v)))



def calculate_lookahead(grid, positions, size=3, k=3, car_moves = dict(), history: dict = None):
    """
    function calculate_lookahead: ...

    INPUT:
        - grid:      current grid (matrix)
        - positions: dictionary with the positions of each car; example: {'A': {'coords': [(0,0), (0,1)], 'direction': 'H'}, 'B': ...}
        - size:      size of the desired lookahead for the move
        - k:         max number of moves to consider for each grid state, using k best moves

    OUTPUT:
        - (new_move, score): the best move found as a (car_id, direction) tuple, and its score
    """


    #scores1 [10 11]
    #scores1_1[12 4]  scores1_2[3 17]    -> scores1[22 28]


    #print(f"lookahead size {size},\n    grid: {grid},\n    positions: {positions},\n    car_moves: {car_moves}")


    if size == 0:
        return None, 0

    possible_next_moves = calculate_moves(grid, positions, car_moves)
    best_moves_indexes, best_moves_scores = get_k_best_moves(possible_next_moves, min(k, len(possible_next_moves)), positions, car_moves)
    

    #print(f"    possible moves: {possible_next_moves}")


    for idx_i, idx in enumerate(best_moves_indexes):
        possible_next_move = possible_next_moves[idx]
        next_car_moves = deepcopy(car_moves)


        if history is not None:
            history[possible_next_move] = {'base_score': best_moves_scores[idx_i], "next": dict()}


        if possible_next_move[0] in next_car_moves:
            next_car_moves[possible_next_move[0]].append(possible_next_move[1])
        else:
            next_car_moves[possible_next_move[0]] = [possible_next_move[1]]

        next = calculate_lookahead(
            grid,
            positions,
            size-1,
            k,
            next_car_moves,
            history[possible_next_move]["next"] if history is not None else None
        )
        #print(f"        lookahead next {next}")
        best_moves_scores[idx_i] += next[1]
        #best_moves_scores[idx_i] = next[1]
        if history:
            history[possible_next_move]["cum_score"] = best_moves_scores[idx_i]


    #print(f"    best move scores: {best_moves_scores}")

    
    if best_moves_scores:
        best_move_idx, best_move_score = max(enumerate(best_moves_scores), key=lambda i: i[1])
        new_move = possible_next_moves[best_moves_indexes[best_move_idx]]
        return new_move, best_move_score
    else:
        return None, 0

COUNT = 0

def increment():
    global COUNT
    COUNT = COUNT+1



# DO NOT CHANGE THE LINES BELLOW
# You can change the default values using the command line, example:
# $ NAME='arrumador' python3 client.py
loop = asyncio.get_event_loop()
SERVER = os.environ.get("SERVER", "localhost")
PORT = os.environ.get("PORT", "8000")
NAME = os.environ.get("NAME", getpass.getuser())
timedic={}

loop.run_until_complete(agent_loop(f"{SERVER}:{PORT}", NAME))
