import numpy as np
from itertools import islice
from aisdata import route 
from aisdata import ship

def intersects_in_time(route, other_ship):
    """ wheter any of other_ship's routes itersects in time with route
    Paramters:
        route the route
        other_ship the other ship
    """
    return any(map(lambda r: r.overlapping_in_time(route), other_ship.routes))

def ships_with_intersecting_routes(route, other_ships):
    """ creates a list of ships which have a intersecting route in time with route.
    Paramters:
        route: the route
        other_ships the other ships
    """
    return [s for s in other_ships if intersects_in_time(route, s)]

def state_to_vector(state):
    """ returns a numpy.array-representation of the fiven state
    Paramters:
        state the state to convert
    """
    return np.array([state.longitude, state.latitude])

def first_route_with_overlap(ship, route):
    """ return the first route of the given ships which intersects in time with route.
    Paramters:
        ship the ship
        route the route
    """
    for r in ship.routes:
        if r.overlapping_in_time(route):
            return r

def states_within(route, t_s, t_e):
    """ creates the states of the route which are after t_s and before t_e
    Parameters:
        route the routes
        t_s the earliest time to include
        t_e the latest time to include
    """
    i_begin = max([int((t_s - route.t_begin) / route.TIME_STEP), 0])
    i_end = min([int((t_e - route.t_begin)  / route.TIME_STEP), len(route.states)])
    return route.states[i_begin:i_end] 

def extract_training_examples_for_route(route, other_ships, n_ships, n_states):
    """ extracts sequences based on the given route and other ships.
    From other ships the ships are selected which have an in time overlapping route.
    From these ships the closest n_ships to route are choosen.
    A bunch of sequences are created for n_ships containing n_states states.
    """
    intersection_ships = ships_with_intersecting_routes(route, other_ships)
    ship_t_begin = lambda s: first_route_with_overlap(s, route).t_begin
    ship_t_end = lambda s: first_route_with_overlap(s, route).t_end
    distance_to_first_route_with_overlap = lambda s: first_route_with_overlap(s, route).distance_to_route(route)
    ships = list(islice(sorted(intersection_ships, key=distance_to_first_route_with_overlap), 0, n_ships))
 
    if not ships:
        return None

    t_s = ship_t_begin(max(ships, key=ship_t_begin))
    t_e = ship_t_end(min(ships, key=ship_t_end))
    n_st = (t_e - t_s) // route.TIME_STEP
    if  n_st < n_states:
        return None
    n_sequences = n_st - n_states
    m = np.zeros((n_ships, n_sequences, n_states, 2))
    y = np.zeros((n_sequences, n_states, 2))
    for i, ship in enumerate(ships):
        for seq in range(n_sequences):
            b = t_s + seq * route.TIME_STEP
            e = b + n_states * route.TIME_STEP
            ey = e + n_states * route.TIME_STEP
            ss = states_within(first_route_with_overlap(ship, route), b, e)
            ssy = states_within(first_route_with_overlap(ship, route), e, ey)
            for j, state in enumerate(ss):
              m[i][seq][j] = state_to_vector(state)

            if i == 0:
                for j, state in enumerate(ssy):
                  y[seq][j] = state_to_vector(state)
        
    return m, y

if __name__ == "__main__":
    from persistent.reader import Reader
    reader = Reader("/home/chris/BAFPracticalCourse-Code/test/examples/persistent/")
    scope = reader.read()
    for ship in list(scope.ships):
      other_ships = list(scope.ships)
      other_ships.remove(ship)
      for route in list(ship.routes):
        examples = extract_training_examples_for_route(route, other_ships, n_ships=2, n_states=3)

