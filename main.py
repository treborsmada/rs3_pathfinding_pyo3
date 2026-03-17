import rs3_pathfinding, time
from cluetrainer import path_to_cluetrainer

if __name__ == '__main__':
    st = time.time()
    floor = 0
    start = (3469, 3431)
    end = (3430, 3388)
    result = rs3_pathfinding.a_star(start, end, floor)
    et = time.time()
    print("Ticks:", result[1])
    print("Path:", result[0])
    print("Cluetrainer:", path_to_cluetrainer(result[0], floor))
    print(et-st)