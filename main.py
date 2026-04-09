import rs3_pathfinding, time
from cluetrainer import path_to_cluetrainer

if __name__ == '__main__':
    st = time.time()
    floor = 0
    start = (3421, 2949)
    end = (3435, 3129)
    result = rs3_pathfinding.a_star(start, end, floor, teleports = [(3423, 3016, 7), (3424, 3140, 5), (3480, 3099, 8), (3432, 2917, 8), (3373, 3080, 7)])
    et = time.time()
    print("Ticks:", result[1])
    print("Path:", result[0])
    print("Cluetrainer:", path_to_cluetrainer(result[0], floor))
    print(et-st)