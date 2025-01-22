import rs3_pathfinding, time

if __name__ == '__main__':
    st = time.time()
    floor = 0
    start = (3469, 3431)
    end = (3430, 3389)
    result = rs3_pathfinding.a_star(start, end, floor)
    et = time.time()
    print(result)
    print(et-st)