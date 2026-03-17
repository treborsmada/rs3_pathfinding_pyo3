import json


def _move_type(s1, s2):
    # tuple: (pos_x, pos_y, direction, scd, sscd, ecd, secd, bdcd)
    if s1[7] == 0 and s2[7] == 17:
        return "dive"
    if s2[3] == 17:
        return "surge"
    if s2[4] == 17:
        return "surge"
    if s2[5] == 17:
        return "escape"
    if s2[6] == 17:
        return "escape"
    if s1[0] != s2[0] or s1[1] != s2[1]:
        return "walk"
    return "wait"


def path_to_cluetrainer(path, floor):
    """Convert an a_star path to cluetrainer.app import format (JSON string)."""
    moves = []
    run_waypoints = None

    for i in range(len(path) - 1):
        s1, s2 = path[i], path[i + 1]
        mt = _move_type(s1, s2)

        pos1 = {"x": int(s1[0]), "y": int(s1[1]), "level": floor}
        pos2 = {"x": int(s2[0]), "y": int(s2[1]), "level": floor}

        if mt == "walk":
            if run_waypoints is None:
                run_waypoints = [pos1, pos2]
            else:
                run_waypoints.append(pos2)
        elif mt == "wait":
            pass
        else:
            if run_waypoints is not None:
                moves.append({"type": "run", "waypoints": run_waypoints})
                run_waypoints = None
            moves.append({"type": "ability", "ability": mt, "from": pos1, "to": pos2})

    if run_waypoints is not None:
        moves.append({"type": "run", "waypoints": run_waypoints})

    return json.dumps(moves, separators=(',', ':'))
