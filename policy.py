"""Policy tree structure and evaluation for RS3 scan clue solving."""
import rs3_pathfinding
from __future__ import annotations
from dataclasses import dataclass
from scandata import scan_info
from collections import deque

SPOTS = scan_info['spots']
RADIUS = scan_info['range']
FLOOR = SPOTS[0]['level']
N = len(SPOTS)
TELEPORTS = scan_info.get('teleports', [])
_SECTION_PADDING = 30


@dataclass(frozen=True)
class AgentState:
    x: int
    y: int
    direction: int
    scd: int
    sscd: int
    ecd: int
    secd: int
    bdcd: int
    tick: int

    def successors(self):
        return [(_state_from_raw(raw, self.tick + cost), cost) for (raw, cost) in rs3_pathfinding.get_successors_preloaded(
            self.x, self.y, self.direction,
            self.scd, self.sscd, self.ecd, self.secd, self.bdcd,
            teleports=TELEPORTS,
        )]


LOBBY = AgentState(x=0, y=0, direction=0, scd=0, sscd=0, ecd=0, secd=0, bdcd=0, tick=0)


def scan_obs(pos1: (int, int), candidates_idx: frozenset[int], pos2: (int, int) = None) -> list[list[int]]:
    x1, y1 = pos1
    if pos2 is None:
        x2, y2 = x1, y1
    else:
        x2, y2 = pos2
    branches = [[], []]
    for candidate in candidates_idx:
        s = SPOTS[candidate]
        d1 = max(abs(x1 - s['x']), abs(y1 - s['y']))
        d2 = max(abs(x2 - s['x']), abs(y2 - s['y']))
        if d1 <= RADIUS or d2 <= RADIUS:
            branches.append([candidate])
        elif d1 <= 2 * RADIUS or d2 <= 2 * RADIUS:
            branches[1].append(candidate)
        else:
            branches[0].append(candidate)
    if len(branches) == 2 and (len(branches[0]) == 0 or len(branches[1]) == 0):
        return []
    return branches


@dataclass
class PolicyEdge:
    child: PolicyNode
    cost: int
    label: str = None


@dataclass
class PolicyNode:
    state: AgentState
    candidates: frozenset[int]
    outgoing: list[PolicyEdge] = None


def _state_from_raw(raw, tick) -> AgentState:
    return AgentState(x=raw[0], y=raw[1], direction=raw[2], scd=raw[3], sscd=raw[4], ecd=raw[5], secd=raw[6], bdcd=raw[7], tick=tick)

# evaluate and validate policy
def evaluate(root: PolicyNode) -> float:

    if root != LOBBY:
        raise Exception(f"Policy root is not LOBBY state.")
    if root.candidates != frozenset(range(N)):
        raise Exception(f"Policy root candidates must be all {N} spots.")

    if not root.outgoing:
        raise Exception(f"Policy ends early.")

    total_ticks = 0
    curr_tick = 0
    curr_edge = root.outgoing[0]
    curr_tick += curr_edge.cost
    curr_node = curr_edge.child

    if (curr_node.state.x, curr_node.state.y, curr_edge.cost) not in TELEPORTS:
        raise Exception(f"Must teleport from LOBBY state.")

    if curr_node.state != AgentState(curr_node.state.x, curr_node.state.y, 0, 0, 0, 0, 0, 0, curr_tick):
        raise Exception(f"Node at {curr_node.state} has incorrect cooldown or tick field.")

    queue = deque([])
    if curr_tick % 2 == 0:
        branches = scan_obs((curr_node.state.x, curr_node.state.y), curr_node.candidates)
        i = 0
        for branch in branches:
            if branch:
                branch_edge = curr_node.outgoing[i]
                if branch_edge.cost != 0:
                    raise Exception(f"Scan observation edge should have cost 0.")
                branch_node = branch_edge.child
                if branch_node.candidates != frozenset(branch):
                    raise Exception(f"Policy does not correctly branch at {curr_node.state}.")
                if branch_node.state != curr_node.state:
                    raise Exception(f"Policy state changes during branching step at {curr_node.state}.")
                i += 1
        if len(curr_node.outgoing) != i:
            raise Exception(f"Policy does not correctly branch at {curr_node.state}.")
        for node in [edge.child for edge in curr_node.outgoing]:
            queue.append(node)
    else:
        queue.append(curr_node)

    while queue:
        curr_node = queue.popleft()
        curr_tick = curr_node.state.tick

        if len(curr_node.candidates) == 1:
            if curr_node.outgoing is not None:
                raise Exception(f"Node at {curr_node.state} has one remaining goal and should not have child nodes.")
            goal_idx, = curr_node.candidates
            goal = SPOTS[goal_idx]
            total_ticks += curr_tick + rs3_pathfinding.a_star_preloaded(
                start=(curr_node.state.x, curr_node.state.y), end=(goal['x'], goal['y']),
                scd=curr_node.state.scd, sscd=curr_node.state.sscd, ecd=curr_node.state.ecd,
                secd=curr_node.state.secd, bdcd=curr_node.state.bdcd, teleports=TELEPORTS)
            continue

        tick_updated = False
        second_scan_pos = None
        while not tick_updated:
            if curr_node.outgoing is None:
                raise Exception(f"Node at {curr_node.state} has more than one candidate remaining but has no child nodes.")
            curr_edge = curr_node.outgoing[0]
            succs = curr_node.state.successors()
            curr_node = curr_edge.child
            curr_tick += curr_edge.cost
            if (curr_node.state, curr_edge.cost) not in succs:
                raise Exception(f"Node at {curr_node.state} is unreachable from previous state.")
            if curr_edge.cost == 0:
                second_scan_pos = (curr_node.state.x, curr_node.state.y)
            else:
                tick_updated = True

        if curr_tick % 2 == 0:
            branches = scan_obs((curr_node.state.x, curr_node.state.y), curr_node.candidates, second_scan_pos)
            i = 0
            for branch in branches:
                if branch:
                    branch_edge = curr_node.outgoing[i]
                    if branch_edge.cost != 0:
                        raise Exception(f"Scan observation edge should have cost 0.")
                    branch_node = branch_edge.child
                    if branch_node.candidates != frozenset(branch):
                        raise Exception(f"Policy does not correctly branch at {curr_node.state}.")
                    if branch_node.state != curr_node.state:
                        raise Exception(f"Policy state changes during branching step at {curr_node.state}.")
                    i += 1
            if len(curr_node.outgoing) != i:
                raise Exception(f"Policy does not correctly branch at {curr_node.state}.")
            for node in [edge.child for edge in curr_node.outgoing]:
                queue.append(node)
    return total_ticks/N


    def build_policy():
