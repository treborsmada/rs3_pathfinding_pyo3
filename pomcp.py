import math
import random
from dataclasses import dataclass, field

import rs3_pathfinding
from scandata import scan_info

SPOTS = scan_info['spots']
RADIUS = scan_info['range']
FLOOR = SPOTS[0]['level']
N = len(SPOTS)
TELEPORTS = scan_info.get('teleports', [])

# Padding added to the scan area bounding box when preloading the map section.
# Large enough to cover movement from any reasonable starting position.
_SECTION_PADDING = 30


def preload(extra_positions=None):
    """
    Load and cache the MapSection covering all scan spots.
    Call this once before planning. extra_positions is a list of (x, y) tuples
    (e.g. starting positions) to include in the bounding box.
    Teleport destinations from scan_info are always included.
    """
    xs = [s['x'] for s in SPOTS] + [t[0] for t in TELEPORTS]
    ys = [s['y'] for s in SPOTS] + [t[1] for t in TELEPORTS]
    if extra_positions:
        xs += [p[0] for p in extra_positions]
        ys += [p[1] for p in extra_positions]
    x_min = max(0, min(xs) - _SECTION_PADDING)
    x_max = max(xs) + _SECTION_PADDING
    y_min = max(0, min(ys) - _SECTION_PADDING)
    y_max = max(ys) + _SECTION_PADDING
    rs3_pathfinding.preload_section(x_min, x_max, y_min, y_max, FLOOR)


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------

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
    tick: int  # absolute game tick, determines scan parity


_LOBBY = (0, 0)  # dummy start position outside the RS3 map; only teleports are valid from here


def _state_key(s: AgentState) -> tuple:
    return (s.x, s.y, s.direction, s.scd, s.sscd, s.ecd, s.secd, s.bdcd)


def _get_successors(state: AgentState) -> list:
    """Return list of (AgentState, cost) for all valid actions from state."""
    if (state.x, state.y) == _LOBBY:
        # Lobby state: only teleport actions are available.
        # Implemented in Python to avoid calling Rust with an out-of-map position.
        return [
            (AgentState(x=t[0], y=t[1], direction=state.direction,
                        scd=max(0, state.scd - t[2]),
                        sscd=max(0, state.sscd - t[2]),
                        ecd=max(0, state.ecd - t[2]),
                        secd=max(0, state.secd - t[2]),
                        bdcd=max(0, state.bdcd - t[2]),
                        tick=state.tick + t[2]),
             t[2])
            for t in TELEPORTS
        ]
    raw = rs3_pathfinding.get_successors_preloaded(
        state.x, state.y, state.direction,
        state.scd, state.sscd, state.ecd, state.secd, state.bdcd,
        teleports=TELEPORTS,
    )
    return [
        (AgentState(x=s[0], y=s[1], direction=s[2],
                    scd=s[3], sscd=s[4], ecd=s[5], secd=s[6], bdcd=s[7],
                    tick=state.tick + cost),
         cost)
        for s, cost in raw
    ]


def _get_scan_pos(prev_tick: int, next_state: AgentState, cost: int, ability_pos):
    """
    Returns the (x, y) position to scan at if a scan fires during this action, else None.

    Scans fire on even ticks. For a 1-tick walk preceded by abilities in the same
    tick, the scan position is the last ability destination (ability_pos), not the
    walk destination. For multi-tick actions (teleports) the scan position is the
    destination. 0-cost actions never advance the tick so never trigger a scan.
    """
    if cost == 0:
        return None
    new_tick = prev_tick + cost
    if not any(t % 2 == 0 for t in range(prev_tick + 1, new_tick + 1)):
        return None
    if cost == 1 and ability_pos is not None:
        return ability_pos
    return (next_state.x, next_state.y)


# ---------------------------------------------------------------------------
# POMCP tree nodes
# ---------------------------------------------------------------------------

@dataclass
class ActionNode:
    visit_count: int = 0
    total_return: float = 0.0
    obs_children: dict = field(default_factory=dict)  # obs (int | None) -> BeliefNode

    @property
    def value(self):
        return self.total_return / self.visit_count if self.visit_count > 0 else float('inf')


@dataclass
class BeliefNode:
    known_state: AgentState
    particles: list          # spot indices consistent with history; may have duplicates
    visit_count: int = 0
    action_nodes: dict = field(default_factory=dict)  # state_key tuple -> ActionNode


# ---------------------------------------------------------------------------
# Core model functions
# ---------------------------------------------------------------------------

def _obs(x, y, spot_idx):
    """Scan observation at position (x,y) when hidden target is spot_idx."""
    s = SPOTS[spot_idx]
    d = max(abs(x - s['x']), abs(y - s['y']))
    if d <= RADIUS:
        return 2
    if d <= 2 * RADIUS:
        return 1
    return 0


def _astar(state: AgentState, tx, ty):
    """A* using the cached section. Returns (path, ticks)."""
    return rs3_pathfinding.a_star_preloaded(
        (state.x, state.y), (tx, ty),
        state.direction, state.scd, state.sscd,
        state.ecd, state.secd, state.bdcd,
        teleports=TELEPORTS,
    )


# ---------------------------------------------------------------------------
# POMCP
# ---------------------------------------------------------------------------

class POMCP:
    def __init__(self, c=80.0, budget=200, max_depth=30, gamma=1.0):
        """
        c         : UCB exploration constant.
        budget    : simulations per planning call.
        max_depth : max individual game actions per simulation.
        gamma     : discount factor.
        """
        self.c = c
        self.budget = budget
        self.max_depth = max_depth
        self.gamma = gamma

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def _rollout(self, hidden: int, state: AgentState, candidates: frozenset,
                 depth: int, ability_pos=None) -> float:
        if len(candidates) == 1:
            target = SPOTS[next(iter(candidates))]
            try:
                _, cost = _astar(state, target['x'], target['y'])
            except Exception:
                cost = 999
            return -cost

        if depth == 0:
            cost = rs3_pathfinding.heuristic_cost(
                state.x, state.y, SPOTS[hidden]['x'], SPOTS[hidden]['y'],
                state.scd, state.sscd, state.ecd, state.secd, state.bdcd,
            )
            return -cost

        actions = _get_successors(state)
        if not actions:
            return -999

        next_state, cost = random.choice(actions)
        new_ability_pos = (next_state.x, next_state.y) if cost == 0 else None

        scan_pos = _get_scan_pos(state.tick, next_state, cost, ability_pos)
        if scan_pos is not None:
            obs = _obs(scan_pos[0], scan_pos[1], hidden)
            new_candidates = frozenset(h for h in candidates if _obs(scan_pos[0], scan_pos[1], h) == obs)
            if not new_candidates:
                new_candidates = candidates
        else:
            new_candidates = candidates

        return -cost + self.gamma * self._rollout(hidden, next_state, new_candidates, depth - 1, new_ability_pos)

    # ------------------------------------------------------------------
    # MCTS simulate
    # ------------------------------------------------------------------

    def _simulate(self, hidden: int, state: AgentState, node: BeliefNode,
                  candidates: frozenset, depth: int, ability_pos=None) -> float:
        if len(candidates) == 1:
            target = SPOTS[next(iter(candidates))]
            try:
                _, cost = _astar(state, target['x'], target['y'])
            except Exception:
                cost = 999
            return -cost

        if depth == 0:
            cost = rs3_pathfinding.heuristic_cost(
                state.x, state.y, SPOTS[hidden]['x'], SPOTS[hidden]['y'],
                state.scd, state.sscd, state.ecd, state.secd, state.bdcd,
            )
            return -cost

        actions = _get_successors(state)
        if not actions:
            return -999

        for ns, _ in actions:
            k = _state_key(ns)
            if k not in node.action_nodes:
                node.action_nodes[k] = ActionNode()

        action_key = self._ucb_select(node, [_state_key(ns) for ns, _ in actions])
        next_state, cost = next((ns, c) for ns, c in actions if _state_key(ns) == action_key)

        new_ability_pos = (next_state.x, next_state.y) if cost == 0 else None

        scan_pos = _get_scan_pos(state.tick, next_state, cost, ability_pos)
        if scan_pos is not None:
            obs = _obs(scan_pos[0], scan_pos[1], hidden)
            new_candidates = frozenset(h for h in candidates if _obs(scan_pos[0], scan_pos[1], h) == obs)
            if not new_candidates:
                new_candidates = candidates
        else:
            obs = None  # no scan this action
            new_candidates = candidates

        an = node.action_nodes[action_key]
        if obs not in an.obs_children:
            child = BeliefNode(known_state=next_state, particles=[hidden])
            an.obs_children[obs] = child
            R = -cost + self.gamma * self._rollout(hidden, next_state, new_candidates, depth - 1, new_ability_pos)
        else:
            child = an.obs_children[obs]
            child.particles.append(hidden)
            child.known_state = next_state
            R = -cost + self.gamma * self._simulate(hidden, next_state, child, new_candidates, depth - 1, new_ability_pos)

        an.visit_count += 1
        an.total_return += R
        node.visit_count += 1
        return R

    def _ucb_select(self, node: BeliefNode, action_keys: list) -> tuple:
        unvisited = [k for k in action_keys if node.action_nodes[k].visit_count == 0]
        if unvisited:
            return random.choice(unvisited)
        log_n = math.log(node.visit_count)
        return max(
            action_keys,
            key=lambda k: (
                node.action_nodes[k].value
                + self.c * math.sqrt(log_n / node.action_nodes[k].visit_count)
            ),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(self, state: AgentState, candidates: list, ability_pos=None) -> tuple:
        """
        Run POMCP and return (next_state, cost) for the best single action to take.
        ability_pos: last 0-cost ability destination used before this plan call, if any.
        """
        cand_set = frozenset(candidates)
        root = BeliefNode(known_state=state, particles=list(candidates))

        for _ in range(self.budget):
            hidden = random.choice(root.particles)
            self._simulate(hidden, state, root, cand_set, depth=self.max_depth,
                           ability_pos=ability_pos)

        actions = _get_successors(state)
        visited = [(ns, c) for ns, c in actions
                   if _state_key(ns) in root.action_nodes
                   and root.action_nodes[_state_key(ns)].visit_count > 0]
        if not visited:
            return random.choice(actions)
        return max(visited, key=lambda nc: root.action_nodes[_state_key(nc[0])].value)

    def update_belief(self, candidates: list, x: int, y: int, obs: int) -> list:
        """Filter candidates to those consistent with scan result obs at (x, y)."""
        return [h for h in candidates if _obs(x, y, h) == obs]

    def solve(self, start_direction: int = 0,
              start_scd: int = 0, start_sscd: int = 0,
              start_ecd: int = 0, start_secd: int = 0, start_bdcd: int = 0,
              true_target: int = None, verbose: bool = True):
        """
        Run a full solve episode one game action at a time.
        Starts from the lobby state: the first action must be one of the teleports.
        Scans fire on even ticks. After abilities followed by a walk, the scan
        position is the last ability destination, not the walk destination.
        Returns total ticks taken.
        """
        state = AgentState(
            x=_LOBBY[0], y=_LOBBY[1], direction=start_direction,
            scd=start_scd, sscd=start_sscd,
            ecd=start_ecd, secd=start_secd, bdcd=start_bdcd,
            tick=0,
        )
        candidates = list(range(N))
        total_ticks = 0
        step = 0
        ability_pos = None  # last 0-cost ability destination in current tick

        while len(candidates) > 1:
            next_state, cost = self.plan(state, candidates, ability_pos=ability_pos)
            total_ticks += cost

            scan_pos = _get_scan_pos(state.tick, next_state, cost, ability_pos)

            # Update ability tracking: reset on any tick-advancing action
            ability_pos = (next_state.x, next_state.y) if cost == 0 else None

            state = next_state
            step += 1

            if scan_pos is not None:
                if true_target is not None:
                    obs = _obs(scan_pos[0], scan_pos[1], true_target)
                else:
                    obs = int(input(f"  Scan at {scan_pos}? [0=far, 1=mid, 2=close]: "))
                candidates = self.update_belief(candidates, scan_pos[0], scan_pos[1], obs)
                if verbose:
                    print(f"step={step:3d} pos=({state.x},{state.y}) scan={scan_pos} cost={cost} obs={obs} candidates={len(candidates)}")
            else:
                if verbose:
                    print(f"step={step:3d} pos=({state.x},{state.y}) cost={cost} (no scan) candidates={len(candidates)}")

        target = SPOTS[candidates[0]]
        _, ticks = _astar(state, target['x'], target['y'])
        total_ticks += ticks
        if verbose:
            print(f"Target identified: {target}")
            print(f"Total ticks: {total_ticks}")
        return total_ticks


# ---------------------------------------------------------------------------
# Fixed policy tree
# ---------------------------------------------------------------------------

@dataclass
class PolicyNode:
    candidates: frozenset
    actions: list        # [(next_state, cost), ...] — deterministic steps until next scan
    scan_pos: tuple      # (x, y) where scan fires; None if terminal
    children: dict       # obs (int) -> PolicyNode; empty if terminal


def _build_node(pomcp: POMCP, state: AgentState, candidates: frozenset,
                ability_pos, memo: dict) -> PolicyNode:
    key = (candidates, _state_key(state), ability_pos)
    if key in memo:
        return memo[key]

    if len(candidates) == 1:
        node = PolicyNode(candidates=candidates, actions=[], scan_pos=None, children={})
        memo[key] = node
        return node

    actions = []
    cur = state
    cur_ap = ability_pos

    for _ in range(500):
        ns, cost = pomcp.plan(cur, list(candidates), ability_pos=cur_ap)
        actions.append((ns, cost))
        sp = _get_scan_pos(cur.tick, ns, cost, cur_ap)
        cur_ap = (ns.x, ns.y) if cost == 0 else None
        cur = ns

        if sp is not None:
            children = {}
            for obs in {_obs(sp[0], sp[1], h) for h in candidates}:
                nc = frozenset(h for h in candidates if _obs(sp[0], sp[1], h) == obs)
                children[obs] = _build_node(pomcp, cur, nc, cur_ap, memo)
            node = PolicyNode(candidates=candidates, actions=actions, scan_pos=sp, children=children)
            memo[key] = node
            return node

    # Safety fallback: no scan found within limit
    node = PolicyNode(candidates=candidates, actions=actions, scan_pos=None, children={})
    memo[key] = node
    return node


def build_policy_tree(pomcp: POMCP) -> PolicyNode:
    """
    Build a deterministic policy tree from the lobby start state.
    At each belief node POMCP is queried to select actions until a scan fires,
    then the tree branches on the scan observation (0/1/2).
    Returns the root PolicyNode.
    """
    preload()
    state = AgentState(x=_LOBBY[0], y=_LOBBY[1], direction=0,
                       scd=0, sscd=0, ecd=0, secd=0, bdcd=0, tick=0)
    root = _build_node(pomcp, state, frozenset(range(N)), ability_pos=None, memo={})
    return root


def follow_policy(root: PolicyNode, true_target: int) -> int:
    """
    Follow a pre-built policy tree for a given true target.
    Returns total ticks including the final walk to the identified target.
    """
    total_ticks = 0
    cur_state = AgentState(x=_LOBBY[0], y=_LOBBY[1], direction=0,
                           scd=0, sscd=0, ecd=0, secd=0, bdcd=0, tick=0)
    node = root

    while True:
        for ns, cost in node.actions:
            total_ticks += cost
            cur_state = ns

        if len(node.candidates) == 1:
            target = SPOTS[next(iter(node.candidates))]
            try:
                _, ticks = _astar(cur_state, target['x'], target['y'])
            except Exception:
                ticks = 999
            total_ticks += ticks
            return total_ticks

        if not node.children or node.scan_pos is None:
            return total_ticks  # incomplete tree

        obs = _obs(node.scan_pos[0], node.scan_pos[1], true_target)
        if obs not in node.children:
            return total_ticks
        node = node.children[obs]


def evaluate_policy(root: PolicyNode) -> tuple:
    """
    Score a pre-built policy tree against every spot.
    Returns (ticks_per_spot dict, avg_ticks).
    """
    ticks_per_spot = {}
    for target in range(N):
        ticks_per_spot[target] = follow_policy(root, target)
        print(f"spot {target:3d}: {ticks_per_spot[target]} ticks")
    avg = sum(ticks_per_spot.values()) / N
    print(f"\nOverall average: {avg:.2f} ticks over {N} spots")
    return ticks_per_spot, avg


def evaluate(pomcp: POMCP, n_trials: int = 1) -> tuple:
    """
    Run the POMCP policy against every spot and return (ticks_per_spot, avg_ticks).
    ticks_per_spot: dict mapping spot index -> list of ticks (length n_trials).
    avg_ticks: mean ticks across all spots and trials.
    """
    preload()
    ticks_per_spot = {}
    for target in range(N):
        ticks_per_spot[target] = []
        for trial in range(n_trials):
            t = pomcp.solve(true_target=target, verbose=False)
            ticks_per_spot[target].append(t)
        avg_spot = sum(ticks_per_spot[target]) / n_trials
        print(f"spot {target:3d}: avg={avg_spot:.1f} ticks  trials={ticks_per_spot[target]}")
    all_ticks = [t for ts in ticks_per_spot.values() for t in ts]
    avg = sum(all_ticks) / len(all_ticks)
    print(f"\nOverall average: {avg:.2f} ticks over {N} spots × {n_trials} trial(s)")
    return ticks_per_spot, avg
