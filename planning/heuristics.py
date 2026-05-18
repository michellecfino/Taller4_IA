from __future__ import annotations

from planning.pddl import (
    ActionSchema,
    State,
    Objects,
    get_all_groundings,
    get_applicable_actions,
)


def nullHeuristic(
    state: State,
    goal: State,
    domain: list[ActionSchema],
    objects: Objects,
) -> float:
    """Trivial heuristic — always returns 0 (equivalent to uniform-cost search)."""
    return 0


def ignorePreconditionsHeuristic(
    state: State,
    goal: State,
    domain: list[ActionSchema],
    objects: Objects,
) -> float:
    """
    Estimate the number of actions needed to satisfy all goal fluents,
    ignoring all action preconditions.
    """
    unsatisfied = frozenset(goal - state)

    if not unsatisfied:
        return 0

    remaining = set(unsatisfied)
    all_actions = get_all_groundings(domain, objects)
    steps = 0

    while remaining:
        best_cover = set()

        for action in all_actions:
            covered = remaining & set(action.add_list)
            if len(covered) > len(best_cover):
                best_cover = covered

        if not best_cover:
            return float("inf")

        remaining -= best_cover
        steps += 1

    return steps


def ignoreDeleteListsHeuristic(
    state: State,
    goal: State,
    domain: list[ActionSchema],
    objects: Objects,
) -> float:
    """
    Estimate the plan cost by solving a relaxed problem where no action
    has a delete list.
    """
    relaxed_state = set(state)
    count = 0

    while not goal.issubset(relaxed_state):
        applicable = get_applicable_actions(
            frozenset(relaxed_state),
            domain,
            objects,
        )

        best_action = None
        best_gain = set()

        for action in applicable:
            gain = (goal - relaxed_state) & set(action.add_list)
            if len(gain) > len(best_gain):
                best_gain = gain
                best_action = action

        if not best_gain or best_action is None:
            return float("inf")

        relaxed_state |= set(best_action.add_list)
        count += 1

    return count
