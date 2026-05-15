from __future__ import annotations

from planning.pddl import Action, Problem, apply_action, is_applicable


# ---------------------------------------------------------------------------
# HTN Infrastructure
# ---------------------------------------------------------------------------


class HLA:
    """
    A High-Level Action (HLA) in HTN planning.

    An HLA is an abstract task that can be refined into sequences of
    more primitive actions (or other HLAs). Each refinement is a list
    of HLA or Action objects.

    name:        Human-readable name for display
    refinements: List of possible refinements, each a list of HLA/Action objects
    """

    def __init__(self, name: str, refinements: list[list] | None = None) -> None:
        self.name = name
        self.refinements = refinements or []

    def __repr__(self) -> str:
        return f"HLA({self.name})"


def is_primitive(action: Action | HLA) -> bool:
    """Return True if action is a primitive (grounded Action), False if it is an HLA."""
    return isinstance(action, Action)


def is_plan_primitive(plan: list[Action | HLA]) -> bool:
    """Return True if every step in the plan is a primitive action."""
    return all(is_primitive(step) for step in plan)


# ---------------------------------------------------------------------------
# Punto 5a – hierarchicalSearch
# ---------------------------------------------------------------------------


def hierarchicalSearch(problem: Problem, hlas: list[HLA]) -> list[Action]:
    """
    HTN planning via BFS over hierarchical plan refinements.

    Start with an initial plan containing a single top-level HLA.
    At each step, find the first non-primitive step in the plan and
    replace it with one of its refinements. Continue until the plan
    is fully primitive and achieves the goal when executed from the
    initial state.

    Returns a list of primitive Action objects, or [] if no plan found.

    Tip: The search space consists of (partial plan, current plan index) pairs.
         Use a Queue (BFS) to explore all refinement choices fairly.
         A plan is a solution when:
           1. It contains only primitive actions (is_plan_primitive), AND
           2. Executing it from the initial state reaches a goal state.
         To simulate execution, apply each action in order using apply_action().
    """
    ### Your code here ###
    initial_plan = list(hlas)
    queue = Queue()
    # Guardamos (plan, estado_actual) para simular correctamente
    queue.push((initial_plan, problem.initial_state))
    visited = set()
    expansions = 0

    while not queue.isEmpty():
        current_plan, current_state = queue.pop()
        expansions += 1

        if expansions <= 5:
            print(f"[DEBUG HTN] Expansión {expansions}, plan: {[s.name for s in current_plan]}")

        first_hla_idx = None
        for i, step in enumerate(current_plan):
            if not is_primitive(step):
                first_hla_idx = i
                break

        if first_hla_idx is None:
            state = current_state
            valid = True
            for action in current_plan:
                if is_applicable(state, action):
                    state = apply_action(state, action)
                else:
                    valid = False
                    break
            if valid and problem.isGoalState(state):
                return current_plan
            continue

        state_before_hla = current_state
        valid = True
        for i in range(first_hla_idx):
            step = current_plan[i]
            if is_primitive(step):
                if is_applicable(state_before_hla, step):
                    state_before_hla = apply_action(state_before_hla, step)
                else:
                    valid = False
                    break
        
        if not valid:
            continue

        hla = current_plan[first_hla_idx]

        for refinement in hla.refinements:
            new_plan = (
                current_plan[:first_hla_idx] 
                + refinement 
                + current_plan[first_hla_idx + 1:]
            )
            plan_key = tuple(step.name for step in new_plan)
            if plan_key not in visited:
                visited.add(plan_key)
                queue.push((new_plan, current_state))

    print(f"[DEBUG HTN] Total expansiones: {expansions}")
    return []
    ### End of your code ###


# ---------------------------------------------------------------------------
# Punto 5b – HLA Definitions
# ---------------------------------------------------------------------------


def build_htn_hierarchy(problem: Problem) -> list[HLA]:
    """
    Build HTN HLAs for the rescue domain.

    The hierarchy defines four HLA types:
      - Navigate(from, to):       Move the robot step by step from one cell to another
      - PrepareSupplies(s, m):    Collect supplies and set them up at the medical post
      - ExtractPatient(p, m):     Pick up the patient and bring them to the medical post
      - FullRescueMission(s,p,m): Complete one rescue: prepare supplies + extract + rescue

    Refinements are built from the ground state to generate concrete Action objects.

    Tip: Refinements for Navigate are all single-step Move sequences between
         adjacent cells. PrepareSupplies and ExtractPatient chain Navigate HLAs
         with primitive PickUp, SetupSupplies, PutDown, and Rescue actions.
    """
    ### Your code here ###

    ### End of your code ###
