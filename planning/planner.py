from __future__ import annotations

from collections.abc import Callable
###Para las métricas de complejidades 
import time
import sys

from planning.pddl import (
    Action,
    ActionSchema,
    Problem,
    State,
    Objects,
    get_all_groundings,
)
from planning.utils import Queue, PriorityQueue
from planning.heuristics import nullHeuristic


# ---------------------------------------------------------------------------
# Reference implementation – read and understand before coding the rest.
# ---------------------------------------------------------------------------


def tinyBaseSearch(problem: Problem) -> list[Action]:
    """
    Hardcoded plan for the tinyBase layout.
    The robot at (1,4) must: pick up supplies at (1,3), set them up at (1,2),
    pick up the patient at (1,1), bring them to (1,2), and execute Rescue.

    Useful to understand the Action object format and plan structure.
    """
    start_time = time.time()
    problem._expanded = 1
    robot = "robot"
    supplies = "supplies_0"
    patient = "patient_0"

    c14 = (1, 4)  # robot start
    c13 = (1, 3)  # supplies
    c12 = (1, 2)  # medical post
    c11 = (1, 1)  # patient

    plan = [
        Action(
            "Move(robot,(1,4),(1,3))",
            [("At", robot, c14), ("Adjacent", c14, c13), ("Free", c13)],
            [],
            [("At", robot, c13), ("Free", c14)],
            [("At", robot, c14), ("Free", c13)],
        ),
        Action(
            "PickUp(robot,supplies_0,(1,3))",
            [
                ("At", robot, c13),
                ("At", supplies, c13),
                ("HandsFree", robot),
                ("Pickable", supplies),
            ],
            [],
            [("Holding", robot, supplies)],
            [("At", supplies, c13), ("HandsFree", robot)],
        ),
        Action(
            "Move(robot,(1,3),(1,2))",
            [("At", robot, c13), ("Adjacent", c13, c12), ("Free", c12)],
            [],
            [("At", robot, c12), ("Free", c13)],
            [("At", robot, c13), ("Free", c12)],
        ),
        Action(
            "SetupSupplies(robot,supplies_0,(1,2))",
            [("At", robot, c12), ("MedicalPost", c12), ("Holding", robot, supplies)],
            [("SuppliesReady", c12)],
            [("SuppliesReady", c12), ("HandsFree", robot)],
            [("Holding", robot, supplies)],
        ),
        Action(
            "Move(robot,(1,2),(1,1))",
            [("At", robot, c12), ("Adjacent", c12, c11), ("Free", c11)],
            [],
            [("At", robot, c11), ("Free", c12)],
            [("At", robot, c12), ("Free", c11)],
        ),
        Action(
            "PickUp(robot,patient_0,(1,1))",
            [
                ("At", robot, c11),
                ("At", patient, c11),
                ("HandsFree", robot),
                ("Pickable", patient),
            ],
            [],
            [("Holding", robot, patient)],
            [("At", patient, c11), ("HandsFree", robot)],
        ),
        Action(
            "Move(robot,(1,1),(1,2))",
            [("At", robot, c11), ("Adjacent", c11, c12), ("Free", c12)],
            [],
            [("At", robot, c12), ("Free", c11)],
            [("At", robot, c11), ("Free", c12)],
        ),
        Action(
            "PutDown(robot,patient_0,(1,2))",
            [("At", robot, c12), ("Holding", robot, patient)],
            [],
            [("At", patient, c12), ("HandsFree", robot)],
            [("Holding", robot, patient)],
        ),
        Action(
            "Rescue(robot,patient_0,(1,2))",
            [
                ("At", robot, c12),
                ("At", patient, c12),
                ("MedicalPost", c12),
                ("SuppliesReady", c12),
            ],
            [],
            [("Rescued", patient)],
            [("At", patient, c12)],
        ),
    ]
    end_time = time.time()

    print("Tiny Base Search")
    print("Expanded states:", problem._expanded)
    print("Plan length:", len(plan))
    print("Execution time:", end_time - start_time, "seconds")
    return plan


# ---------------------------------------------------------------------------
# Punto 2 – Forward Planning
# ---------------------------------------------------------------------------


def forwardBFS(problem: Problem) -> list[Action]:
    """
    Forward BFS in state space.

    Explore states reachable from the initial state by applying actions,
    in breadth-first order, until a goal state is found.

    Returns a list of Action objects forming a valid plan, or [] if no plan exists.

    Tip: The state is a frozenset of fluents. Use problem.getSuccessors(state)
         to get (next_state, action, cost) triples. Track visited states to
         avoid revisiting the same state twice (graph search, not tree search).
    """
    ### Your code here ###
    start_state = problem.getStartState()
    if problem.isGoalState(start_state):
        return []

    goal_fluents = frozenset(problem.goal)

    queue = Queue()
    queue.push((start_state, []))
    
    visited = {start_state}

    while not queue.isEmpty():
        current_state, path = queue.pop()

        successors = problem.getSuccessors(current_state)

        
        prioritized_successors = sorted(
            successors,
            key=lambda item: (
                0 if item[1].name.startswith("Rescue") else
                1 if item[1].name.startswith("SetupSupplies") else
                2 if item[1].name.startswith("PickUp") else
                3,
                len(goal_fluents - item[0]),
            ),
        )

        for next_state, action, _ in prioritized_successors:
            if next_state not in visited:
                if problem.isGoalState(next_state):
                    print("Forward BFS")
                    print("Expanded states:", problem._expanded)
                    print("Plan length:", len(path + [action]))
                    return path + [action]
                
                visited.add(next_state)
                queue.push((next_state, path + [action]))

    return []

    ### End of your code ###


# ---------------------------------------------------------------------------
# Punto 3 – Backward Planning
# ---------------------------------------------------------------------------


def regress(goal_set: State, action: Action) -> State | None:
    """
    Compute the regression of goal_set through action.

    Given a goal description (set of fluents that must be true) and an action,
    return the new goal description that, if satisfied, guarantees the original
    goal is satisfied after executing action.

    REGRESS(g, a) = (g − ADD(a)) ∪ PRECOND_pos(a)
        IF:  ADD(a) ∩ g ≠ ∅   (action is relevant: contributes to the goal)
        AND: DEL(a) ∩ g = ∅   (action does not undo any goal fluent)
    Returns None if the action is not relevant or creates a contradiction.

    Tip: Use frozenset operations: intersection (&), difference (-), union (|).
         Check relevance first, then check for contradictions, then compute.
    """
    ### Your code here ###
    goal_set = frozenset(goal_set)

    add_effects = frozenset(action.add_list)
    del_effects = frozenset(action.del_list)
    preconditions = frozenset(action.precond_pos)

    if not (add_effects & goal_set):
        return None

    if del_effects & goal_set:
        return None

    regressed_goal = (goal_set - add_effects) | preconditions

    return frozenset(regressed_goal)
    ### End of your code ###



def backwardSearch(problem: Problem) -> list[Action]:
    """
    Backward search (regression search) from the goal.

    Start from the goal description and apply action regressions until
    the resulting goal is satisfied by the initial state.

    Returns a list of Action objects forming a valid plan (in forward order),
    or [] if no plan exists.

    Tip: The "state" in backward search is a frozenset of fluents that must
         be true (a partial goal description). The initial state is reached
         when all fluents in the current goal are satisfied by problem.initial_state.
         Only consider actions whose add_list has at least one unsatisfied goal fluent
         (relevant actions). Use regress() to compute the new subgoal.
         Skip subgoals that contain static predicates (MedicalPost, Adjacent,
         Pickable) that are false in the initial state — these are dead ends.
    """
    ### Your code here ###
    initial_state = frozenset(problem.initial_state)
    goal = frozenset(problem.goal)
    if goal.issubset(initial_state):
        return []

    all_actions = get_all_groundings(problem.domain, problem.objects)
    actions_by_fluent = {}
    for action in all_actions:
        for fluent in action.add_list:
            actions_by_fluent.setdefault(fluent, []).append(action)

    static_predicates = {"MedicalPost", "Adjacent", "Pickable", "Free"}
    
    queue = Queue()
    queue.push((goal, None, None)) 
    
    visited = [goal]
    LIMITE_EXPANSIONES = 100000

    while not queue.isEmpty():
        current_node = queue.pop()
        current_goal, action_taken, parent_node = current_node
        problem._expanded += 1

        # ==========================================================
        # CORTOCIRCUITO DE SEGURIDAD (Resource Bound)
        # ==========================================================
        if problem._expanded > LIMITE_EXPANSIONES:
            sys.__stdout__.write(f"\n[ALERTA] Límite de seguridad alcanzado ({LIMITE_EXPANSIONES} expansiones). Abortando búsqueda hacia atrás.\n")
            sys.__stdout__.flush()
            return []

        # ==========================================================
        # LATIDO DE CONSOLA (Heartbeat)
        # ==========================================================
        if problem._expanded % 2000 == 0:
            mensaje = f"[DEBUG] Explorando... Nodos expandidos: {problem._expanded} | Tamaño subobjetivo: {len(current_goal)}\n"
            sys.__stdout__.write(mensaje)
            sys.__stdout__.flush() 

        # ====================================================================
        # DETECCIÓN DE MUTEX (ESTADOS FÍSICAMENTE IMPOSIBLES) AVANZADA
        # ====================================================================
        locations = {}
        carrying = []
        robot_free = False
        impossible = False

        for fluent in current_goal:
            # 1. Un objeto no puede estar en dos lugares a la vez
            if fluent[0] == "At":
                obj, loc = fluent[1], fluent[2]
                if obj in locations and locations[obj] != loc:
                    impossible = True
                    break
                locations[obj] = loc
            
            # 2. Registrar qué está cargando el robot
            elif fluent[0] == "Carrying":
                carrying.append(fluent[2]) 
                
            # 3. Registrar si se requiere que el robot esté libre
            elif fluent[0] == "Free":
                robot_free = True

        # Evaluaciones de Mutex Lógico:
        if not impossible:
            # Regla A: El robot NO puede estar libre y cargando algo al mismo tiempo
            if robot_free and len(carrying) > 0:
                impossible = True
                
            # Regla B: El robot solo puede cargar UNA cosa a la vez 
            if len(carrying) > 1:
                impossible = True
                
            # Regla C: Un objeto NO puede estar siendo cargado y estar en el piso al mismo tiempo
            for obj in carrying:
                if obj in locations:
                    impossible = True
                    break

        if impossible:
            continue
        # ====================================================================

        relevant_actions = set()
        for fluent in current_goal:
            if fluent in actions_by_fluent:
                relevant_actions.update(actions_by_fluent[fluent])

        for action in relevant_actions:
            regressed = regress(current_goal, action)

            if regressed is None:
                continue

            if any(f[0] in static_predicates and f not in initial_state for f in regressed):
                continue

            if regressed.issubset(initial_state):
                plan = [action]
                curr = current_node
                while curr[2] is not None:
                    plan.append(curr[1])
                    curr = curr[2]
                return plan

            is_subsumed = False
            for v in visited:
                if v.issubset(regressed):
                    is_subsumed = True
                    break
            
            if not is_subsumed:
                visited.append(regressed)
                queue.push((regressed, action, current_node))

    return []
    ### End of your code ###




# ---------------------------------------------------------------------------
# Punto 4 – A* Planner
# ---------------------------------------------------------------------------

# Heuristic signature:  heuristic(state, goal, domain, objects) -> float
star_time = time.time()
Heuristic = Callable[[State, State, list[ActionSchema], Objects], float]


def aStarPlanner(
    problem: Problem,
    heuristic: Heuristic = nullHeuristic,
) -> list[Action]:
    """
    Forward A* search guided by a heuristic.

    Combines the real accumulated cost g(n) with the heuristic estimate h(n)
    to prioritize which state to expand next: f(n) = g(n) + h(n).

    Returns a list of Action objects forming a valid plan, or [] if no plan exists.

    Tip: The heuristic signature is heuristic(state, goal, domain, objects) → float.
         Use PriorityQueue with priority = g + h(next_state).
         Track the best g-cost seen for each state to avoid stale expansions.
    """
    ### Your code here ###
    start_state = problem.getStartState()

    if problem.isGoalState(start_state):
        return []

    frontier = PriorityQueue()

    start_h = heuristic(
        start_state,
        problem.goal,
        problem.domain,
        problem.objects,
    )

    frontier.push((start_state, [], 0), start_h)

    best_cost = {}
    best_cost[start_state] = 0

    while not frontier.isEmpty():

        current_state, path, g = frontier.pop()
        problem._expanded += 1

        if problem.isGoalState(current_state):
            print("A* Planner")
            print("Expanded states:", problem._expanded)
            print("Plan length:", len(path))
            print("Solution cost:", g)
            return path

        for next_state, action, cost in problem.getSuccessors(current_state):

            new_g = g + cost

            if (
                next_state not in best_cost
                or new_g < best_cost[next_state]
            ):

                best_cost[next_state] = new_g

                h = heuristic(
                    next_state,
                    problem.goal,
                    problem.domain,
                    problem.objects,
                )

                f = new_g + h

                frontier.push(
                    (next_state, path + [action], new_g),
                    f
                )

    return []
    ### End of your code ###


# Aliases used by the command-line argument parser
tinyBaseSearch = tinyBaseSearch
forwardBFS = forwardBFS
backwardSearch = backwardSearch
aStarPlanner = aStarPlanner
