from __future__ import annotations
from planning.pddl import Action, Problem, apply_action, is_applicable, get_all_groundings
from planning.utils import Queue
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
    problem._expanded = 0
    queue = Queue()
    # Guardamos (plan, estado_actual) para simular correctamente
    queue.push((initial_plan, problem.initial_state))
    visited = set()
    expansions = 0

    while not queue.isEmpty():
        current_plan, current_state = queue.pop()
        expansions += 1
        problem._expanded = expansions

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

        # Refinar el primer HLA no primitivo
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
    initial_state = problem.initial_state
    objects = problem.objects

    cells = objects["cells"]
    robots = objects["robots"]
    supplies_list = objects["supplies"]
    patients_list = objects["patients"]
    medical_posts = objects["medical_posts"]

    robot = robots[0]
    all_actions = get_all_groundings(problem.domain, problem.objects)

    # ------------------------------------------------------------------
    # Helpers para encontrar acciones primitivas
    # ------------------------------------------------------------------
    def get_move(from_cell, to_cell):
        name = f"Move({robot}, {from_cell}, {to_cell})"
        for a in all_actions:
            if a.name == name:
                return a
        return None

    def get_pickup(obj, loc):
        name = f"PickUp({robot}, {obj}, {loc})"
        for a in all_actions:
            if a.name == name:
                return a
        return None

    def get_putdown(obj, loc):
        name = f"PutDown({robot}, {obj}, {loc})"
        for a in all_actions:
            if a.name == name:
                return a
        return None

    def get_setup(supplies, loc):
        name = f"SetupSupplies({robot}, {supplies}, {loc})"
        for a in all_actions:
            if a.name == name:
                return a
        return None

    def get_rescue(patient, loc):
        name = f"Rescue({robot}, {patient}, {loc})"
        for a in all_actions:
            if a.name == name:
                return a
        return None

    # ------------------------------------------------------------------
    # Construir grafo de adyacencia desde el estado inicial
    # ------------------------------------------------------------------
    adjacency = {}
    for fluent in initial_state:
        if fluent[0] == "Adjacent":
            c1, c2 = fluent[1], fluent[2]
            adjacency.setdefault(c1, []).append(c2)

    # ------------------------------------------------------------------
    # BFS para encontrar ruta entre dos celdas
    # ------------------------------------------------------------------
    def find_path(start, goal):
        if start == goal:
            return []
        from collections import deque
        queue = deque([(start, [start])])
        visited = {start}
        while queue:
            current, path = queue.popleft()
            for neighbor in adjacency.get(current, []):
                if neighbor == goal:
                    return path[1:] + [neighbor]  # celdas a visitar
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    # ------------------------------------------------------------------
    # Construir secuencia de Move primitivos entre dos celdas
    # ------------------------------------------------------------------
    def make_move_sequence(start, goal):
        path = find_path(start, goal)
        if path is None:
            return None
        moves = []
        current = start
        for next_cell in path:
            move = get_move(current, next_cell)
            if move is None:
                return None
            moves.append(move)
            current = next_cell
        return moves

    # ------------------------------------------------------------------
    # Encontrar posiciones iniciales
    # ------------------------------------------------------------------
    def find_location(obj):
        for fluent in initial_state:
            if fluent[0] == "At" and fluent[1] == obj:
                return fluent[2]
        return None

    robot_loc = find_location(robot)

    # ------------------------------------------------------------------
    # Navigate HLA: refinamiento con secuencia de Moves
    # ------------------------------------------------------------------
    navigate_hlas = {}
    for c1 in cells:
        for c2 in cells:
            if c1 == c2:
                continue
            moves = make_move_sequence(c1, c2)
            if moves:
                hla = HLA(f"Navigate({c1},{c2})", refinements=[moves])
                navigate_hlas[(c1, c2)] = hla

    # ------------------------------------------------------------------
    # PrepareSupplies: robot_loc → supplies_loc → post
    # ------------------------------------------------------------------
    prepare_hlas = []
    for supplies in supplies_list:
        for post in medical_posts:
            supplies_loc = find_location(supplies)
            if supplies_loc is None:
                continue

            pickup = get_pickup(supplies, supplies_loc)
            setup = get_setup(supplies, post)
            if pickup is None or setup is None:
                continue

            refinement = []

            # Mover robot a suministros
            if robot_loc != supplies_loc:
                nav1 = navigate_hlas.get((robot_loc, supplies_loc))
                if nav1 is None:
                    continue
                refinement.append(nav1)

            refinement.append(pickup)

            # Mover de suministros al post
            if supplies_loc != post:
                nav2 = navigate_hlas.get((supplies_loc, post))
                if nav2 is None:
                    continue
                refinement.append(nav2)

            refinement.append(setup)

            hla = HLA(f"PrepareSupplies({supplies},{post})", refinements=[refinement])
            prepare_hlas.append(hla)

    # ------------------------------------------------------------------
    # ExtractPatient: post → patient_loc → post
    # (robot está en el post después de PrepareSupplies)
    # ------------------------------------------------------------------
    extract_hlas = []
    for patient in patients_list:
        for post in medical_posts:
            patient_loc = find_location(patient)
            if patient_loc is None:
                continue

            pickup = get_pickup(patient, patient_loc)
            putdown = get_putdown(patient, post)
            rescue = get_rescue(patient, post)
            if pickup is None or putdown is None or rescue is None:
                continue

            refinement = []

            # Robot viene del post (donde quedó después de PrepareSupplies)
            if post != patient_loc:
                nav1 = navigate_hlas.get((post, patient_loc))
                if nav1 is None:
                    continue
                refinement.append(nav1)

            refinement.append(pickup)

            if patient_loc != post:
                nav2 = navigate_hlas.get((patient_loc, post))
                if nav2 is None:
                    continue
                refinement.append(nav2)

            refinement.append(putdown)
            refinement.append(rescue)

            hla = HLA(f"ExtractPatient({patient},{post})", refinements=[refinement])
            extract_hlas.append(hla)

    # ------------------------------------------------------------------
    # FullRescueMission por cada par (supplies, patient, post)
    # y AllRescueMissions encadenando todas secuencialmente
    # ------------------------------------------------------------------
    
    # Emparejar supplies con patients en orden
    missions = []
    used_supplies = []
    used_patients = []
    
    # Simular estado para construir misiones secuenciales
    sim_state = dict()  # posición del robot después de cada misión
    current_robot_loc = robot_loc

    for i, (supplies, patient) in enumerate(zip(supplies_list, patients_list)):
        for post in medical_posts:
            supplies_loc = find_location(supplies)
            patient_loc = find_location(patient)

            if supplies_loc is None or patient_loc is None:
                continue

            pickup_s = get_pickup(supplies, supplies_loc)
            setup = get_setup(supplies, post)
            pickup_p = get_pickup(patient, patient_loc)
            putdown = get_putdown(patient, post)
            rescue = get_rescue(patient, post)

            if None in (pickup_s, setup, pickup_p, putdown, rescue):
                continue

            # PrepareSupplies desde current_robot_loc
            prep_refinement = []
            if current_robot_loc != supplies_loc:
                nav = navigate_hlas.get((current_robot_loc, supplies_loc))
                if nav is None:
                    continue
                prep_refinement.append(nav)
            prep_refinement.append(pickup_s)
            if supplies_loc != post:
                nav = navigate_hlas.get((supplies_loc, post))
                if nav is None:
                    continue
                prep_refinement.append(nav)
            prep_refinement.append(setup)

            prep_hla = HLA(f"PrepareSupplies({supplies},{post})", refinements=[prep_refinement])

            # ExtractPatient desde post (donde quedó el robot tras PrepareSupplies)
            ext_refinement = []
            if post != patient_loc:
                nav = navigate_hlas.get((post, patient_loc))
                if nav is None:
                    continue
                ext_refinement.append(nav)
            ext_refinement.append(pickup_p)
            if patient_loc != post:
                nav = navigate_hlas.get((patient_loc, post))
                if nav is None:
                    continue
                ext_refinement.append(nav)
            ext_refinement.append(putdown)
            ext_refinement.append(rescue)

            ext_hla = HLA(f"ExtractPatient({patient},{post})", refinements=[ext_refinement])

            mission_hla = HLA(
                f"FullRescueMission({supplies},{patient},{post})",
                refinements=[[prep_hla, ext_hla]]
            )
            missions.append(mission_hla)

            # Actualizar posición del robot para la siguiente misión
            # Después de ExtractPatient el robot queda en el post
            current_robot_loc = post

    if not missions:
        return []
    elif len(missions) == 1:
        return missions
    else:
        all_missions_hla = HLA(
            "AllRescueMissions",
            refinements=[missions]
        )
        return [all_missions_hla]    ### End of your code ###
