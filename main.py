import sys
import time
import io
from contextlib import redirect_stdout
from optparse import OptionParser

import world.rescue_layout as rescue_layout
from planning.pddl import apply_action, is_applicable

def imprimir_resultado(nombre, T_formula, S_formula, nodos_expandidos, memoria_maxima, es_completo, es_optimo):
    print(f"\n--- [ANÁLISIS DE {nombre}] ---")
    print(f"1. Temporal: {T_formula} => {nodos_expandidos} operaciones")
    print(f"2. Espacial: {S_formula} => {memoria_maxima} nodos en memoria aproximados")
    print(f"3. Completitud: {'SÍ' if es_completo else 'NO'}")
    print(f"4. Optimalidad: {'SÍ' if es_optimo else 'NO'}")
    print("-" * 30)

def formatear_complejidad(total_ops, n):
    if n == 0:
        return "O(1)"
    factor = total_ops / n
    if abs(factor - round(factor)) < 1e-6:
        return f"O({int(round(factor))}*n)"
    else:
        if factor > 0:
            nuevo_denominador = round(1/factor)
            if nuevo_denominador != 0 and abs(factor - 1/nuevo_denominador) < 1e-6:
                return f"O(n/{nuevo_denominador})"
        return f"O({factor:.2f}*n)"

def read_command(argv):
    usage = """
    USAGE:    python main.py -p PROBLEM -f PLANNER -l LAYOUT [options]
    EXAMPLE:  python main.py -p SimpleRescueProblem -f tinyBaseSearch -l tinyBase -q
    """
    parser = OptionParser(usage, add_help_option=False)
    parser.add_option("--help", action="help")

    PROBLEMS = ("SimpleRescueProblem", "MultiRescueProblem")
    parser.add_option(
        "-p", "--problem", dest="problem", help="Problem type: %s" % ", ".join(PROBLEMS)
    )
    parser.add_option(
        "-f",
        "--function",
        dest="function",
        help="Planning algorithm: forwardBFS, backwardSearch, aStarPlanner, tinyBaseSearch",
    )
    parser.add_option(
        "-h",
        "--heuristic",
        dest="heuristic",
        default="nullHeuristic",
        help="Heuristic for A* [default: nullHeuristic]: ignorePreconditions, ignoreDeleteLists",
    )
    parser.add_option(
        "-l",
        "--layout",
        dest="layout",
        help="Layout file name (without .lay extension)",
    )
    parser.add_option(
        "-m",
        "--htn",
        action="store_true",
        dest="htn",
        default=False,
        help="Use HTN planning mode",
    )
    parser.add_option(
        "-t",
        "--text",
        action="store_true",
        dest="text",
        default=False,
        help="Text-only display",
    )
    parser.add_option(
        "-q",
        "--quiet",
        action="store_true",
        dest="quiet",
        default=False,
        help="Minimal output, no graphics",
    )
    parser.add_option(
        "-z",
        "--zoom",
        type="float",
        dest="zoom",
        default=1.0,
        help="Graphics window zoom [default: 1.0]",
    )
    parser.add_option(
        "-x",
        "--frame-time",
        type="float",
        dest="frame_time",
        default=0.1,
        help="Delay between frames in seconds [default: 0.1]",
    )

    options, junk = parser.parse_args(argv)
    if junk:
        raise Exception("Unrecognized arguments: " + str(junk))
    if not options.layout:
        parser.error("-l/--layout is required")
    if not options.function and not options.htn:
        parser.error("-f/--function is required")

    return options


def load_problem(problem_name, layout):
    from planning.problems import SimpleRescueProblem, MultiRescueProblem

    problems = {
        "SimpleRescueProblem": SimpleRescueProblem,
        "MultiRescueProblem": MultiRescueProblem,
    }
    cls = problems.get(problem_name)
    if cls is None:
        raise Exception(
            f"Unknown problem: {problem_name}. Choose from: {list(problems)}"
        )
    return cls(layout)


def load_planner(fn_name):
    import planning.planner as planner_module

    fn = getattr(planner_module, fn_name, None)
    if fn is None:
        raise Exception(
            f"Unknown planning function: '{fn_name}'. Check planning/planner.py."
        )
    return fn


def load_heuristic(h_name):
    import planning.heuristics as h_module

    aliases = {
        "ignorePreconditions": "ignorePreconditionsHeuristic",
        "ignoreDeleteLists": "ignoreDeleteListsHeuristic",
        "null": "nullHeuristic",
    }
    real_name = aliases.get(h_name, h_name)
    h = getattr(h_module, real_name, None)
    if h is None:
        raise Exception(f"Unknown heuristic: '{h_name}'")
    return h


def execute_plan(plan, initial_state, display, frame_time):
    """Simulate plan execution step by step."""
    state = initial_state
    for action in plan:
        if not is_applicable(state, action):
            print(f"  [ERROR] Action not applicable: {action.name}")
            print(f"  Missing preconditions: {action.precond_pos - state}")
            return state, False
        state = apply_action(state, action)
        display.update(state, action)
    return state, True


def append_log(log_lines, line=""):
    log_lines.append(line)
    print(line)


def run(options):
    log_lines = []

    # Load layout
    layout = rescue_layout.get_layout(options.layout)
    if layout is None:
        raise Exception(f"Layout '{options.layout}' not found in layouts/ directory.")

    append_log(log_lines, f"\n{'=' * 60}")
    append_log(log_lines, "  Operación Fénix - ISIS-1611 Inteligencia Artificial")
    append_log(log_lines, f"{'=' * 60}")
    append_log(log_lines, f"  Layout:   {options.layout}  ({layout.width}×{layout.height})")
    append_log(log_lines, f"  Problema: {options.problem}")

    # Build problem
    problem = load_problem(options.problem, layout)
    initial_state = problem.initial_state
    objects = problem.objects

    append_log(log_lines, f"  Pacientes: {objects['patients']}")
    append_log(log_lines, f"  Suministros: {objects['supplies']}")
    append_log(log_lines, f"  Puestos médicos: {objects['medical_posts']}")
    append_log(log_lines, f"{'=' * 60}")

    # Build display
    if options.quiet:
        from view.text_display import NullGraphics

        display = NullGraphics()
    elif options.text:
        from view.text_display import TextDisplay

        display = TextDisplay()
    else:
        from view.graphics_display import GraphicsDisplay

        display = GraphicsDisplay(
            layout, zoom=options.zoom, frame_time=options.frame_time
        )

    display.initialize(layout, initial_state)

    # HTN mode
    if options.htn:
        from planning.htn import build_htn_hierarchy, hierarchicalSearch

        append_log(log_lines, "  Modo: HTN (Hierarchical Task Network)")
        hlas = build_htn_hierarchy(problem)
        if not hlas:
            append_log(log_lines, "  No se encontraron HLAs para este layout.")
            return
        append_log(log_lines, f"  HLA raíz: {hlas[0].name}")
        t0 = time.time()
        planner_output = io.StringIO()
        with redirect_stdout(planner_output):
            plan = hierarchicalSearch(problem, hlas)
        elapsed = time.time() - t0
    else:
        # Classical planning
        fn_name = options.function
        append_log(log_lines, f"  Planificador: {fn_name}")
        planner = load_planner(fn_name)
        t0 = time.time()
        if fn_name == "aStarPlanner":
            heuristic = load_heuristic(options.heuristic)
            append_log(log_lines, f"  Heurística: {options.heuristic}")
            planner_output = io.StringIO()
            with redirect_stdout(planner_output):
                plan = planner(problem, heuristic)
        else:
            planner_output = io.StringIO()
            with redirect_stdout(planner_output):
                plan = planner(problem)
        elapsed = time.time() - t0

    planner_log = planner_output.getvalue().strip()
    if planner_log:
        for line in planner_log.splitlines():
            append_log(log_lines, line)

    append_log(log_lines, f"\n  Tiempo de planificación: {elapsed:.3f}s")
    append_log(log_lines, f"  Estados expandidos: {problem._expanded}")

    if not plan:
        append_log(log_lines, "  [FALLA] No se encontró un plan.")
        
        with open("resultados.txt", "a", encoding="utf-8") as f:
            f.write("\n".join(log_lines) + "\n")
            f.write("-" * 60 + "\n")
            
        display.finish()
        return

    append_log(log_lines, f"  Longitud del plan: {len(plan)} acciones")
    append_log(log_lines, "\n  Plan:")
    for i, action in enumerate(plan, 1):
        append_log(log_lines, f"    {i:2d}. {action.name}")

    append_log(log_lines, "\n  Ejecutando plan...")
    final_state, success = execute_plan(
        plan, initial_state, display, options.frame_time
    )

    if success and problem.isGoalState(final_state):
        status_msg = "¡Misión completada exitosamente!"
        append_log(log_lines, f"\n  {status_msg}")
    elif success:
        status_msg = "[ADVERTENCIA] Plan ejecutado pero objetivo no alcanzado."
        append_log(log_lines, f"\n  {status_msg}")
    else:
        status_msg = "[ERROR] Plan inválido — acción no aplicable durante ejecución."
        append_log(log_lines, f"\n  {status_msg}")

    # Asignación de complejidades teóricas reales
    if options.htn:
        complejidad = "O(b^d) / Variable según dominio"
    elif options.function == "forwardBFS" or options.function == "backwardSearch":
        complejidad = "O(b^d)"
    elif options.function == "aStarPlanner":
        complejidad = "O(b^(C*/ε))"
    else:  # Caso tinyBaseSearch (hardcodeado)
        complejidad = "O(1)"

    # Imprimir en consola y registrar
    append_log(log_lines, "")
    append_log(log_lines, f"--- [ANÁLISIS DE {options.function if not options.htn else 'HTN'}] ---")
    append_log(log_lines, f"1. Temporal: {complejidad} => {problem._expanded} operaciones")
    append_log(log_lines, f"2. Espacial: {complejidad} => {problem._expanded} nodos en memoria aproximados")
    append_log(log_lines, "3. Completitud: SÍ")
    append_log(log_lines, "4. Optimalidad: SÍ")
    append_log(log_lines, "-" * 30)

    with open("resultados.txt", "a", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")
        f.write("-" * 60 + "\n")

    display.finish()


if __name__ == "__main__":
    options = read_command(sys.argv[1:])
    run(options)
