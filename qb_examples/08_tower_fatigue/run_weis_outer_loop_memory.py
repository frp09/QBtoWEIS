import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path


# Input data
N_ITER = 1
TOWER_MASS_RTOL = 0.00005
TOWER_MASS_ATOL = 1.0
RESTART_FROM_OUTER = None


# Input files
run_dir = Path(__file__).resolve().parent
script_path = Path(__file__).resolve()

fname_wt_input = run_dir / "MED-15-300-RWT.yaml"
fname_modeling_options = run_dir / 'modeling_options_MED.yaml'
fname_analysis_options = run_dir / 'analysis_options_MED.yaml'

input_files = [fname_wt_input, fname_modeling_options, fname_analysis_options]


def load_restart_state(outer_iteration):
    """
    Load tower geometry and mass from a previous OpenMDAO recorder.

    This function is called only inside the isolated WEIS process.
    """
    import numpy as np
    import openmdao.api as om

    sql_path = (run_dir / f"tower_fatigue_output_{outer_iteration}" / f"opt_outer_{outer_iteration}" / f"recorder_outer_{outer_iteration}.sql")

    reader = om.CaseReader(str(sql_path))
    case_names = reader.list_cases("driver", out_stream=None)

    if not case_names:
        raise RuntimeError(f"No driver cases found in {sql_path}")

    last_case = reader.get_case(case_names[-1])

    geometry_override = {"tower.diameter": np.asarray(last_case.get_val("tower.diameter"), dtype=float).copy(),
                         "tower.layer_thickness": np.asarray(last_case.get_val("tower.layer_thickness"), dtype=float).copy()
                        }

    tower_mass = float(np.asarray(last_case.get_val("towerse.tower_mass"), dtype=float).ravel()[0])

    return geometry_override, tower_mass


def load_state_file(state_file):
    """
    Load the tower geometry and mass produced by the previous outer loop.
    """
    import numpy as np

    with open(state_file, "r") as f:
        state = json.load(f)

    geometry_override = {"tower.diameter": np.asarray(state["diameter"], dtype=float),
                         "tower.layer_thickness": np.asarray(state["layer_thickness"], dtype=float)
                        }

    return geometry_override, float(state["tower_mass"])


def run_single_outer(outer_iteration, result_file, previous_state_file=None, restart_from_outer=None):
    """
    Execute one outer-loop WEIS optimization in an isolated Python process.

    When this function finishes, the Python process exits and Linux releases
    all memory owned by WEIS/OpenMDAO/QBlade post-processing.
    """
    import numpy as np
    from weis.glue_code.runWEIS import run_weis

    print(f"\nStarting isolated WEIS process for outer loop n {outer_iteration} - PID {os.getpid()}\n", flush=True)

    previous_tower_mass = None

    if previous_state_file is not None:
        geometry_override, previous_tower_mass = load_state_file(previous_state_file)

    elif restart_from_outer is not None:
        geometry_override, previous_tower_mass = load_restart_state(restart_from_outer)

    else:
        geometry_override = None

    modeling_override = {
        "General": {
            "qblade_configuration": {"QB_run_dir": str(run_dir / f"tower_fatigue_output_{outer_iteration}" / f"qblade_outer_{outer_iteration}")}
                   },
        "QBlade": {"freeze_loads": True},
    }

    analysis_override = {
        "general": {
            "folder_output": str(run_dir / f"tower_fatigue_output_{outer_iteration}" / f"opt_outer_{outer_iteration}")
        },
        "recorder": {
            "file_name": f"recorder_outer_{outer_iteration}.sql",
            "flag": True,
        },
    }

    wt_opt = None

    try:
        wt_opt, _, _ = run_weis(
            *(str(path) for path in input_files),
            geometry_override=geometry_override,
            modeling_override=modeling_override,
            analysis_override=analysis_override,
        )

        new_diameter = np.asarray(wt_opt.get_val("tower.diameter"), dtype=float).copy()
        new_thickness = np.asarray(wt_opt.get_val("tower.layer_thickness"), dtype=float).copy()

        current_tower_mass = float(np.asarray(wt_opt.get_val("towerse.tower_mass"), dtype=float).ravel()[0])
        regression_values = {
            "towerse.tower_mass": np.asarray(wt_opt.get_val("towerse.tower_mass"), dtype=float).copy(),
            "tower_fatigue_post.constr_fatigue": np.asarray(wt_opt.get_val("tower_fatigue_post.constr_fatigue"), dtype=float).copy(),
            "towerse_post.constr_stress": np.asarray(wt_opt.get_val("towerse_post.constr_stress"), dtype=float).copy(),
            "towerse_post.constr_global_buckling": np.asarray(wt_opt.get_val("towerse_post.constr_global_buckling"), dtype=float).copy(),
            "towerse_post.constr_shell_buckling": np.asarray(wt_opt.get_val("towerse_post.constr_shell_buckling"), dtype=float).copy(),
        }

    finally:
        if wt_opt is not None:
            try:
                wt_opt.cleanup()
            except Exception as cleanup_error:
                print(f"Warning: cleanup failed for outer_{outer_iteration}: {cleanup_error}")

    # Only these small quantities have to survive after this process exits.
    state = {
        "diameter": new_diameter.tolist(),
        "layer_thickness": new_thickness.tolist(),
        "tower_mass": current_tower_mass,
        "previous_tower_mass": previous_tower_mass,
        "regression_values": {name: value.tolist() for name, value in regression_values.items()},
    }

    result_file.parent.mkdir(parents=True, exist_ok=True)

    with open(result_file, "w") as f:
        json.dump(state, f)

    print(f"\nOuter loop n {outer_iteration} finished in PID {os.getpid()}.\n WEIS process PID {os.getpid()} will now terminate.\n", flush=True)


def run_outer_loop():
    """
    Lightweight supervisor.

    It never imports WEIS, OpenMDAO or NumPy. Each run_weis() call is executed
    by a fresh Python interpreter.
    """

    if os.environ.get("SLURM_NTASKS") not in (None, "1"):
        raise RuntimeError("This script must be run with one Slurm task (SLURM_NTASKS={os.environ['SLURM_NTASKS']}).")

    converged = False
    previous_tower_mass = None
    previous_state_file = None

    if RESTART_FROM_OUTER is not None:
        if RESTART_FROM_OUTER >= N_ITER:
            raise ValueError("RESTART_FROM_OUTER leaves no remaining outer iterations.")

        start_outer = RESTART_FROM_OUTER + 1

        print(f"\nRestart mode on: restarting from outer loop n {RESTART_FROM_OUTER}\n", flush=True)

    else:
        start_outer = 0

    print(f"\nOuter-loop supervisor PID: {os.getpid()}\n", flush=True)

    start_outer_time = time.perf_counter()

    for outer_iteration in range(start_outer, N_ITER):

        print(f"\nStarting outer loop n {outer_iteration}\n", flush=True)

        start_inner_time = time.perf_counter()
        output_dir = (run_dir / f"tower_fatigue_output_{outer_iteration}")
        result_file = (output_dir / f"outer_state_{outer_iteration}.json")

        # Never accept an old state file if the new WEIS process fails.
        if result_file.exists():
            result_file.unlink()

        command = [
            sys.executable,
            str(script_path),
            "--single-outer",
            str(outer_iteration),
            "--result-file",
            str(result_file),
        ]

        if previous_state_file is not None:
            command.extend(
                [
                    "--previous-state-file",
                    str(previous_state_file),
                ]
            )

        elif RESTART_FROM_OUTER is not None:
            command.extend(
                [
                    "--restart-from-outer",
                    str(RESTART_FROM_OUTER),
                ]
            )

        subprocess.run(command, check=True)

        # If this line is reached, the WEIS process has completely terminated.
        if not result_file.exists():
            raise RuntimeError(f"Outer loop {outer_iteration} completed without producing {result_file}")

        with open(result_file, "r") as f:
            state = json.load(f)

        current_tower_mass = float(state["tower_mass"])

        # Only needed for the first iteration after a restart.
        if previous_tower_mass is None:
            saved_previous_mass = state["previous_tower_mass"]

            if saved_previous_mass is not None:
                previous_tower_mass = float(saved_previous_mass)

        previous_state_file = result_file
        end_inner_time = time.perf_counter()

        print(f"\n⏱️ Outer loop n{outer_iteration} optimization time {end_inner_time - start_inner_time:.4f} s\n")


        if previous_tower_mass is not None:

            print(f"\nPrevious outer loop mass: {previous_tower_mass}\n")
            print(f"\nCurrent outer loop mass: {current_tower_mass}\n")

            relative_change = (abs(current_tower_mass - previous_tower_mass) / max(abs(previous_tower_mass), 1.0))

            if relative_change < TOWER_MASS_RTOL:

                print(f"\nOuter loop converged as relative change between outer loop {outer_iteration} and outer loop {outer_iteration - 1} is {relative_change}\n")
                converged = True
                break

            absolute_change = abs(current_tower_mass - previous_tower_mass)

            if absolute_change < TOWER_MASS_ATOL:

                print(f"\nOuter loop converged as absolute difference between outer loop {outer_iteration} and outer loop {outer_iteration - 1} is {absolute_change}\n")
                converged = True
                break

        previous_tower_mass = current_tower_mass

    if not converged:
        print("\nOuter loop stopped as maximum number of outer loop iterations was reached\n")

    end_outer_time = time.perf_counter()

    print(f"\n⏱️ Whole optimization time {end_outer_time - start_outer_time:.4f} s\n")

    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-outer", type=int)
    parser.add_argument("--result-file", type=Path)
    parser.add_argument("--previous-state-file", type=Path)
    parser.add_argument("--restart-from-outer", type=int)

    args = parser.parse_args()

    if args.single_outer is not None:

        if args.result_file is None:
            raise ValueError("--result-file is required with --single-outer.")

        run_single_outer(
            outer_iteration=args.single_outer,
            result_file=args.result_file,
            previous_state_file=args.previous_state_file,
            restart_from_outer=args.restart_from_outer,
        )

    else:
        run_outer_loop()


if __name__ == "__main__":
    main()
