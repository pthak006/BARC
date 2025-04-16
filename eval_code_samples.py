import os
import traceback
import sys
import json
from enum import Enum
import argparse
import numpy as np

from utils import parse_code
from execution import multi_execute_transformation
from seeds.common import *
from arc import train_problems, validation_problems
from arc.read import parse_dir
from arc.types import ArcIOPair, ArcProblem

from tqdm import tqdm

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def trace_calls(frame, event, arg):
    # Debugging function, can remain the same
    if event != 'call':
        return
    co = frame.f_code
    func_name = co.co_name
    if func_name == 'execve':
        filename = co.co_filename
        line_no = frame.f_lineno
        if 'lscpu' in str(arg):
            print(f"lscpu called from {filename}:{line_no}")
            traceback.print_stack(frame)
    return trace_calls

sys.settrace(trace_calls)

# --------------------------------------------------------------------------------
# Load ConceptARC (if needed)
def get_concept_arc_problems():
    problems = []
    if os.path.isdir("ConceptARC"):
        for problem_directory in os.listdir("ConceptARC"):
            full_path = os.path.join("ConceptARC", problem_directory)
            if os.path.isdir(full_path):
                problems.extend(parse_dir(full_path))
    return problems

concept_arc_problems = get_concept_arc_problems()
new_problems = []
for problem in concept_arc_problems:
    # Each ConceptARC problem can have multiple test pairs
    for ti, test_pair in enumerate(problem.test_pairs):
        new_problem = ArcProblem(
            uid=f"{problem.uid}-{ti}",
            train_pairs=problem.train_pairs,
            test_pairs=[test_pair]
        )
        new_problems.append(new_problem)
    # Possibly an assertion that each ConceptARC problem has 3 test pairs
concept_arc_problems = new_problems

# --------------------------------------------------------------------------------
# Global flags
TRANSPOSE = False
MULTI_EXECUTE = True

class GridComparisonResult(Enum):
    EQUAL = 0
    SHAPE_MISMATCH = 1
    CONTENT_MISMATCH = 2
    TYPE_MISMATCH = 3
    ERROR = 4
    NON_2D_ARRAY = 5

def compare_grids(output_grid, expected_output_grid):
    """Checks if output_grid == expected_output_grid, returns (result, ratio_of_correct)."""
    if isinstance(output_grid, str):
        return GridComparisonResult.ERROR, 0.0

    if not isinstance(output_grid, np.ndarray):
        return GridComparisonResult.TYPE_MISMATCH, 0.0

    if len(output_grid.shape) != 2:
        return GridComparisonResult.NON_2D_ARRAY, 0.0

    if output_grid.shape != expected_output_grid.shape:
        return GridComparisonResult.SHAPE_MISMATCH, 0.0

    if np.array_equal(output_grid, expected_output_grid):
        return GridComparisonResult.EQUAL, 1.0

    # Shapes match but content differs
    ratio = np.sum(output_grid == expected_output_grid) / np.prod(expected_output_grid.shape)
    return GridComparisonResult.CONTENT_MISMATCH, ratio

def get_arc_problem(uid):
    # Gather all ARC problems (train + validation + conceptARC if any)
    all_problems = train_problems + validation_problems + concept_arc_problems
    for problem in all_problems:
        if problem.uid == uid:
            return problem
    raise ValueError(f"Problem {uid} not found")

# --------------------------------------------------------------------------------
# Modified multi_validate that also stores the output grid

def multi_validate(arc_problem, codes):
    """
    For each snippet in codes, and for each input (both train & test),
    produce:
       results[code_idx][pair_idx] = (boolEqual, ratio, output_grid_as_ndarray or error).
    """
    pairs = arc_problem.train_pairs + arc_problem.test_pairs
    results = [[] for _ in range(len(codes))]

    # We run once per input pair, with multi_execute_transformation for all codes at once
    for pair_idx, pair in enumerate(pairs):
        input_grid = pair.x
        if TRANSPOSE:
            input_grid = input_grid.T
        try:
            # Execute all codes in parallel for this input
            output_grids = multi_execute_transformation(
                codes, [input_grid]*len(codes),
                random_seeds=[0]*len(codes),
                timeout=2,
                function_name="transform",
                num_workers=64
            )
        except KeyboardInterrupt:
            sys.exit()
        except Exception as e:
            print(f"multi_execute error: {e}")
            # fallback: treat them all as errors
            output_grids = ["error"]*len(codes)

        # Compare each code's output to the ground truth
        # If we used transpose for input, also transpose expected output
        expected_output = pair.y if not TRANSPOSE else pair.y.T

        for code_idx, out_grid in enumerate(output_grids):
            if isinstance(out_grid, str):
                # Some error
                results[code_idx].append((False, 0.0, out_grid))
                continue

            comparison_result, ratio = compare_grids(out_grid, expected_output)
            is_equal = (comparison_result == GridComparisonResult.EQUAL)
            results[code_idx].append((is_equal, ratio, out_grid))

    return results

# --------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer_file", help="Path to the .jsonl inference file (induction code).")
    parser.add_argument("--start-index", type=int, default=0, help="Index to start processing from")
    args = parser.parse_args()

    if not args.answer_file or not os.path.exists(args.answer_file):
        raise ValueError("Must provide a valid --answer_file path to the .jsonl with code responses.")

    with open(args.answer_file) as f:
        problem_answers = [json.loads(line) for line in f]

    # We'll save the expanded results in results/<filename>_exec_results_v4.jsonl
    import pathlib
    saving_path = pathlib.Path(args.answer_file).with_suffix("")  # remove .jsonl
    saving_file = saving_path.name + "_exec_results_v4.jsonl"
    saving_file = pathlib.Path("results") / saving_file
    os.makedirs("results", exist_ok=True)
    print(f"[eval_code_samples] Saving to {saving_file}")

    # Create the file first
    with open(saving_file, "w") as f:
        pass

    # Initialize accepted count with known value when using start-index
    accepted = 29 if args.start_index > 0 else 0
    
    for problem_idx, p in enumerate(tqdm(problem_answers[args.start_index:], desc="Evaluating code", initial=args.start_index, total=len(problem_answers))):
        uid = p["uid"]
        responses = p["responses"]
        print(f"\nProblem: {uid}")

        # For each snippet, parse out the code's text
        codes = []
        for i, response_text in enumerate(responses):
            parsed_codes = parse_code(response_text)  # parse_code returns a list of code blocks
            if parsed_codes:
                codes.append(parsed_codes[0])  # Just use the first code block
            else:
                codes.append("")               # No code found

        # Retrieve the corresponding ARC problem
        arc_problem = get_arc_problem(uid)

        # We'll store booleans: "passed all train pairs" and "passed train+test"
        train_verdicts = []
        train_test_verdicts = []
        # We'll store the actual output grids for each code snippet
        all_output_grids = []

        if MULTI_EXECUTE:
            # --- Multi-batch inference ---
            # Returns a list-of-lists: results[code_idx][pair_idx] = (is_equal, ratio, out_grid)
            results = multi_validate(arc_problem, codes)

            # Now interpret these results for each code
            for code_idx, result_for_code in enumerate(results):
                # result_for_code is a list of (boolEqual, ratio, out_grid), one per pair
                n_train = len(arc_problem.train_pairs)
                n_total = len(arc_problem.train_pairs) + len(arc_problem.test_pairs)

                # train_verdict: must pass all training pairs
                code_passes_train = all(item[0] for item in result_for_code[:n_train])
                # train+test verdict: must pass *all* pairs
                code_passes_train_test = all(item[0] for item in result_for_code)

                train_verdicts.append(code_passes_train)
                train_test_verdicts.append(code_passes_train_test)

                # Collect the actual outputs as JSON-serializable lists-of-lists
                code_outputs = []
                for (is_eq, ratio, out_grid) in result_for_code:
                    if isinstance(out_grid, str):
                        code_outputs.append(out_grid)  # e.g. "error"
                    else:
                        code_outputs.append(out_grid.tolist())
                all_output_grids.append(code_outputs)

                icon = "[+]" if code_passes_train else "[ ]"
                print(f"  {icon} Code {code_idx}: train_test={code_passes_train_test}")

        else:
            # --- Single-thread approach (slower) but could store each code's output in detail ---
            for code_text in codes:
                # We'll do exactly what validate(...) does but store outputs
                # If you want the multi approach, you can remove or keep this block
                pass
            # ... not showing the single-thread code for brevity

        # Decide if *any* snippet passes train+test
        if any(train_test_verdicts):
            accepted += 1

        # Attach to the JSON we'll save
        p["train_verdicts"] = train_verdicts            # list of booleans
        p["train_test_verdicts"] = train_test_verdicts  # list of booleans
        p["output_grids"] = all_output_grids            # each item is a list of shape (#pairs) sub-lists

        print(f"Accepted so far: {accepted}/{args.start_index + problem_idx + 1}")
        
        # Write this problem's result immediately after processing
        with open(saving_file, "a") as f:
            f.write(json.dumps(p, cls=NumpyEncoder) + "\n")
            f.flush()  # Ensure it's written to disk

    print(f"Accepted: {accepted}/{len(problem_answers)}")
    print(f"[eval_code_samples] All done. Wrote to: {saving_file}")

if __name__ == "__main__":
    main()
