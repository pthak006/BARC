from tqdm import tqdm
import orjsonl
import os
from collections import Counter
# from datasets import load_dataset  # If you’re not using this, can remove
from arc import validation_problems

# Only evaluate induction solutions.
MAX_FILES_TO_LOAD = 10000

# POINT THIS TO YOUR DIRECTORY WITH THE .jsonl RESULT FILE(S):
INDUCTION_SAMPLE_EXEC_RESULTS_DIRS_AND_SAMPLE_SIZE = [
    ("results", 128),  # e.g. "results" is the folder containing your single .jsonl
]

def grid_2d_to_tuple(grid):
    return tuple(tuple(row) for row in grid)

def tuple_to_grid_2d(t):
    return [list(row) for row in t]

def main():
    uid_to_problem = {p.uid: p for p in validation_problems}

    # 1) Gather all induction `.jsonl` files from the specified directories
    data_from_each_folder = []
    for induction_dir, num_samples_used in INDUCTION_SAMPLE_EXEC_RESULTS_DIRS_AND_SAMPLE_SIZE:
        jsonl_files = [
            f for f in os.listdir(induction_dir) if f.endswith(".jsonl")
        ]
        jsonl_files.sort()
        jsonl_files = jsonl_files[:MAX_FILES_TO_LOAD]

        print(f"Loading {len(jsonl_files)} jsonl files from {induction_dir}")
        all_data = []
        for file in tqdm(jsonl_files):
            path = os.path.join(induction_dir, file)
            all_data.append(orjsonl.load(path=path))

        # Combine all JSONL data into a single dict keyed by problem UID
        data_dict = {}
        print("Collecting induction samples…")
        for chunk in tqdm(all_data):
            for problem_dict in chunk:
                uid = problem_dict["uid"]
                if uid not in data_dict:
                    data_dict[uid] = {"train_verdicts": [], "output_grids": []}
                data_dict[uid]["train_verdicts"].extend(problem_dict["train_verdicts"])
                data_dict[uid]["output_grids"].extend(problem_dict["output_grids"])

        # Cap the number of used samples according to num_samples_used
        for uid, vals in data_dict.items():
            vals["train_verdicts"] = vals["train_verdicts"][:num_samples_used]
            vals["output_grids"]   = vals["output_grids"][:num_samples_used]
            assert len(vals["train_verdicts"]) == len(vals["output_grids"]) == num_samples_used

        data_from_each_folder.append(data_dict)

    # 2) Merge across multiple induction sources (if you have more than one folder)
    merged_data = {}
    for data_dict in data_from_each_folder:
        for uid, vals in data_dict.items():
            if uid not in merged_data:
                merged_data[uid] = {"train_verdicts": [], "output_grids": []}
            merged_data[uid]["train_verdicts"].extend(vals["train_verdicts"])
            merged_data[uid]["output_grids"].extend(vals["output_grids"])

    # Ensure merged counts match
    for uid, vals in merged_data.items():
        n_verdicts = len(vals["train_verdicts"])
        n_outputs  = len(vals["output_grids"])
        assert n_verdicts == n_outputs, f"Mismatch for {uid}"

    # 3) Perform pass@2 with majority voting
    induction_submission = {p.uid: [] for p in validation_problems}

    for uid, vals in merged_data.items():
        problem = uid_to_problem[uid]
        num_test   = len(problem.test_pairs)
        num_train  = len(problem.train_pairs)

        # We'll store final top-2 solutions per test index
        per_test_outputs = [[] for _ in range(num_test)]

        # For each test index, we do a majority vote among all code solutions
        # that passed the training examples
        for test_idx in range(num_test):
            # Tally: code solutions that pass train → which final grid for test_idx
            counter = Counter()
            for train_ok, grids in zip(vals["train_verdicts"], vals["output_grids"]):
                if train_ok:  # means it passed all train pairs
                    # the test output is after the train outputs in the same list
                    test_grid_tuple = grid_2d_to_tuple(grids[num_train + test_idx])
                    counter[test_grid_tuple] += 1

            # top2 solutions by frequency
            top_2 = counter.most_common(2)

            # Convert each solution from tuple-back to a 2D list
            final_solutions = []
            for (grid_tuple, _) in top_2:
                final_solutions.append(tuple_to_grid_2d(grid_tuple))

            per_test_outputs[test_idx] = final_solutions

        induction_submission[uid] = per_test_outputs

    # 4) Calculate pass@2 for induction
    induction_pass_at_2 = 0.0
    for uid, test_outputs_for_uid in induction_submission.items():
        problem = uid_to_problem[uid]
        for test_idx, test_pair in enumerate(problem.test_pairs):
            # ground truth
            gt_grid = test_pair.y.tolist()
            candidate_solutions = test_outputs_for_uid[test_idx]
            # If any of top 2 = ground truth
            if any(sol == gt_grid for sol in candidate_solutions):
                # partial credit if multiple test pairs
                induction_pass_at_2 += 1.0 / len(problem.test_pairs)

    # total # of tasks
    num_tasks = len(validation_problems)
    print(f"Induction pass@2: {induction_pass_at_2}/{num_tasks} = {induction_pass_at_2/num_tasks:.4f}")


if __name__ == "__main__":
    main()
