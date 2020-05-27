import pandas as pd

from learning_wavelets.evaluate_tmp.multiscale_eval import DEFAULT_NOISE_STDS

def results_to_csv(metrics_names, results, output_path):
    results_df = None
    i = 0
    for parameters, eval_res in results:
        params_couples = sorted(list(parameters.items()), key=lambda x: x[0])
        params_keys = [x[0] for x in params_couples]
        params_values = [x[1] for x in params_couples]
        if results_df is None:
            results_df = pd.DataFrame(
                columns=params_keys + ['noise_std'] + metrics_names,
            )
        for noise_res, noise_std in zip(eval_res, DEFAULT_NOISE_STDS):
            # NOTE: not efficient as per pandas doc, but easy
            results_df.append(params_values + [noise_std] + noise_res)
            i += 1
    results_df.to_csv(output_path)
    return results_df
