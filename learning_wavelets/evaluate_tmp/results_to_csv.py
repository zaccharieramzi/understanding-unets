import pandas as pd

from learning_wavelets.evaluate_tmp.multiscale_eval import DEFAULT_NOISE_STDS

def results_to_csv(metrics_names, results, output_path):
    results_df = None
    i = 0
    for parameters, eval_res in results:
        if results_df is None:
            results_df = pd.DataFrame(
                columns=list(parameters.keys()) + ['noise_std'] + metrics_names,
            )
        for noise_res, noise_std in zip(eval_res, DEFAULT_NOISE_STDS):
            # NOTE: not efficient as per pandas doc, but easy
            row_dict = parameters
            row_dict.update({
                'noise_std': noise_std,
            })
            row_dict.update(dict(zip(metrics_names, noise_res)))
            results_df = results_df.append(row_dict)
            i += 1
    results_df.to_csv(output_path)
    return results_df
