
import argparse
import yaml
from pathlib import Path
from typing import List, Dict, Any
import datetime
import numpy as np
from ctf4science.data_module import load_dataset, get_prediction_timesteps, parse_pair_ids, get_applicable_plots
from ctf4science.eval_module import evaluate, save_results
from ctf4science.visualization_module import Visualization
import esn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

def main(config_path: str) -> None:
    """
    Main function to run the naive baseline model on specified sub-datasets.

    Loads configuration, parses pair_ids, initializes the model, generates predictions,
    evaluates them, and saves results for each sub-dataset under a batch identifier.

    Args:
        config_path (str): Path to the configuration file.
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load dataset name and get list of sub-dataset train/test pair_ids
    dataset_name = config['dataset']['name']
    pair_ids = parse_pair_ids(config['dataset'])

    model_name = f"{config['model']['name']}"

    # Generate a unique batch_id for this run, you can add any descriptions you want
    #   e.g. f"batch_{learning_rate}_"
    batch_id = f"batch_"
    # temp = str(config['model']['Wr_spectral_radius'])
    # batch_id = batch_id + temp
 
    # Define the name of the output folder for your batch
    batch_id = f"{batch_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Initialize batch results dictionary for summary
    batch_results = {
        'batch_id': batch_id,
        'model': model_name,
        'dataset': dataset_name,
        'pairs': []
    }

    # Initialize Visualization object
    viz = Visualization()

    # Get applicable visualizations for the dataset
    applicable_plots = get_applicable_plots(dataset_name)

    # Process each sub-dataset
    for pair_id in pair_ids:
        # Load sub-dataset
        train_data, init_data = load_dataset(dataset_name, pair_id, transpose=True)
        train_data = np.array(train_data)

        # # Load initialization matrix if it exists
        # if init_data is None:
        #     # Stack all training matrices to get a single training matrix
        #     train_data = np.concatenate(train_data, axis=1)
        # else:
        #     # If we are given a burn-in matrix, use it as the training matrix
        #     train_data = init_data
        params = config['model']
        spec_rad = params['Wr_spectral_radius']
        leak_rate = params['leak_rate']
        embedding_scaling = params['embedding_scaling']
        Wr_density = params['Wr_density']
        chunks = params['chunks']
        locality = params['locality']
        beta = float(params['beta'])
        bias = params['bias']
        seed = params['seed']
        res_dim = params['res_dim']
        data_dim = params['data_dim']
        spinup = params['spinup']

        model = esn.ESN(data_dim=data_dim,
                      res_dim=res_dim,
                      leak_rate=leak_rate,
                      bias=bias,
                      embedding_scaling=embedding_scaling,
                      Wr_density=Wr_density,
                      Wr_spectral_radius=spec_rad,
                      dtype=jnp.float64,
                      seed=seed,
                      chunks=chunks,
                      locality=locality)
        
        train_seq = np.swapaxes(train_data, 1, 2)
        
        model, R = esn.train_ESN_forecaster(model,
                                            train_seq,
                                            spinup=spinup,
                                            initial_res_state=jnp.zeros((chunks, res_dim)),
                                            beta=beta)

        # Load metadata (to provide forecast length)
        prediction_timesteps = get_prediction_timesteps(dataset_name, pair_id)
        prediction_horizon_steps = prediction_timesteps.shape[0]

        # Initialize the model with the config and train_data
        #  model = NaiveBaseline(config, train_data, prediction_horizon_steps, pair_id)

        # Generate predictions
        # pred_data = model.predict()
        if pair_id in [1, 3, 5, 6, 7]:
            pred_data = model.forecast(prediction_horizon_steps, R[-1]).T

        elif pair_id in [2, 4]:
            train_seq = train_seq[0]
            pred_data = model.denoise(train_seq, jnp.zeros((chunks, res_dim)))[:-1]
            pred_data = jnp.vstack([train_seq[:1, :], pred_data])
            pred_data = pred_data.at[:spinup].set(train_seq[:spinup])
            pred_data = pred_data.T

        elif pair_id in [8, 9]:
            R = model.force(init_data.T[:-1], jnp.zeros((chunks, res_dim)))
            pred_data = model.forecast(prediction_horizon_steps, R[-1]).T
        else:
            raise ValueError('Incorect pair_id.')

        # Evaluate predictions using default metrics
        results = evaluate(dataset_name, pair_id, pred_data.T)

        # Save results for this sub-dataset and get the path to the results directory
        results_directory = save_results(dataset_name, model_name, batch_id, pair_id, config, pred_data.T, results)

        # Append metrics to batch results
        # Convert metric values to plain Python floats for YAML serialization
        batch_results['pairs'].append({
            'pair_id': pair_id,
            'metrics': results
        })

        # Generate and save visualizations that are applicable to this dataset
        for plot_type in applicable_plots:
            fig = viz.plot_from_batch(dataset_name, pair_id, results_directory, plot_type=plot_type)
            viz.save_figure_results(fig, dataset_name, model_name, batch_id, pair_id, plot_type, results_directory)

    # Save aggregated batch results
    with open(results_directory.parent / 'batch_results.yaml', 'w') as f:
        yaml.dump(batch_results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="Path to the configuration file.")
    args = parser.parse_args()
    main(args.config)
