
import argparse
import yaml
from pathlib import Path
from typing import List, Dict, Any
import datetime
import numpy as np
from ctf4science.data_module import load_dataset, get_prediction_timesteps, parse_pair_ids, get_applicable_plots, load_validation_dataset, get_validation_prediction_timesteps
from ctf4science.eval_module import evaluate_custom, save_results
from ctf4science.visualization_module import Visualization
import esn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

file_dir = Path(__file__).parent

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

    if 'batch_id' in config['model']:
        batch_id = config['model']['batch_id']
    else:
        batch_id = "test_batch"
    pair_ids = parse_pair_ids(config['dataset'])

    model_name = f"{config['model']['name']}"


    # batch_id = f"batch_"

    # batch_id = f"{batch_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Initialize batch results dictionary for summary
    batch_results = {
        'batch_id': batch_id,
        'model': model_name,
        'dataset': dataset_name,
        'pairs': []
    }


    # Get applicable visualizations for the dataset
    applicable_plots = get_applicable_plots(dataset_name)

    # Process each sub-dataset
    for pair_id in pair_ids:
        # Load sub-dataset
        if 'train_split' in config['model']:
            train_split = config['model']['train_split']
        else:
            train_split = 0.8
        train_data, val_data, init_data = load_validation_dataset(dataset_name, pair_id, train_split)
        prediction_timesteps = get_validation_prediction_timesteps(dataset_name, pair_id, train_split)
        prediction_horizon_steps = prediction_timesteps.shape[0]

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
        
        if pair_id not in [8,9]:
            train_seq = np.swapaxes(train_data, 1, 2)
            model, R = esn.train_ESN_forecaster(model,
                                                train_seq,
                                                spinup=spinup,
                                                initial_res_state=jnp.zeros((chunks, res_dim)),
                                                beta=beta)


        if pair_id in [1, 3, 5, 6, 7]:
            pred_data = model.forecast(prediction_horizon_steps, R[-1]).T

        elif pair_id in [2, 4]:
            train_seq = train_seq[0]
            pred_data = model.denoise(train_seq, jnp.zeros((chunks, res_dim)))[:-1]
            pred_data = jnp.vstack([train_seq[:1, :], pred_data])
            pred_data = pred_data.at[:spinup].set(train_seq[:spinup])
            pred_data = pred_data.T

        elif pair_id in [8, 9]:

            
            train_data = np.vstack([np.array(train_data), np.array(val_data).reshape(1, val_data.shape[0], -1)])
            cutoff_index = int(train_split * train_data.shape[2])

            val_data = train_data[:, :, cutoff_index:]
            train_data = train_data[:,:, :cutoff_index]


            train_data = np.swapaxes(train_data, 1,2)
            print(pair_id)
            print(train_data.shape)
            val_data = np.swapaxes(val_data, 1,2)

            model, R = esn.train_ESN_forecaster(model,
                                                train_data,
                                                spinup=spinup,
                                                initial_res_state=jnp.zeros((chunks, res_dim)),
                                                beta=beta)

            prediction_horizon_steps = val_data.shape[1]
            R1 = model.force(train_data[0, -spinup-1:-1], jnp.zeros((chunks, res_dim)))
            pred_data1 = model.forecast(prediction_horizon_steps, R1[-1]).T
            val_data1 = val_data[0,:prediction_horizon_steps].T

            R2 = model.force(train_data[1, -spinup-1:-1], jnp.zeros((chunks, res_dim)))
            pred_data2 = model.forecast(prediction_horizon_steps, R2[-1]).T
            val_data2 = val_data[1,:prediction_horizon_steps].T

            R3 = model.force(train_data[2, -spinup-1:-1], jnp.zeros((chunks, res_dim)))
            pred_data3 = model.forecast(prediction_horizon_steps, R3[-1]).T
            val_data3 = val_data[2,:prediction_horizon_steps].T

            val_data = jnp.vstack([val_data1, val_data2, val_data3])
            pred_data = jnp.vstack([pred_data1, pred_data2, pred_data3])


        else:
            raise ValueError('Incorect pair_id.')
        results = evaluate_custom(dataset_name, pair_id, val_data, pred_data)

        # Save results for this sub-dataset and get the path to the results directory
        # results_directory = save_results(dataset_name, model_name, batch_id, pair_id, config, pred_data, results)

        # Append metrics to batch results
        # Convert metric values to plain Python floats for YAML serialization
        batch_results['pairs'].append({
            'pair_id': pair_id,
            'metrics': results
        })

    # with open(results_directory.parent / 'batch_results.yaml', 'w') as f:
    #     yaml.dump(batch_results, f)
    results_file = file_dir / f"results_{batch_id}.yaml"
    with open(results_file, 'w') as f:
        yaml.dump(batch_results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="Path to the configuration file.")
    args = parser.parse_args()
    main(args.config)