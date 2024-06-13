from utilsd import get_output_dir, get_checkpoint_dir, setup_experiment
from utilsd.experiment import print_config
from utilsd.config import PythonConfig, RegistryConfig, RuntimeConfig, configclass

from forecaster.dataset import DATASETS
from forecaster.model import MODELS
from forecaster.network import NETWORKS
# import wandb

import warnings
warnings.filterwarnings('ignore')

@configclass
class ForecastConfig(PythonConfig):
    data: RegistryConfig[DATASETS]
    network: RegistryConfig[NETWORKS]
    model: RegistryConfig[MODELS]
    runtime: RuntimeConfig = RuntimeConfig()


def run_train(config):
    setup_experiment(config.runtime)
    print_config(config)
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="my-awesome-project",
        
    #     # track hyperparameters and run metadata
    #     config={
    #         "network_type": config.network.type,
    #         "dataset": config.data.file,
    #         "horizon": config.data.horizon
    #     }
    # )
    trainset = config.data.build(dataset_name="train")
    validset = config.data.build(dataset_name="valid")
    testset = config.data.build(dataset_name="test")
    network = config.network.build(input_size=trainset.num_variables, max_length=trainset.max_seq_len)
    model = config.model.build(
        network=network,
        output_dir=get_output_dir(),
        checkpoint_dir=get_checkpoint_dir(),
        out_size=config.model.out_size or trainset.num_classes,
    )
    print(model)
    model.fit(trainset, validset, testset)
    model.predict(trainset, "train")
    model.predict(validset, "valid")
    model.predict(testset, "test")
    return


if __name__ == "__main__":
    _config = ForecastConfig.fromcli()
    run_train(_config)
