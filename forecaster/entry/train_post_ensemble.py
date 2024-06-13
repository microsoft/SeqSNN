from utilsd import get_output_dir, get_checkpoint_dir, setup_experiment
from utilsd.experiment import print_config
from utilsd.config import PythonConfig, RegistryConfig, RuntimeConfig, configclass

from ..dataset import DATASETS, DATASETWRAPPER
from ..model import MODELS
from ..network import NETWORKS


@configclass
class ForecastConfig(PythonConfig):
    data: RegistryConfig[DATASETS]
    datasetwrapper: RegistryConfig[DATASETWRAPPER]
    network: RegistryConfig[NETWORKS]
    dispatcher: RegistryConfig[MODELS]
    ensemble: RegistryConfig[MODELS]
    runtime: RuntimeConfig = RuntimeConfig()


def run_train(config):
    setup_experiment(config.runtime)
    print_config(config)

    n_models = len(config.datasetwrapper.model_list)
    trainset = config.data.build(dataset="train")
    trainset_wrapper = config.datasetwrapper.build(data_index=trainset.get_index(), split="train")
    wrapped_trainset = trainset_wrapper.wrap(trainset)
    normalizer = wrapped_trainset.get_normalizer()

    # if validset is available, then create validset
    # here we use testset instead of it for usage description

    # validset = config.data.build(dataset="valid", normalizer=normalizer)
    validset = config.data.build(dataset="test", normalizer=normalizer)
    validset_wrapper = config.datasetwrapper.build(data_index=validset.get_index(), split="valid")
    wrapped_validset = validset_wrapper.wrap(validset)
    print(len(wrapped_validset.__getitem__(1)[0]))

    testset = config.data.build(dataset="test", normalizer=normalizer)
    testset_wrapper = config.datasetwrapper.build(data_index=testset.get_index(), split="test")
    wrapped_testset = testset_wrapper.wrap(testset)

    encoder = config.network.build(
        input_size=trainset.num_variables, max_length=trainset.max_seq_len
    )
    dispatcher = config.dispatcher.build(
        n_estimators=n_models,
        feature_encoder=encoder,
    )
    ensemble = config.ensemble.build(
        n_estimators=n_models,
        dispatcher=dispatcher,
        output_dir=get_output_dir(),
        checkpoint_dir=get_checkpoint_dir(),
    )
    ensemble.fit(wrapped_trainset, wrapped_validset, wrapped_testset)
    ensemble.predict(wrapped_trainset, "train")
    ensemble.predict(wrapped_validset, "valid")
    ensemble.predict(wrapped_testset, "test")
    return


if __name__ == "__main__":
    _config = ForecastConfig.fromcli()
    run_train(_config)
