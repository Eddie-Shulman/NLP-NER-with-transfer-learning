from NERTranserLearning.Experiment import Experiment


class ExperimentSoloTraining(Experiment):
    """
    Contains the configurations of the conventional training on various data-sets.
    Comment all configurations but the one you wish to run and run this code
    as a module.
    """

    @property
    def EXPERIMENT_PLAN(self):
        return [
            {
                'train': {
                    'dataset': 'ritter_dataset_train',
                    'input_checkpoint': None,
                    'output_checkpoint': 'trained/solo/ritter_train-tdd-simple.ckpt',
                    'model': 'tdd_simple_embedding_model',
                    'train': 'train_tdd_output',
                    'freeze_bi_lstm': False,
                    'freeze_output_layer': False
                },
                'test': ['ritter_dataset_test'],
            },
            # {
            #     'train': {
            #         'dataset': 'ritter_dataset_train',
            #         'input_checkpoint': None,
            #         'output_checkpoint': 'trained/solo/ritter_train-tdd-elmo.ckpt',
            #         'model': 'tdd_elmo_embedding_model',
            #         'train': 'train_tdd_output',
            #         'freeze_bi_lstm': False,
            #         'freeze_output_layer': False
            #     },
            #     'test': ['ritter_dataset_test']
            # },
            # {
            #     'train': {
            #         'dataset': 'ritter_dataset_train',
            #         'input_checkpoint': None,
            #         'output_checkpoint': 'trained/solo/ritter_train-crf-simple.ckpt',
            #         'model': 'crf_simple_embedding_model',
            #         'train': 'train_crf_output',
            #         'freeze_bi_lstm': False,
            #         'freeze_output_layer': False
            #     },
            #     'test': ['ritter_dataset_test']
            # },
            # {
            #     'train': {
            #         'dataset': 'ritter_dataset_train',
            #         'input_checkpoint': None,
            #         'output_checkpoint': 'trained/solo/ritter_train-crf-elmo.ckpt',
            #         'model': 'crf_elmo_embedding_model',
            #         'train': 'train_crf_output',
            #         'freeze_bi_lstm': False,
            #         'freeze_output_layer': False
            #     },
            #     'test': ['ritter_dataset_test']
            # },
            # {
            #     'train': {
            #         'dataset': 'btc_dataset_train',
            #         'input_checkpoint': None,
            #         'output_checkpoint': 'trained/solo/btc_train-crf-elmo.ckpt',
            #         'model': 'crf_elmo_embedding_model',
            #         'train': 'train_crf_output',
            #         'freeze_bi_lstm': False,
            #         'freeze_output_layer': False
            #     },
            #     'test': ['btc_dataset_test']
            # },
        ]


if __name__ == '__main__':
    ExperimentSoloTraining().run_experiment()
