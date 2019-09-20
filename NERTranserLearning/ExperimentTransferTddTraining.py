from NERTranserLearning.Experiment import Experiment


class ExperimentTransferTddTraining(Experiment):
    """
    Contains the configurations of the transfer learning training on various data-sets for both Elmo and naive
     embeddings with TDD output layer.
    It's crucial the training will run in a sequential order as they depend one on another.
    Comment all configurations but the one you wish to run and run this code
    as a module.
    """

    @property
    def EXPERIMENT_PLAN(self):

        return [
            {
                'train': {
                    'dataset': 'gmb_dataset',
                    'input_checkpoint': None,
                    'output_checkpoint': '../trained/transfer_tdd/gmb-tdd-simple.ckpt',
                    'model': 'tdd_simple_embedding_model',
                    'train': 'train_tdd_output',
                    'freeze_bi_lstm': False,
                    'freeze_output_layer': False
                },
                'test': ['btc_dataset','ritter_dataset'],
            },
            # {
            #     'train': {
            #         'dataset': 'btc_dataset_train',
            #         'input_checkpoint': '../trained/transfer_tdd/gmb-tdd-simple.ckpt',
            #         'output_checkpoint': '../trained/transfer_tdd/gmb+btc_train-tdd-simple.ckpt',
            #         'model': 'tdd_simple_embedding_model',
            #         'train': 'train_tdd_output',
            #         'freeze_bi_lstm': False,
            #         'freeze_output_layer': False
            #     },
            #     'test': ['btc_dataset_test','ritter_dataset'],
            # },
            # {
            #     'train': {
            #         'dataset': 'btc_dataset',
            #         'input_checkpoint': '../trained/transfer_tdd/gmb-tdd-simple.ckpt',
            #         'output_checkpoint': '../trained/transfer_tdd/gmb+btc-tdd-simple.ckpt',
            #         'model': 'tdd_simple_embedding_model',
            #         'train': 'train_tdd_output',
            #         'freeze_bi_lstm': False,
            #         'freeze_output_layer': False
            #     },
            #     'test': ['ritter_dataset'],
            # },
            # {
            #     'train': {
            #         'dataset': 'ritter_dataset_train',
            #         'input_checkpoint': '../trained/transfer_tdd/gmb+btc-tdd-simple.ckpt',
            #         'output_checkpoint': '../trained/transfer_tdd/gmb+btc+ritter_train-tdd-simple.ckpt',
            #         'model': 'tdd_simple_embedding_model',
            #         'train': 'train_tdd_output',
            #         'freeze_bi_lstm': False,
            #         'freeze_output_layer': False
            #     },
            #     'test': ['ritter_dataset_test'],
            # },
            # {
            #     'train': {
            #         'dataset': 'ritter_dataset',
            #         'input_checkpoint': '../trained/transfer_tdd/gmb+btc-tdd-simple.ckpt',
            #         'output_checkpoint': '../trained/transfer_tdd/gmb+btc+ritter-tdd-simple.ckpt',
            #         'model': 'tdd_simple_embedding_model',
            #         'train': 'train_tdd_output',
            #         'freeze_bi_lstm': False,
            #         'freeze_output_layer': False
            #     },
            #     'test': ['wsj_dataset'],
            # },
            # {
            #     'train': {
            #         'dataset': 'ritter_wsj_dataset_train',
            #         'input_checkpoint': '../trained/transfer_tdd/gmb+btc-tdd-simple.ckpt',
            #         'output_checkpoint': '../trained/transfer_tdd/gmb+btc+ritter_wsj_train-tdd-simple.ckpt',
            #         'model': 'tdd_simple_embedding_model',
            #         'train': 'train_tdd_output',
            #         'freeze_bi_lstm': False,
            #         'freeze_output_layer': False
            #     },
            #     'test': ['wsj_dataset_test'],
            # },
            # {
            #     'train': {
            #         'dataset': 'gmb_dataset',
            #         'input_checkpoint': None,
            #         'output_checkpoint': '../trained/transfer_tdd/gmb-tdd-elmo.ckpt',
            #         'model': 'tdd_elmo_embedding_model',
            #         'train': 'train_tdd_output',
            #         'freeze_bi_lstm': False,
            #         'freeze_output_layer': False
            #     },
            #     'test': ['btc_dataset','ritter_dataset'],
            # },
            # {
            #     'train': {
            #         'dataset': 'btc_dataset_train',
            #         'input_checkpoint': '../trained/transfer_tdd/gmb-tdd-elmo.ckpt',
            #         'output_checkpoint': '../trained/transfer_tdd/gmb+btc_train-tdd-elmo.ckpt',
            #         'model': 'tdd_elmo_embedding_model',
            #         'train': 'train_tdd_output',
            #         'freeze_bi_lstm': False,
            #         'freeze_output_layer': False
            #     },
            #     'test': ['btc_dataset_test','ritter_dataset'],
            # },
            # {
            #     'train': {
            #         'dataset': 'btc_dataset',
            #         'input_checkpoint': '../trained/transfer_tdd/gmb-tdd-elmo.ckpt',
            #         'output_checkpoint': '../trained/transfer_tdd/gmb+btc-tdd-elmo.ckpt',
            #         'model': 'tdd_elmo_embedding_model',
            #         'train': 'train_tdd_output',
            #         'freeze_bi_lstm': False,
            #         'freeze_output_layer': False
            #     },
            #     'test': ['ritter_dataset'],
            # },
            # {
            #     'train': {
            #         'dataset': 'ritter_dataset_train',
            #         'input_checkpoint': '../trained/transfer_tdd/gmb+btc-tdd-elmo.ckpt',
            #         'output_checkpoint': '../trained/transfer_tdd/gmb+btc+ritter_train-tdd-elmo.ckpt',
            #         'model': 'tdd_elmo_embedding_model',
            #         'train': 'train_tdd_output',
            #         'freeze_bi_lstm': False,
            #         'freeze_output_layer': False
            #     },
            #     'test': ['ritter_dataset_test'],
            # },
            # {
            #     'train': {
            #         'dataset': 'ritter_dataset',
            #         'input_checkpoint': '../trained/transfer_tdd/gmb+btc-tdd-elmo.ckpt',
            #         'output_checkpoint': '../trained/transfer_tdd/gmb+btc+ritter-tdd-elmo.ckpt',
            #         'model': 'tdd_elmo_embedding_model',
            #         'train': 'train_tdd_output',
            #         'freeze_bi_lstm': False,
            #         'freeze_output_layer': False
            #     },
            #     'test': ['wsj_dataset'],
            # },
            # {
            #     'train': {
            #         'dataset': 'ritter_wsj_dataset_train',
            #         'input_checkpoint': '../trained/transfer_tdd/gmb+btc-tdd-elmo.ckpt',
            #         'output_checkpoint': '../trained/transfer_tdd/gmb+btc+ritter_wsj_train-tdd-elmo.ckpt',
            #         'model': 'tdd_elmo_embedding_model',
            #         'train': 'train_tdd_output',
            #         'freeze_bi_lstm': False,
            #         'freeze_output_layer': False
            #     },
            #     'test': ['wsj_dataset_test'],
            # },
        ]


if __name__ == '__main__':
    ExperimentTransferTddTraining().run_experiment()
