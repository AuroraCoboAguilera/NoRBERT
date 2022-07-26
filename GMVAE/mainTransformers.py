'''

Main programme
Conditional Gaussian Mixture Variational Autoencoder
Used for Transformer models

Author: Aurora Cobo Aguilera
Date: 1st April 2020
Update: 22th May 2020


'''

from utils import *
from TransformerOutputDataset import TransformerOutputDataset
from torch.utils.data import DataLoader
from GMVAE import *


# python mainTransformers.py --hidden_dim=1500 --l_rate=5e-5 --sigma=1e-2 --z_dim=150 --w_dim=50 --K=20 --train=1 --restore=0 --results=0 --epochs=500 --layers=6 --batch_size=1024 --dropout=0.3 --dataset_name="bert-base-uncased_sst2_10epochs_top" --train_dataroot="../TransformersNLP/results/bert-base-uncased_sst2_10epochs/topHiddenState_6229_train.hdf5" --test_dataroot="../TransformersNLP/results/bert-base-uncased_sst2_10epochs/topHiddenState_693_test.hdf5" --cuda=1 --device=2
# python mainTransformers.py --hidden_dim=1500 --l_rate=5e-5 --sigma=1e-2 --z_dim=150 --w_dim=50 --K=20 --train=1 --restore=0 --results=0 --epochs=500 --layers=6 --batch_size=1024 --dropout=0.3 --dataset_name="bert-base-uncased_trec_10epochs_top" --train_dataroot="../TransformersNLP/results/bert-base-uncased_trec_10epochs/topHiddenState_4907_train.hdf5" --test_dataroot="../TransformersNLP/results/bert-base-uncased_trec_10epochs/topHiddenState_547_test.hdf5" --cuda=1 --device=3
# python mainTransformers.py --hidden_dim=1500 --l_rate=5e-5 --sigma=1e-2 --z_dim=150 --w_dim=50 --K=20 --train=1 --restore=0 --results=0 --epochs=500 --layers=6 --batch_size=1024 --dropout=0.3 --dataset_name="bert-base-uncased_multi30k_10epochs_top" --train_dataroot="../TransformersNLP/results/bert-base-uncased_multi30k_10epochs/topHiddenState_29000_train.hdf5" --test_dataroot="../TransformersNLP/results/bert-base-uncased_multi30k_10epochs/topHiddenState_1000_test.hdf5" --cuda=1 --device=3
# python mainTransformers.py --hidden_dim=1500 --l_rate=5e-5 --sigma=1e-2 --z_dim=150 --w_dim=50 --K=20 --train=1 --restore=0 --results=0 --epochs=500 --layers=6 --batch_size=1024 --dropout=0.3 --dataset_name="roberta-base_sst2_10epochs_top" --train_dataroot="../TransformersNLP/results/roberta-base_sst2_10epochs/topHiddenState_6229_train.hdf5" --test_dataroot="../TransformersNLP/results/roberta-base_sst2_10epochs/topHiddenState_693_test.hdf5" --cuda=1 --device=0
# python mainTransformers.py --hidden_dim=1500 --l_rate=5e-5 --sigma=1e-2 --z_dim=150 --w_dim=50 --K=20 --train=1 --restore=0 --results=0 --epochs=500 --layers=6 --batch_size=1024 --dropout=0.3 --dataset_name="roberta-base_trec_10epochs_top" --train_dataroot="../TransformersNLP/results/roberta-base_trec_10epochs/topHiddenState_4907_train.hdf5" --test_dataroot="../TransformersNLP/results/roberta-base_trec_10epochs/topHiddenState_547_test.hdf5" --cuda=1 --device=1
# python mainTransformers.py --hidden_dim=1500 --l_rate=5e-5 --sigma=1e-2 --z_dim=150 --w_dim=50 --K=20 --train=1 --restore=0 --results=0 --epochs=500 --layers=6 --batch_size=1024 --dropout=0.3 --dataset_name="roberta-base_multi30k_10epochs_top" --train_dataroot="../TransformersNLP/results/roberta-base_multi30k_10epochs/topHiddenState_29000_train.hdf5" --test_dataroot="../TransformersNLP/results/roberta-base_multi30k_10epochs/topHiddenState_1000_test.hdf5" --cuda=1 --device=2
# python mainTransformers.py --hidden_dim=1500 --l_rate=5e-5 --sigma=1e-2 --z_dim=150 --w_dim=50 --K=20 --train=1 --restore=0 --results=0 --epochs=500 --layers=6 --batch_size=1024 --dropout=0.3 --dataset_name="xlm-roberta-base_trec_10epochs_top" --train_dataroot="../TransformersNLP/results/xlm-roberta-base_trec_10epochs/topHiddenState_4907_train.hdf5" --test_dataroot="../TransformersNLP/results/xlm-roberta-base_trec_10epochs/topHiddenState_547_test.hdf5" --cuda=1 --device=2
# python mainTransformers.py --hidden_dim=1500 --l_rate=5e-5 --sigma=1e-2 --z_dim=150 --w_dim=50 --K=20 --train=1 --restore=0 --results=0 --epochs=500 --layers=6 --batch_size=1024 --dropout=0.3 --dataset_name="xlm-roberta-base_sst2_10epochs_top" --train_dataroot="../TransformersNLP/results/xlm-roberta-base_sst2_10epochs/topHiddenState_6229_train.hdf5" --test_dataroot="../TransformersNLP/results/xlm-roberta-base_sst2_10epochs/topHiddenState_693_test.hdf5" --cuda=1 --device=1
# python mainTransformers.py --hidden_dim=1500 --l_rate=5e-5 --sigma=1e-2 --z_dim=150 --w_dim=50 --K=20 --train=1 --restore=0 --results=0 --epochs=500 --layers=6 --batch_size=1024 --dropout=0.3 --dataset_name="xlm-roberta-base_multi30k_10epochs_top" --train_dataroot="../TransformersNLP/results/xlm-roberta-base_multi30k_10epochs/topHiddenState_29000_train.hdf5" --test_dataroot="../TransformersNLP/results/xlm-roberta-base_multi30k_10epochs/topHiddenState_1000_test.hdf5" --cuda=1 --device=3

# python mainTransformers.py --hidden_dim=1500 --l_rate=5e-5 --sigma=1e-4 --z_dim=150 --w_dim=50 --K=20 --train=1 --restore=0 --results=0 --epochs=4000 --layers=6 --batch_size=1024 --dropout=0.3 --dataset_name="bert-base-uncased_sst2_10epochs_1deep" --train_dataroot="../TransformersNLP/results/bert-base-uncased_sst2_10epochs/1deepHiddenState_6229_train.hdf5" --test_dataroot="../TransformersNLP/results/bert-base-uncased_sst2_10epochs/1deepHiddenState_693_test.hdf5" --cuda=1 --device=3
# python mainTransformers.py --hidden_dim=1500 --l_rate=5e-5 --sigma=1e-5 --z_dim=150 --w_dim=50 --K=20 --train=1 --restore=0 --results=0 --epochs=5000 --layers=6 --batch_size=1024 --dropout=0.3 --dataset_name="bert-base-uncased_trec_10epochs_1deep" --train_dataroot="../TransformersNLP/results/bert-base-uncased_trec_10epochs/1deepHiddenState_4907_train.hdf5" --test_dataroot="../TransformersNLP/results/bert-base-uncased_trec_10epochs/1deepHiddenState_547_test.hdf5" --cuda=1 --device=2
# python mainTransformers.py --hidden_dim=1500 --l_rate=5e-5 --sigma=1e-4 --z_dim=150 --w_dim=50 --K=20 --train=1 --restore=0 --results=0 --epochs=2000 --layers=6 --batch_size=1024 --dropout=0.3 --dataset_name="bert-base-uncased_multi30k_10epochs_1deep" --train_dataroot="../TransformersNLP/results/bert-base-uncased_multi30k_10epochs/1deepHiddenState_29000_train.hdf5" --test_dataroot="../TransformersNLP/results/bert-base-uncased_multi30k_10epochs/1deepHiddenState_1000_test.hdf5" --cuda=1 --device=2
# python mainTransformers.py --hidden_dim=1500 --l_rate=5e-5 --sigma=1e-7 --z_dim=150 --w_dim=50 --K=20 --train=1 --restore=0 --results=0 --epochs=5000 --layers=6 --batch_size=1024 --dropout=0.3 --dataset_name="roberta-base_trec_10epochs_1deep" --train_dataroot="../TransformersNLP/results/roberta-base_trec_10epochs/1deepHiddenState_4907_train.hdf5" --test_dataroot="../TransformersNLP/results/roberta-base_trec_10epochs/1deepHiddenState_547_test.hdf5" --cuda=1 --device=1
# python mainTransformers.py --hidden_dim=1500 --l_rate=5e-5 --sigma=1e-4 --z_dim=150 --w_dim=50 --K=20 --train=1 --restore=0 --results=0 --epochs=2000 --layers=6 --batch_size=1024 --dropout=0.3 --dataset_name="roberta-base_multi30k_10epochs_1deep" --train_dataroot="../TransformersNLP/results/roberta-base_multi30k_10epochs/1deepHiddenState_29000_train.hdf5" --test_dataroot="../TransformersNLP/results/roberta-base_multi30k_10epochs/1deepHiddenState_1000_test.hdf5" --cuda=1 --device=1
# python mainTransformers.py --hidden_dim=1500 --l_rate=5e-5 --sigma=1e-4 --z_dim=150 --w_dim=50 --K=20 --train=1 --restore=0 --results=0 --epochs=4000 --layers=6 --batch_size=1024 --dropout=0.3 --dataset_name="roberta-base_sst2_10epochs_1deep" --train_dataroot="../TransformersNLP/results/roberta-base_sst2_10epochs/1deepHiddenState_6229_train.hdf5" --test_dataroot="../TransformersNLP/results/roberta-base_sst2_10epochs/1deepHiddenState_693_test.hdf5" --cuda=1 --device=2
# python mainTransformers.py --hidden_dim=1500 --l_rate=5e-5 --sigma=1e-4 --z_dim=150 --w_dim=50 --K=20 --train=1 --restore=0 --results=0 --epochs=5000 --layers=6 --batch_size=1024 --dropout=0.3 --dataset_name="xlm-roberta-base_trec_10epochs_1deep" --train_dataroot="../TransformersNLP/results/xlm-roberta-base_trec_10epochs/1deepHiddenState_4907_train.hdf5" --test_dataroot="../TransformersNLP/results/xlm-roberta-base_trec_10epochs/1deepHiddenState_547_test.hdf5" --cuda=1 --device=3
# python mainTransformers.py --hidden_dim=1500 --l_rate=5e-5 --sigma=1e-4 --z_dim=150 --w_dim=50 --K=20 --train=1 --restore=0 --results=0 --epochs=4000 --layers=6 --batch_size=1024 --dropout=0.3 --dataset_name="xlm-roberta-base_sst2_10epochs_1deep" --train_dataroot="../TransformersNLP/results/xlm-roberta-base_sst2_10epochs/1deepHiddenState_6229_train.hdf5" --test_dataroot="../TransformersNLP/results/xlm-roberta-base_sst2_10epochs/1deepHiddenState_693_test.hdf5" --cuda=1 --device=3
# python mainTransformers.py --hidden_dim=1500 --l_rate=5e-5 --sigma=1e-4 --z_dim=150 --w_dim=50 --K=20 --train=1 --restore=0 --results=0 --epochs=2000 --layers=6 --batch_size=1024 --dropout=0.3 --dataset_name="xlm-roberta-base_multi30k_10epochs_1deep" --train_dataroot="../TransformersNLP/results/xlm-roberta-base_multi30k_10epochs/1deepHiddenState_29000_train.hdf5" --test_dataroot="../TransformersNLP/results/xlm-roberta-base_multi30k_10epochs/1deepHiddenState_1000_test.hdf5" --cuda=1 --device=2

#TODO



'''  ------------------------------------------------------------------------------
                        GET PARAMETERS AND CONFIGURATION
    ------------------------------------------------------------------------------ '''
print('\n Loading parameters...')
# Obtain input parameters
args = get_args()

# capture the config path from the input arguments
config, flags = get_config_and_flags(vars(args))

# Create the experiment directories
create_dirs([config.board_dir, config.checkpoint_dir, config.result_dir])

# Save configuration arguments in a txt
save_args(args, config.board_dir)

# Set GPU configuration
if config.cuda and torch.cuda.is_available():
    torch.cuda.set_device(config.device)
else:
    print('cuda set to False')
    config.cuda = 0


print('='*100)
print('PARAMETERS AND CONFIGURATION')
for k, v in config.items(): print('- {}: {}'.format(k, v))
print('='*100 + '\n')

'''  ------------------------------------------------------------------------------
                                     GET DATA
    ------------------------------------------------------------------------------ '''
print('\n Loading data...')

#train_dataroot = '../TransformersNLP/results/dataset13/bert_base_retrained_dataset13/last_hidden_state_30000_train.hdf5'
#test_dataroot = '../TransformersNLP/results/dataset13/bert_base_retrained_dataset13/last_hidden_state_1000_test.hdf5'

training_data = TransformerOutputDataset(config.train_dataroot, option=config.option)
train_loader = DataLoader(training_data, batch_size=config.batch_size, shuffle=True, num_workers=6)


validation_data = TransformerOutputDataset(config.test_dataroot, option=config.option)
validation_loader = DataLoader(validation_data, batch_size=config.batch_size, shuffle=True, num_workers=6)


config.input_dim = training_data.dim0 * training_data.dim1

'''  -----------------------------------------------------------------------------
                        COMPUTATION GRAPH (Build the model)
    ------------------------------------------------------------------------------ '''
print('\n Building computation graph...')

GMVAE_model = GMVAE(config)


'''  -----------------------------------------------------------------------------
                                TRAIN THE MODEL
    ------------------------------------------------------------------------------ '''

if flags.train:

    print('\n Training the model...')

    GMVAE_model.train(train_loader, validation_loader)


'''  ------------------------------------------------------------------------------
                            			RESULTS
    ------------------------------------------------------------------------------ '''

if flags.results:
    print('PRODUCING RESULTS...')

    # device = torch.device('cpu')
    # model = TheModelClass(*args, **kwargs)
    # model.load_state_dict(torch.load(PATH, map_location=device))

    if flags.restore:
        GMVAE_model.restore_model()

    number_batches = 10
    train_loader = DataLoader(training_data, batch_size=config.batch_size, shuffle=False)

    # Generate training samples from bunch 0
    x_true_train, x_generated_train = GMVAE_model.random_generation(train_loader, number_batches=number_batches)

    # Save results
    print('Saving results in %s...' % (config.result_dir))
    torch.save(x_true_train, '%s/x_true_train.pt' % (config.result_dir))
    torch.save(x_generated_train, '%s/x_generated_train.pt' % (config.result_dir))

    validation_loader = DataLoader(validation_data, batch_size=config.batch_size, shuffle=False)

    # Generate test samples
    x_true_test, x_generated_test = GMVAE_model.random_generation(validation_loader, number_batches=number_batches)

    # Save results
    torch.save(x_true_test, '%s/x_true_test.pt' % (config.result_dir))
    torch.save(x_generated_test, '%s/x_generated_test.pt' % (config.result_dir))

if flags.reconstruct:
    print('PRODUCING RECONSTRUCTION...')

    # device = torch.device('cpu')
    # model = TheModelClass(*args, **kwargs)
    # model.load_state_dict(torch.load(PATH, map_location=device))

    if flags.restore:
        GMVAE_model.restore_model()

    number_batches = 10
    train_loader = DataLoader(training_data, batch_size=config.batch_size, shuffle=False)

    it = iter(train_loader)
    data_batch = it.next()

    # Reconstruct training samples from bunch 0
    x_reconstructed_train = GMVAE_model.batch_reconstruction(data_batch, K_select=config.K_select)

    print(x_reconstructed_train[0])




