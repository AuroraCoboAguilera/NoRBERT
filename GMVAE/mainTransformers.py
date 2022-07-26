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




