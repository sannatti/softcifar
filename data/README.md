

# Datasets and Reading Survey Data


read_data.ipynb = a jupyter notebook for reading the soft CIFAR-10 survey results

survey_answers  = a pickle file with a list of arrays of survey results and original CIFAR-10 labels 

data_batch_1    = a pickle file of CIFAR-10 1/5 training dataset with a dictionary of 
                   * b'batch_label', = 'training batch 1 of 5'
                   * b'labels'       = CIFAR-10 labels 
                   * b'data'         = CIFAR-10 images
                   * b'filenames'    = CIFAR-10 image names

batch_names    = a pickled file with CIFAR-10 of
                   * b'num_cases_per_batch' = the batch size
                   * b'label names'         = label names 

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.