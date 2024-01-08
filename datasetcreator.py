from ecgwaveletfeatureextractor import ECGWaveletFeatureExtractor

import os


def process_ecg_file(file_name, wavelets, wavelets_level, sampto, save_path_base, window_width, offset, noise_range, plot_first_10_samples):
    """
    Process a single ECG file with wavelet feature extraction and save the results.

    Args:
        file_name (str): The name of the ECG file to process.
        wavelets (list): List of wavelet types for transformation.
        wavelets_level (int): The level of wavelet transformation.
        sampto (int): Number of samples to read, set to None for all data.
        save_path_base (str): Base directory to save the images.
        window_width (int): Width of the window for sampling.
        offset (int): Offset for non-QRS sampling.
        noise_range (int): Range for noise addition.
        plot_first_10_samples (bool): Flag to plot first 10 samples.
    """
    data_folder = os.path.join(file_name)
    save_path = os.path.join(save_path_base)
    ecg_extractor = ECGWaveletFeatureExtractor(data_folder, sampto, wavelets, wavelets_level)
    ecg_extractor.apply_wavelet_transform()
    ecg_extractor.sample_and_save(window_width, offset, noise_range, save_path, plot_first_10_samples)


# Parameters check args in the process_ecg_file docstring for information
wavelets = ['bior1.1', 'bior1.3', 'bior1.5', 'bior2.2', 'bior2.4', 'bior2.6', 'bior2.8', 'bior3.1', 'bior3.3', 'bior3.5', 'bior3.7']
wavelets_level = 5
sampto = None
save_path_base = "trainingDataset"
window_width = len(wavelets)*wavelets_level
offset = 150
noise_range = 25
plot_first_10_samples = False

# ECG file names to process
ecg_files = ["100", "101", "102"]

# Processing each file
for file_name in ecg_files:
    process_ecg_file(file_name, wavelets, wavelets_level, sampto, save_path_base, window_width, offset, noise_range, plot_first_10_samples)
