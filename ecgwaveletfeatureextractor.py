import wfdb
import numpy as np
import matplotlib.pyplot as plt
import pywt
import os
from PIL import Image

class ECGWaveletFeatureExtractor:
    """
    A class to extract wavelet features from ECG data and save them as images.
    
    Attributes:
        data_folder (str): The folder where ECG data is stored.
        sampto (int): Number of samples to read.
        wavelets (list): List of wavelet types to use for transformation.
        wavelets_level (int): The level of wavelet transformation.
        
    """
    
    def __init__(self, data_folder, sampto, wavelets, wavelets_level):
        self.data_folder = data_folder
        self.sampto = sampto
        self.wavelets = wavelets
        self.wavelets_level = wavelets_level

        self.record = wfdb.rdrecord(self.data_folder, sampto=self.sampto)
        self.annotations = wfdb.rdann(self.data_folder, 'atr', sampto=self.sampto)
        self.data = self.record.p_signal
        self.ECG1 = self.data[:, 0]
        self.ECG2 = self.data[:, 1]  # Added ECG2
        self.time_values = np.arange(len(self.ECG1), dtype=float) / self.record.fs

        self.ecgDataset1 = None
        self.ecgDataset2 = None  # Added dataset for ECG2

    def normalize_signal(self, signal):
        """
        Normalizes the ECG signal.
        
        Args:
            signal (numpy.ndarray): The ECG signal to normalize.
        
        Returns:
            numpy.ndarray: Normalized ECG signal.
        """
        signal = signal - np.mean(signal)
        return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

    def apply_wavelet_transform(self):
        """
        Applies wavelet transform to both ECG1 and ECG2 signals.
        """
        self.ECG1 = self.normalize_signal(self.ECG1)
        self.ECG2 = self.normalize_signal(self.ECG2)  # Normalize ECG2
        signal_length = len(self.ECG1)
        self.ecgDataset1 = np.zeros(((self.wavelets_level + 1) * len(self.wavelets), signal_length))
        self.ecgDataset2 = np.zeros_like(self.ecgDataset1)  # Initialize dataset for ECG2

        counter = 0
        for wavelet in self.wavelets:
            coeffs1 = pywt.wavedec(self.ECG1, wavelet, level=self.wavelets_level)
            coeffs2 = pywt.wavedec(self.ECG2, wavelet, level=self.wavelets_level)  # Wavelet transform for ECG2
            time_values = [np.linspace(0, signal_length, len(coef), endpoint=False, dtype=int) for coef in coeffs1]

            for j, (coef1, coef2) in enumerate(zip(coeffs1, coeffs2)):  # Process both ECG1 and ECG2
                coef1 = self.normalize_signal(coef1)
                coef2 = self.normalize_signal(coef2)  # Normalize ECG2 coefficients
                self.ecgDataset1[counter][time_values[j]] = coef1
                self.ecgDataset2[counter][time_values[j]] = coef2  # Apply to ECG2 dataset
                counter += 1

    def sample_and_save(self, window_width, offset, noise_range, save_path, plot_first_10_sample=True):
        """
        Samples the wavelet-transformed ECG data around QRS annotations and saves images.

        Args:
            window_width (int): Width of the window to sample around the annotation.
            offset (int): Offset to use for sampling non-QRS regions for augmentation.
            noise_range (int): Range of random noise to add for data augmentation.
            save_path (str): Directory to save the images.
            plot_first_10_sample (bool): Whether to plot the first 10 samples for visualization.
        """
        os.makedirs(os.path.join(save_path, "QRS"), exist_ok=True)
        os.makedirs(os.path.join(save_path, "notQRS"), exist_ok=True)
        
        for idx, annotation in enumerate(self.annotations.sample):
            
            # Adding noise to start of the annotation for data augmentation 
            random_noise = np.random.randint(-noise_range, noise_range + 1)
            start = annotation - window_width // 2 + random_noise

            # Process for both ECG1 and ECG2
            qrs1 = self.ECG1[start:start + window_width]
            qrs2 = self.ECG2[start:start + window_width]
            not_qrs1 = self.ECG1[(start + offset):(start + window_width + offset)]
            not_qrs2 = self.ECG2[(start + offset):(start + window_width + offset)]
            
            image1 = self.ecgDataset1[:, start:start + window_width]
            image2 = self.ecgDataset2[:, start:start + window_width]
            notqrs_image1 = self.ecgDataset1[:, (start + offset):(start + window_width+offset)]
            notqrs_image2 = self.ecgDataset2[:, (start + offset):(start + window_width+offset)]
            
            # First and last annotations seems wrong so i didnt include it
            if idx < 2 or qrs1.shape != (window_width,) or qrs2.shape != (window_width,):
                continue

            if plot_first_10_sample and idx < 10:
                plt.figure(figsize=(10, 4))
                plt.subplot(2, 4, 1)
                plt.imshow(image1)
                plt.title("ECG1 Wavelet")
                plt.subplot(2, 4, 2)
                plt.plot(self.time_values[start:start+window_width], qrs1)
                plt.title("ECG1 Raw")
                plt.subplot(2, 4, 3)
                plt.imshow(image2)
                plt.title("ECG2 Wavelet")
                plt.subplot(2, 4, 4)
                plt.plot(self.time_values[start:start+window_width], qrs2)
                plt.title("ECG2 Raw")
                plt.subplot(2, 4, 5)
                plt.imshow(notqrs_image1)
                plt.title("ECG1 Wavelet notQRS")
                plt.subplot(2, 4, 6)
                plt.plot(self.time_values[start:start+window_width], not_qrs1)
                plt.title("ECG1 Raw notQRS")
                plt.subplot(2, 4, 7)
                plt.imshow(notqrs_image2)
                plt.title("ECG2 Wavelet notQRS")
                plt.subplot(2, 4, 8)
                plt.plot(self.time_values[start:start+window_width], not_qrs2)
                plt.title("ECG2 Raw notQRS")
                plt.tight_layout()
                plt.show()

            # Save images for both ECG1 and ECG2
            for i, image in enumerate([image1, image2]):
                image_data_normalized = (255 * (image - np.min(image)) / (np.max(image) - np.min(image))).astype(np.uint8)
                img = Image.fromarray(image_data_normalized)
                img.save(os.path.join(save_path, "QRS", f'ECG_folder_{self.data_folder[-3:]}_ECG{i+1}_anno{idx}.png'))
                
            for i, image in enumerate([notqrs_image1, notqrs_image2]):
                image_data_normalized = (255 * (image - np.min(image)) / (np.max(image) - np.min(image))).astype(np.uint8)
                img = Image.fromarray(image_data_normalized)
                img.save(os.path.join(save_path, "notQRS", f'ECG_folder_{self.data_folder[-3:]}_ECG{i+1}_anno{idx}.png'))

