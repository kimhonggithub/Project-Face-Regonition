from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from scipy.fftpack import dct
from skimage import feature
import numpy as np
import cv2

def pca_transform(X_train, X_test, n_components=None):
    pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca, pca

def lda_transform(X_train, y_train, X_test, n_components=None):
    lda = LDA(n_components=n_components)
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)
    return X_train_lda, X_test_lda
    
def lbp_transform(images, radius=3, n_points=8):
    transformed_images = []

    for image in images:
        lbp = feature.local_binary_pattern(image, n_points, radius, method='uniform')
        transformed_images.append(lbp.flatten())

    return np.array(transformed_images)

# Initialize the MSER detector
mser = cv2.MSER_create()

# Function to extract MSER features
def extract_mser_features(images):
    mser_features = []

    for image in images:
        # Convert the image to grayscale if it's not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Ensure the image is in the correct format (8-bit)
        if gray.dtype != np.uint8:
            gray = gray.astype(np.uint8)

        # Detect MSER regions
        regions, _ = mser.detectRegions(gray)

        # Extract features from the MSER regions (customize this based on your needs)
        mser_features.append(extract_features_from_regions(regions))

    return np.array(mser_features)

# Function to extract features from MSER regions (customize this based on your needs)
def extract_features_from_regions(regions):
    # For example, you might want to compute the number of detected regions
    return len(regions)

# Function to extract DCT features
def extract_dct_features(images, block_size=8):
    dct_features = []

    for image in images:
        # Check if the image is grayscale (2D) or color (3D)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)  # Add channel dimension

        # Apply 2D DCT block-wise
        m, n, _ = image.shape
        dct_coefficients = np.zeros_like(image, dtype=float)

        for i in range(0, m, block_size):
            for j in range(0, n, block_size):
                # Extract block from the image
                block = image[i:i+block_size, j:j+block_size, 0]  # Assuming single channel (grayscale)
                
                # Apply 2D DCT to the block
                dct_block = dct(dct(block, axis=0), axis=1)

                # Store the DCT coefficients in the corresponding block of the result array
                dct_coefficients[i:i+block_size, j:j+block_size, 0] = dct_block

        # Flatten the DCT coefficients and append to the list
        dct_features.append(dct_coefficients.flatten())

    return np.array(dct_features)
