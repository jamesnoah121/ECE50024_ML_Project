"""

# ML Project - CGAN Reimplementation

# Baseline ML Mastery CGAN by Jason Brownlee - FMNIST
# Applied to CIFAR-10

"""

##
import keras
model = keras.models.load_model('cifar100_conditional_generator_75_epochs.h5')

##
import numpy as np
import keras
from keras.datasets.cifar10 import load_data
from keras.models import Model
from keras.utils import plot_model
from keras import models
from keras import layers
import matplotlib as mpl


from matplotlib import pyplot as plt


###

# Load the CIFAR10 Dataset and Plot some Examples

(x_train, y_train), (x_test, y_test) = load_data()

# Plot an Image from Each Class

plt.figure(figsize=(16,16))
unique_labels = np.unique(y_train)
for ii in unique_labels:
    plt.subplot(2, 5, 1 + ii)
    first_instance_class_index = np.where(y_train == ii)
    plt.imshow(x_train[first_instance_class_index[0][0]])
    plt.title(str(ii))
    ax = plt.gca()
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.savefig('CIFAR10.png')

plt.show()

## Data Loading and Simple Preprocessing Functions
def load_CIFAR_Data():
    # load dataset
    (x_train, y_train), (_, _) = load_data()  # CIFAR10

    x_train_processed = x_train.astype('float32')

    # Preprocess Pixel intensity to occupy values that support tanh function

    x_train_processed = (x_train_processed - 127.5) / 127.5  # Generator uses tanh activation so rescale

    # Since the Generator Images occupy space from -1 to 1 the Training Data should do the same

    return [x_train_processed, y_train]

## Define the Discriminator for the GAN

def build_discriminator(input_image=(32,32,3)):

	# Build the Discriminator using Keras Functional API rather than Sequential

	# Define the Inputs to the Discriminator (3 Channel Images 32x32) - 10 Classes
    in_label = keras.layers.Input(shape=(1,))

    # We will Map each Class n_classes to a different 50 element Vector Representation that will be learned by the Discriminator
    layer_stack = keras.layers.Embedding(100, 50)(in_label)  # Embed 100 Class Representation into Vector of 50

	# Scale up to the Image Size
    layer_stack = keras.layers.Dense(input_image[0] * input_image[1])(layer_stack)  # Learn Representation Image as Flattened Vector

	# Reshape to Image Size
    layer_stack = keras.layers.Reshape((input_image[0], input_image[1], 1))(layer_stack)  # 32x32x1

    input_image = keras.Input(shape=input_image)

    # Image Input Size
    concatenated_input = keras.layers.Concatenate()([input_image , layer_stack])  # Stacked 32x32x3 (Image) + 32x32x1 (Embedded Representation of Label)

    # Similar to Traditional GAN (CNN) - Downsample Images
    model_f = keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(concatenated_input)  # 16x16x128
    model_f = keras.layers.LeakyReLU(alpha=0.2)(model_f)

    # Downsample Images
    model_f = keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(model_f)  # 8x8x128
    model_f = keras.layers.LeakyReLU(alpha=0.2)(model_f)

    # Apply Flattening Layer
    model_f = keras.layers.Flatten()(model_f)  # 8192  (8*8*128=8192)

    # Apply Dropout Layer
    model_f = keras.layers.Dropout(0.4)(model_f)

    # Output of Generator
    out_layer = keras.layers.Dense(1, activation='sigmoid')(model_f) # Needs to be Binary -> Determine if Image is Real or Not

    # Define the Full Discriminator Model
    model = keras.Model([input_image, in_label], out_layer) # Inputs are concatenated image and label -> Out_Layer is the Entire Architecture

    # Compile with Adam Optimizer
    opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model

discriminator_model = build_discriminator()
print(discriminator_model.summary())
# plot_model(discriminator_model, to_file='discriminator_model.png')

## Define the Generator for the GAN, should receive Latent Dimension and Input Label

def build_generator(latent_dim):

    in_label = keras.layers.Input(shape=(1,)) # Similar to Discriminator

	# Each Label will be represented as a vector of 50
    layer_stack = keras.layers.Embedding(100, 50)(in_label)  # Shape 100,50

    # Linear Multiplication - > Since Input images are 32x32 and they are Downsampled Twice -> Need 8x8
    layer_stack = keras.layers.Dense(64)(layer_stack)  # 1,64

    # Reshape
    layer_stack = keras.layers.Reshape((8, 8, 1))(layer_stack)

    # Create Generator Input Layer
    in_layer = keras.layers.Input(shape=(latent_dim,))  # Input Vector with Dimension 100

    # Now we will Create the 8x8 Image and Upscale to 32x32 for Output
    # foundation for 8x8 image

    image_dim = [8,8]

    model_gn = keras.layers.Dense(128 * image_dim[0] * image_dim[1])(in_layer)  # Will be a 128 x 8 x 8 Dense Layer
    model_gn = keras.layers.LeakyReLU(alpha=0.2)(model_gn)
    model_gn = keras.layers.Reshape((8, 8, 128))(model_gn)  # Rehape to 8 x 8 x 128

    concatenated_input = keras.layers.Concatenate()([model_gn, layer_stack])  # Shape=8x8x129 ( Will Input includes the Labels for Conditioning )

    # Upsample Images 16x16
    gn = keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(concatenated_input)  # 16x16x128
    gn = keras.layers.LeakyReLU(alpha=0.2)(gn)

    # Upsample Images 16x16
    gn = keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(gn)  # 32x32x128
    gn = keras.layers.LeakyReLU(alpha=0.2)(gn)

    # Output Layer
    out_layer = keras.layers.Conv2D(3, (8, 8), activation='tanh', padding='same')(gn)  # 32x32x3

    model = keras.Model([in_layer, in_label], out_layer)

    # We do not compile the model as it isn't trained like we train the discriminator.

    return model

latent_dim = 100

generator_model = build_generator(latent_dim)
print(generator_model.summary())
# plot_model(generator_model, to_file='generator_model.png')

##

# Define the GAN by combining the Generator and Discriminator, We will hold the Discriminator Constant while
# Generator is being trained.

def build_CGAN(generator_model, discriminator_model):
    discriminator_model.trainable = False  # Discriminator is trained separately. So set to not trainable.

    # Get Size Constraints of Generator
    generator_noise, generator_label = generator_model.input  # Latent Vector + Label Size

    # Get Image Output Size Constraints Generator
    generator_output = generator_model.output

    # generator image output and corresponding input label are inputs to discriminator

    # Generated Image and Input Label are input to Discriminator
    CGAN_Output = discriminator_model([generator_output, generator_label])

    # Send Latent Noise Vector plus Label as Inputs
    model = keras.Model([generator_noise, generator_label], CGAN_Output)
    # compile model
    opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

CGAN_model = build_CGAN(generator_model,discriminator_model)
print(CGAN_model.summary())
# plot_model(CGAN_model, to_file='CGAN_Model.png')


## Need a function to Generate Random Latent Vector Points for the Generator

def generate_generator_inputs(latent_dim, n_samples):

    # Generate Random Latent Points for N-Samples
    latent_points = np.random.randn(latent_dim * n_samples)

    # Reshape for Network
    generator_input = latent_points.reshape(n_samples, latent_dim)

    # Generate Random Labels for the Inputs up to the True Number of Classes in Dataset
    labels = np.random.randint(0, 100, n_samples)

    return [generator_input, labels]

## Need a function to Sample Real Dataset - Assign Label of 1 to all these Samples indicating
# they are real images.

def sample_real_dataset(dataset, n_samples):

    # Split TF Dataset into Images + Labels
    images, labels = dataset

    # Select N Random Samples
    random_index = np.random.randint(0, images.shape[0], n_samples)

    # X_Samples
    X_Samples, Sampled_Labels = images[random_index], labels[random_index]

    # Labels for Binary (Real, Non-Real) Classifier  == 1 (REAL)
    Y = np.ones((n_samples, 1))  # Label=1 indicating they are real

    return [X_Samples, Sampled_Labels], Y

## Function to Generate Random Examples with Labels

def generate_fake_samples(generator, latent_dim, n_samples):

    # Generate Latent Sample Points for N Samples
    generator_input, labels = generate_generator_inputs(latent_dim, n_samples)

    # Predict Outputs of the Trained Generator
    images = generator.predict([generator_input, labels])

    # Automatically Label Generated Samples as Fake
    Y = np.zeros((n_samples, 1))  # Label=0 indicating they are fake

    return [images, labels], Y

# Steps to Train CGAN
# 1.) Select Random Images/Labels from Real Dataset
# 2.) Generate Set of Images/Labels from Generator
# 3.) Feed both sets into Discriminator
# 4.) Set Loss for Real and Fake and Combined Loss

r_loss = []
f_loss = []
model_loss = []

def train(generator_model, discriminator_model, CGAN_Model, real_dataset, latent_dim, n_batch=128):

    # Compute the Number of Batches per Epoch by [Size Dataset / Size Batch]
    n_batches_in_epoch = int(dataset[0].shape[0] / n_batch)

    # Epoch Loop
    for i in range(75):

        print('EPOCH: ' + str(i) )
        # Create a Dataset of 64 Fake and Real Images
        for j in range(n_batches_in_epoch):

            # Sample from the True Dataset
            [X_Real, Labels_Real], Y_Real = sample_real_dataset(real_dataset, int(n_batch / 2))

            discriminator_real_loss , _ = discriminator_model.train_on_batch([X_Real, Labels_Real], Y_Real)

            r_loss.append(discriminator_real_loss)

            # Generate Samples from the Generator
            [X_Fake, Fake_labels], Y_Fake = generate_fake_samples(generator_model, latent_dim, int(n_batch / 2))

            # Train Discriminator on Fake Data
            discriminator_fake_loss, _ = discriminator_model.train_on_batch([X_Fake, Fake_labels], Y_Fake)

            f_loss.append(discriminator_fake_loss)

            # Prepare Latent Point Vectors for Generator
            [generator_input, labels] = generate_generator_inputs(latent_dim, n_batch)

            # The Generator wants the Discriminator to Labels images as 1 or REAL
            Y_GAN = np.ones((n_batch, 1))

            GAN_Loss = CGAN_Model.train_on_batch([generator_input, labels], Y_GAN)

            model_loss.append(GAN_Loss)

    # Save the Generator Model
    generator_model.save('cifar100_conditional_generator_75_epochs.h5')
    generator_model.save('./drive/MyDrive/ML_Project/cifar10_conditional_generator_75_epochs.h5')


## Train the CGAN
# Initialize the Latent Size Dimension
latent_dim = 100
# Build the Discriminator using Keras Functional API
discriminator_model = build_discriminator()
# Build the Generator using Keras Functional API
generator_model = build_generator(latent_dim)
# Combine both Models into CGAN
CGAN_Model = build_CGAN(generator_model, discriminator_model)

# Load the Real CIFAR100 Dataset
dataset = load_CIFAR_Data()
# train model
train(generator_model, discriminator_model, CGAN_Model, dataset, latent_dim)



