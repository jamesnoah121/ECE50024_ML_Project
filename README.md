# ECE50024_ML_Project
CGAN_Project_ML_50024

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:1000}"
executionInfo="{&quot;elapsed&quot;:8598,&quot;status&quot;:&quot;error&quot;,&quot;timestamp&quot;:1682560181802,&quot;user&quot;:{&quot;displayName&quot;:&quot;Noah James&quot;,&quot;userId&quot;:&quot;10101508046660084723&quot;},&quot;user_tz&quot;:240}"
id="0aTv_GEJRx7P" outputId="aa12c302-dea2-4fbc-d2f2-d46492cb8122">

``` python
import tensorflow as tf

model = tf.keras.saving.load_model('./cifar10_conditional_generator_200_epochs.h5')

# generate multiple images

latent_points, labels = generate_generator_inputs(100, 100)
# specify labels - generate 10 sets of labels each gping from 0 to 9
labels = np.asarray([x for _ in range(10) for x in range(10)])
# generate images

print(latent_points.shape)
print(labels.shape)
X  = model.predict([latent_points, labels])
# scale from [-1,1] to [0,1]
X = (X + 1) / 2.0
X = (X*255).astype(np.uint8)
# plot the result (10 sets of images, all images in a column should be of same class in the plot)
# Plot generated images 

plt.figure(figsize=(20,20))
def show_plot(examples, n):
	for i in range(n * n):
		plt.subplot(n, n, 1 + i)
		plt.axis('off')
		plt.imshow(examples[i, :, :, :])
		plt.title(labels[i])
	
	
	plt.show()
    
show_plot(X, 10)
plt.savefig('Samples_with_Conditioning_200_Epochs.png')

## Plot the Discriminator and Generator Losses
plt.figure(figsize=(12,8))
plt.plot(g_losses,label='Generator Loss')
plt.plot(d_losses,label='Discriminator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.title('Discriminator vs. Generator Loss for CIFAR10 Training')
plt.legend()
plt.show()
plt.savefig('Losses_Plot_200_Epochs.png')

```

<div class="output stream stderr">

    WARNING:tensorflow:No training configuration found in the save file, so the model was *not* compiled. Compile it manually.

</div>

<div class="output stream stdout">

    (100, 100)
    (100,)
    4/4 [==============================] - 0s 7ms/step

</div>

<div class="output display_data">

![](db970a846dad7febe254880188e48790099ea536.png)

</div>

<div class="output error" ename="NameError" evalue="ignored">

    ---------------------------------------------------------------------------
    NameError                                 Traceback (most recent call last)
    <ipython-input-38-88ba5b24ee4b> in <cell line: 37>()
         35 ## Plot the Discriminator and Generator Losses
         36 plt.figure(figsize=(12,8))
    ---> 37 plt.plot(g_losses,label='Generator Loss')
         38 plt.plot(d_losses,label='Discriminator Loss')
         39 plt.xlabel('Epoch')

    NameError: name 'g_losses' is not defined

</div>

<div class="output display_data">

    <Figure size 640x480 with 0 Axes>

</div>

<div class="output display_data">

    <Figure size 1200x800 with 0 Axes>

</div>

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
executionInfo="{&quot;elapsed&quot;:611091,&quot;status&quot;:&quot;ok&quot;,&quot;timestamp&quot;:1682563648691,&quot;user&quot;:{&quot;displayName&quot;:&quot;Noah James&quot;,&quot;userId&quot;:&quot;10101508046660084723&quot;},&quot;user_tz&quot;:240}"
id="_HooUXolU85C" outputId="8e939619-55a8-4ce3-d53f-ae29a3a03d7f">

``` python
from skimage.transform import resize
import scipy

## Evaluate all the models and generated pictures vs. each Label using the Frechet Inception Distance (adapted from MLMastery)

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input

g_model = tf.keras.saving.load_model('./cifar10_conditional_generator_200_epochs.h5')

model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3)) # We will have to Resize Images so they may be evaluated by InceptionV3

(x_train, y_train), (x_test, y_test) = load_data() # Load the CIFAR10 Data

# For the purpose of CGAN just checking FID is not enough, we want to do an evaluation per class to see how well the network was conditioned for labels. We will compare to the images in the training set.

unique_labels = np.unique(y_train)


# FID Calculator
def calculate_fid(model, images1, images2):
	# calculate activations
	act1 = model.predict(images1)
	act2 = model.predict(images2)
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = np.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if np.iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid

# Image array scaling function
def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return np.asarray(images_list)


FID_SCORES = np.zeros((3,len(unique_labels)))

for jj in range(3):

    if jj == 0:
      g_model = tf.keras.saving.load_model('./cifar10_conditional_generator_mode_collapse.h5')
      print('cGAN MODE COLLAPSE FID SCORES:')
    if jj == 1:
      g_model = tf.keras.saving.load_model('./cifar10_conditional_generator_75_epochs.h5')
      print('cGAN 75 EPOCH TUNED FID SCORES:')
    if jj == 2:
      g_model = tf.keras.saving.load_model('./cifar10_conditional_generator_200_epochs.h5')
      print('cGAN 200 EPOCH TUNED FID SCORES:')

    for ii in unique_labels:
        match_class_index = np.where(y_train == ii)
        this_class_images = x_train[match_class_index[0]]
        np.random.shuffle(this_class_images)
        this_class_images = this_class_images[:500] # Only Compare 500 Real and 500 Fake
        print(this_class_images.shape)

        # Generate Fake Samples from same Class
        latent_points, labels = generate_generator_inputs(100, len(this_class_images))
        # Specify Labels
        labels = np.ones(len(this_class_images)) * ii
        # Generate Images
        X  = g_model.predict([latent_points, labels])
        # Scale Images to [0,1]
        X = (X + 1) / 2.0
        X = (X*255).astype(np.uint8)
        # Convert to Floating Point
        images_real = this_class_images.astype('float32')
        images_fake = X.astype('float32')

        print(X.shape)
        # Resize Images
        images_real = scale_images(images_real,(299,299,3))
        images_fake = scale_images(images_fake,(299,299,3))

        images_real = preprocess_input(images_real)  
        images_fake = preprocess_input(images_fake)

        # calculate fid
        fid = calculate_fid(model, images_real, images_fake)
        print('CLASS ' + str(ii))
        print('FID: %.3f' % fid)
        FID_SCORES[jj][ii] = fid


print(FID_SCORES)
```

<div class="output stream stderr">

    WARNING:tensorflow:No training configuration found in the save file, so the model was *not* compiled. Compile it manually.
    WARNING:tensorflow:No training configuration found in the save file, so the model was *not* compiled. Compile it manually.

</div>

<div class="output stream stdout">

    cGAN MODE COLLAPSE FID SCORES:
    (500, 32, 32, 3)
    16/16 [==============================] - 0s 7ms/step
    (500, 32, 32, 3)
    16/16 [==============================] - 3s 108ms/step
    16/16 [==============================] - 2s 103ms/step
    CLASS 0
    FID: 336.402
    (500, 32, 32, 3)
    16/16 [==============================] - 0s 7ms/step
    (500, 32, 32, 3)
    16/16 [==============================] - 2s 109ms/step
    16/16 [==============================] - 2s 104ms/step
    CLASS 1
    FID: 346.599
    (500, 32, 32, 3)
    16/16 [==============================] - 0s 7ms/step
    (500, 32, 32, 3)
    16/16 [==============================] - 2s 108ms/step
    16/16 [==============================] - 2s 104ms/step
    CLASS 2
    FID: 295.096
    (500, 32, 32, 3)
    16/16 [==============================] - 0s 7ms/step
    (500, 32, 32, 3)
    16/16 [==============================] - 2s 110ms/step
    16/16 [==============================] - 2s 103ms/step
    CLASS 3
    FID: 315.103
    (500, 32, 32, 3)
    16/16 [==============================] - 0s 7ms/step
    (500, 32, 32, 3)
    16/16 [==============================] - 2s 107ms/step
    16/16 [==============================] - 2s 104ms/step
    CLASS 4
    FID: 272.295
    (500, 32, 32, 3)
    16/16 [==============================] - 0s 7ms/step
    (500, 32, 32, 3)
    16/16 [==============================] - 2s 109ms/step
    16/16 [==============================] - 2s 104ms/step
    CLASS 5
    FID: 311.634
    (500, 32, 32, 3)
    16/16 [==============================] - 0s 7ms/step
    (500, 32, 32, 3)
    16/16 [==============================] - 2s 108ms/step
    16/16 [==============================] - 2s 104ms/step
    CLASS 6
    FID: 257.361
    (500, 32, 32, 3)
    16/16 [==============================] - 0s 7ms/step
    (500, 32, 32, 3)
    16/16 [==============================] - 2s 109ms/step
    16/16 [==============================] - 2s 105ms/step
    CLASS 7
    FID: 334.465
    (500, 32, 32, 3)
    16/16 [==============================] - 0s 7ms/step
    (500, 32, 32, 3)
    16/16 [==============================] - 2s 108ms/step
    16/16 [==============================] - 2s 104ms/step
    CLASS 8
    FID: 317.615
    (500, 32, 32, 3)
    16/16 [==============================] - 0s 7ms/step
    (500, 32, 32, 3)
    16/16 [==============================] - 2s 110ms/step
    16/16 [==============================] - 2s 104ms/step
    CLASS 9
    FID: 329.758

</div>

<div class="output stream stderr">

    WARNING:tensorflow:No training configuration found in the save file, so the model was *not* compiled. Compile it manually.

</div>

<div class="output stream stdout">

    cGAN 75 EPOCH TUNED FID SCORES:
    (500, 32, 32, 3)
    16/16 [==============================] - 0s 7ms/step
    (500, 32, 32, 3)
    16/16 [==============================] - 2s 108ms/step
    16/16 [==============================] - 2s 104ms/step
    CLASS 0
    FID: 144.552
    (500, 32, 32, 3)
    16/16 [==============================] - 0s 7ms/step
    (500, 32, 32, 3)
    16/16 [==============================] - 2s 108ms/step
    16/16 [==============================] - 2s 105ms/step
    CLASS 1
    FID: 164.511
    (500, 32, 32, 3)
    16/16 [==============================] - 0s 7ms/step
    (500, 32, 32, 3)
    16/16 [==============================] - 2s 108ms/step
    16/16 [==============================] - 2s 105ms/step
    CLASS 2
    FID: 154.471
    (500, 32, 32, 3)
    16/16 [==============================] - 0s 7ms/step
    (500, 32, 32, 3)
    16/16 [==============================] - 2s 110ms/step
    16/16 [==============================] - 2s 104ms/step
    CLASS 3
    FID: 139.001
    (500, 32, 32, 3)
    16/16 [==============================] - 0s 7ms/step
    (500, 32, 32, 3)
    16/16 [==============================] - 2s 109ms/step
    16/16 [==============================] - 2s 104ms/step
    CLASS 4
    FID: 117.340
    (500, 32, 32, 3)
    16/16 [==============================] - 0s 7ms/step
    (500, 32, 32, 3)
    16/16 [==============================] - 2s 108ms/step
    16/16 [==============================] - 2s 105ms/step
    CLASS 5
    FID: 151.307
    (500, 32, 32, 3)
    16/16 [==============================] - 0s 7ms/step
    (500, 32, 32, 3)
    16/16 [==============================] - 2s 110ms/step
    16/16 [==============================] - 2s 105ms/step
    CLASS 6
    FID: 117.945
    (500, 32, 32, 3)
    16/16 [==============================] - 0s 7ms/step
    (500, 32, 32, 3)
    16/16 [==============================] - 2s 109ms/step
    16/16 [==============================] - 2s 104ms/step
    CLASS 7
    FID: 160.527
    (500, 32, 32, 3)
    16/16 [==============================] - 0s 7ms/step
    (500, 32, 32, 3)
    16/16 [==============================] - 2s 109ms/step
    16/16 [==============================] - 2s 105ms/step
    CLASS 8
    FID: 141.108
    (500, 32, 32, 3)
    16/16 [==============================] - 0s 7ms/step
    (500, 32, 32, 3)
    16/16 [==============================] - 2s 109ms/step
    16/16 [==============================] - 2s 105ms/step
    CLASS 9
    FID: 161.552

</div>

<div class="output stream stderr">

    WARNING:tensorflow:No training configuration found in the save file, so the model was *not* compiled. Compile it manually.

</div>

<div class="output stream stdout">

    cGAN 200 EPOCH TUNED FID SCORES:
    (500, 32, 32, 3)
    16/16 [==============================] - 0s 7ms/step
    (500, 32, 32, 3)
    16/16 [==============================] - 2s 108ms/step
    16/16 [==============================] - 2s 105ms/step
    CLASS 0
    FID: 135.885
    (500, 32, 32, 3)
    16/16 [==============================] - 0s 7ms/step
    (500, 32, 32, 3)
    16/16 [==============================] - 2s 109ms/step
    16/16 [==============================] - 2s 105ms/step
    CLASS 1
    FID: 145.301
    (500, 32, 32, 3)
    16/16 [==============================] - 0s 7ms/step
    (500, 32, 32, 3)
    16/16 [==============================] - 2s 108ms/step
    16/16 [==============================] - 2s 105ms/step
    CLASS 2
    FID: 141.674
    (500, 32, 32, 3)
    16/16 [==============================] - 0s 7ms/step
    (500, 32, 32, 3)
    16/16 [==============================] - 2s 110ms/step
    16/16 [==============================] - 2s 105ms/step
    CLASS 3
    FID: 128.871
    (500, 32, 32, 3)
    16/16 [==============================] - 0s 7ms/step
    (500, 32, 32, 3)
    16/16 [==============================] - 2s 110ms/step
    16/16 [==============================] - 2s 105ms/step
    CLASS 4
    FID: 112.269
    (500, 32, 32, 3)
    16/16 [==============================] - 0s 7ms/step
    (500, 32, 32, 3)
    16/16 [==============================] - 2s 107ms/step
    16/16 [==============================] - 2s 105ms/step
    CLASS 5
    FID: 147.482
    (500, 32, 32, 3)
    16/16 [==============================] - 0s 7ms/step
    (500, 32, 32, 3)
    16/16 [==============================] - 2s 109ms/step
    16/16 [==============================] - 2s 105ms/step
    CLASS 6
    FID: 105.914
    (500, 32, 32, 3)
    16/16 [==============================] - 0s 7ms/step
    (500, 32, 32, 3)
    16/16 [==============================] - 2s 109ms/step
    16/16 [==============================] - 2s 104ms/step
    CLASS 7
    FID: 144.395
    (500, 32, 32, 3)
    16/16 [==============================] - 0s 7ms/step
    (500, 32, 32, 3)
    16/16 [==============================] - 2s 109ms/step
    16/16 [==============================] - 2s 105ms/step
    CLASS 8
    FID: 136.754
    (500, 32, 32, 3)
    16/16 [==============================] - 0s 7ms/step
    (500, 32, 32, 3)
    16/16 [==============================] - 2s 108ms/step
    16/16 [==============================] - 2s 105ms/step
    CLASS 9
    FID: 137.082
    [[336.40181244 346.59889843 295.09602724 315.1034405  272.29533133
      311.63431596 257.36072756 334.46489741 317.61490182 329.75801644]
     [144.55175631 164.51067675 154.47103504 139.00133978 117.34049876
      151.30665276 117.94460743 160.52708525 141.10762638 161.55248123]
     [135.88503546 145.30120965 141.67389442 128.87094983 112.26902894
      147.48248244 105.91413451 144.39491535 136.75424028 137.08195853]]

</div>

</div>
