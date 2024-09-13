# What is CNN?

CNNs, or Convolutional Neural Networks, are models used in deep learning, especially for image and video tasks. CNNs use convolution operations to find spatial and temporal patterns in images. This helps them achieve high success in tasks like object recognition, face recognition, and handwriting recognition.

![Convolution Layer](https://miro.medium.com/v2/resize:fit:1200/1*XbuW8WuRrAY5pC4t-9DZAQ.jpeg)


In a basic CNN architecture:

- **Convolution Layer**
- **Activation Layer**
- **Pooling Layer**
- **Flattening & Fully Connected (Dense) Layers**

---

## Convolution Layer

This is the main part of CNN. It is responsible for detecting features in images.

In this layer, we apply filters or kernels to the image to extract low-level and high-level features.

In CNN, images are expressed with matrices. Operations are done on these matrices. The values inside the matrices are the pixel values of the image.

In this layer, we extract features (like finding edges, corners, objects in the image) from the input image.

We do this extraction using filters or kernels.

![Convolution](https://indiantechwarrior.com/wp-content/uploads/2022/05/conv2-1.gif)


After calculating the size of the output matrix, we place the filter matrix at the top-left corner of the image matrix. We multiply and sum the overlapping values of these two matrices. This gives us the first value of our output matrix. Then we move the filter one step to the right and repeat the process. When the filter finishes moving over the image, all values of the output matrix are calculated. This way, features can be determined in the convolution layer.

**Stride** is a value that can be changed in CNN models. This value determines how many pixels the filter moves over the image.

**Padding:** If we want the input matrix and output matrix to be the same size, we need to add pixels to the input matrix. This is called padding.


---

## Activation Layer

Activation is the non-linear transformation we apply to the input signal. This transformed output is then sent as input to the next neuron layer.

We need activation functions to teach artificial neural networks complex data from the real world, like images, sounds, or videos.

Common activation functions are **ReLU**, tanh, and sigmoid. ReLU is often used because it is fast.

![Activation Layer](https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/71_blog_image_1.png)

---

## Pooling Layer

The purpose is to reduce the size of the input matrix in terms of width and height while keeping the number of channels the same. This reduces computational complexity. Therefore, it is used after the convolution layer.

It reduces the size of the feature map.

![Pooling Layer](https://miro.medium.com/v2/resize:fit:1400/1*fXxDBsJ96FKEtMOa9vNgjA.gif)


- **Max Pooling**

  After convolution, if we use max pooling, we select the maximum value among the pixel values where the filter overlaps the image. This creates a new, smaller image matrix.

- **Average Pooling**

  If we use average pooling, we take the average of the pixel values where the filter overlaps the image. This also creates a new, smaller image matrix.

**Benefits of Pooling:**

- Reduces computational cost by reducing size.
- Removes unnecessary features and highlights important ones.
- Makes the model more robust.


---

## Flattening & Fully Connected Layers

![Flattening & Fully Connected Layers](https://miro.medium.com/v2/resize:fit:1400/1*IWUxuBpqn2VuV-7Ubr01ng.png)


### Flattening Layer

Until this layer, all operations were done on matrices. To transfer these operations to the next layer and make them processable, we need to convert them into a one-dimensional vector. This is done in the flattening layer.

### Fully Connected (Dense) Layer

In this layer, the vectors from the flattening layer are taken and given as input to artificial neural networks. This starts the learning process.

In literature, it may also be called the **Dense Layer**.

After convolution and pooling layers extract features from the input image, the dense layer combines these features to make a final prediction. In a CNN, the dense layer is usually the last layer and is used to produce output predictions. The activations from previous layers are flattened and passed as input to the dense layer, which performs a weighted sum and applies an activation function to produce the final output.


---

## Pretrained CNN Models

These are models that have been prepared, trained, and optimized for various visual recognition tasks.

### Advantages of Using Pretrained Models:

- **Resource and Time Saving**
- **Performance**
- **Customization and Adaptation**

### Resources to Access These Models:

- **Keras Applications:** Contains many pretrained deep learning models. Especially good for image classification and transfer learning.

- **TorchVision:** An extension of PyTorch. Contains datasets, model structures, and pretrained models for image processing.

- **Keras CV (Computer Vision):** Handles image processing processes in a general framework. It is a library that considers the image processing process as a whole pipeline, including creating layers, data augmentation, and loss functions.

- **OpenCV DNN:** A module of OpenCV that contains modules needed for deep learning models. Suitable for performance-focused image processing and deep learning operations.

- **Hugging Face:** More focused on natural language processing, but also provides resources for image processing and object detection.

---

## Transfer Learning and Fine Tuning

![Transfer](https://media.springernature.com/lw685/springer-static/image/art%3A10.1007%2Fs11042-019-07821-9/MediaObjects/11042_2019_7821_Fig1_HTML.png)


This is the process of applying the knowledge and experience of a pretrained model from one task or dataset to a different task or dataset.

Transfer learning provides a starting point by adapting a general model to a specific task.

Fine tuning allows us to customize this model to get the best result in that task.

### How to Apply:

1. **Load a Pretrained Model:**

   - Use Keras, PyTorch, or TensorFlow to select a pretrained model (e.g., VGG, ResNet, Inception).

2. **Freeze Layers:**

   - Freeze the initial layers (set `trainable=False`) and include only the last layers in training.

3. **Add New Layers:**

   - Add layers suitable for your dataset at the end of the model.

4. **Train the Model:**

   - Train the model with your new dataset and evaluate its performance.

5. **Fine Tuning:**

   - If necessary, include more layers in training to improve the model's performance.

### Fine Tuning Techniques:

- **Feature Extractor:** Extract features using the pretrained model.

- **Full Network Fine Tuning:** Do not freeze any layers. Use the basic architecture and customize it for your dataset.

- **Freezing Layers:** Freeze some layers to prevent them from being updated during training.

- **Gradual Unfreezing:** Gradually unfreeze layers as training progresses and as you monitor the model's performance.

---

## Datasets You Can Use

### Level 1: Beginner Datasets

1. **MNIST Dataset**

   - **Description:** Images of handwritten digits (28x28 pixels).
   - **Link:** [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

2. **Fashion-MNIST**

   - **Description:** Black-and-white images of clothing items in 10 categories.
   - **Link:** [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)

3. **Kuzushiji-MNIST**

   - **Description:** Contains Japanese handwritten characters.
   - **Link:** [Kuzushiji-MNIST](https://github.com/rois-codh/kmnist)

4. **EMNIST**

   - **Description:** Extended MNIST dataset with handwritten letters and digits.
   - **Link:** [EMNIST Dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset)

5. **Sign Language MNIST**

   - **Description:** Images of American Sign Language (ASL) letters.
   - **Link:** [Sign Language MNIST](https://www.kaggle.com/datamunge/sign-language-mnist)

**My Comment:**

These beginner-level datasets are ideal for learning basic CNN applications. They are small and simple. You can train models quickly and understand basic concepts.

---

### Level 2: Intermediate Datasets

1. **CIFAR-10**

   - **Description:** Color images (32x32 pixels); contains 10 classes.
   - **Link:** [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)

2. **CIFAR-100**

   - **Description:** Similar to CIFAR-10 but with 100 classes.
   - **Link:** [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)

3. **SVHN (Street View House Numbers)**

   - **Description:** Images of real-world house numbers.
   - **Link:** [SVHN Dataset](http://ufldl.stanford.edu/housenumbers/)

4. **GTSRB (German Traffic Sign Recognition Benchmark)**

   - **Description:** Images of German traffic signs.
   - **Link:** [GTSRB](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)

5. **FER-2013 (Facial Expression Recognition)**

   - **Description:** Faces with different emotions.
   - **Link:** [FER-2013](https://www.kaggle.com/msambare/fer2013)

6. **STL-10**

   - **Description:** Larger images (96x96 pixels) with fewer labels.
   - **Link:** [STL-10](https://cs.stanford.edu/~acoates/stl10/)

7. **Dogs vs Cats**

   - **Description:** Classification of dog and cat images.
   - **Link:** [Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats)

8. **Flowers Recognition**

   - **Description:** Images of different types of flowers.
   - **Link:** [Flowers Recognition](https://www.kaggle.com/alxmamaev/flowers-recognition)

9. **Intel Image Classification**

   - **Description:** Images of different natural scenes.
   - **Link:** [Intel Image Classification](https://www.kaggle.com/puneet6060/intel-image-classification)

10. **ASL Alphabet**

    - **Description:** Images of the American Sign Language alphabet.
    - **Link:** [ASL Alphabet](https://www.kaggle.com/grassknoted/asl-alphabet)

**My Comment:**

These intermediate-level datasets have more classes and complexity. You can improve your model by using data augmentation and regularization techniques.

---

### Level 3: Advanced Datasets

1. **ImageNet**

   - **Description:** Over 1 million high-resolution images and more than 1,000 classes.
   - **Link:** [ImageNet](http://www.image-net.org/)

2. **COCO (Common Objects in Context)**

   - **Description:** Dataset for object detection, segmentation, and captioning.
   - **Link:** [COCO Dataset](https://cocodataset.org/)

3. **Open Images Dataset**

   - **Description:** Over 9 million images with many labels.
   - **Link:** [Open Images](https://storage.googleapis.com/openimages/web/index.html)

4. **Places365**

   - **Description:** Scene recognition dataset with over 10 million images.
   - **Link:** [Places365](http://places2.csail.mit.edu/)

5. **Kinetics-700**

   - **Description:** Large-scale dataset for video-based action recognition.
   - **Link:** [Kinetics-700](https://deepmind.com/research/open-source/kinetics)

6. **Cityscapes Dataset**

   - **Description:** Images for segmentation and detection in urban streets.
   - **Link:** [Cityscapes](https://www.cityscapes-dataset.com/)

7. **ADE20K**

   - **Description:** Dataset for scene understanding and segmentation.
   - **Link:** [ADE20K](http://groups.csail.mit.edu/vision/datasets/ADE20K/)

8. **VOC Dataset (PASCAL Visual Object Classes)**

   - **Description:** Used for object classification, detection, and segmentation.
   - **Link:** [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)

9. **LSUN (Large-scale Scene Understanding)**

   - **Description:** Large-scale dataset of scenes and objects.
   - **Link:** [LSUN](https://www.yf.io/p/lsun)

10. **YouTube-8M**

    - **Description:** Over 8 million videos for video classification.
    - **Link:** [YouTube-8M](https://research.google.com/youtube8m/)

**My Comment:**

These advanced-level datasets are ideal for large and complex tasks. Training can take a long time and may need powerful hardware. Using transfer learning and pretrained models is beneficial.


## Conclusion

This guide provides an overview of Convolutional Neural Networks (CNNs) and how to get started with them using various datasets and techniques like transfer learning and fine-tuning. Whether you're a beginner or an advanced practitioner, there are resources and datasets available to help you build and improve your CNN models.

By understanding the fundamental components of CNNs and practicing with different datasets, you can develop models capable of handling complex image and video tasks. Leveraging pretrained models can save time and computational resources, allowing you to focus on customizing and optimizing your models for specific applications.

---

## References

- **Deep Learning Path** on Miuul [Link](https://miuul.com/deep-learning-path)
- **CS231n: Convolutional Neural Networks for Visual Recognition** by Stanford University ([Course Website](http://cs231n.stanford.edu/))
- **Andrew Ng's Deep Learning Specialization** on Coursera ([Link](https://www.coursera.org/specializations/deep-learning))
- **Keras Documentation**: [https://keras.io/](https://keras.io/)
- **PyTorch Documentation**: [https://pytorch.org/](https://pytorch.org/)
- **TensorFlow Documentation**: [https://www.tensorflow.org/](https://www.tensorflow.org/)
- **Deep Learning Book** by Ian Goodfellow, Yoshua Bengio, and Aaron Courville ([Online Version](https://www.deeplearningbook.org/))
- **Papers with Code**: A resource for state-of-the-art papers and code implementations ([Link](https://paperswithcode.com/))
- **Awesome Deep Learning Resources**: Curated list of deep learning resources ([Link](https://github.com/ChristosChristofidis/awesome-deep-learning))
- **CNN Explainer**: Interactive visualization of CNNs ([Link](https://poloclub.github.io/cnn-explainer/))


