# Lookup terms
**Singel-modal network**: NN learning from ine source *e.g. bloodpressure*

**Multi-modal network**: Multiple sources *e.g. heartrate, bloodpressure, temprature*

**Cross-modal networks**: Levereges from one modality to another 
*image to text*

**Knowledge distillation**: Transfering knowledge from a larger model to another smaller model

**N-shot capability**[[explanaition](https://www.pinecone.io/learn/series/image-search/zero-shot-image-classification-clip/)]: Here we define N as the number of samples required to train a model to begin making predictions in a new domain or on a new task.

**Zero-shot capability**: A learner(model) observers a sample from classes which were not observed during training

**Cosine simularity**: Used to check, if 2 vectors point in the same direction. To do this, the cosine of the angel between the vectors must be calculated.

**Gaussian Error Linear Unit(GeLU)**:
A unit which introduces non linearity in most transformers. It is rounder than ReLU and has some negativ values.

**BLEU Score**:
BLEU is an algorithm for evaluating the quality of text which has been machine-translated from one language to another.

# Scene Understanding [[Paper](https://www.mdpi.com/2076-3417/9/10/2110)]

Scene Understanding is something that to understand a scene. For instance, iPhone has function that help eye disabled person to take a photo by discribing what the camera sees.

# Transformer
Inputs get split up into tokens and get assigned to a vector with destinct values. This proceder is called encoding.

*text gets split in words, images in smaller images*

Becasue of the vector, one can imagine a image, where every token is vector in a high dimensional space. The closer the vectors are, the more similar they are. 

### Structure

**Embedding matrix**: Used to encode a token into a vector.
They also encode the position.

*The position of a word in a scentens*

**Attention Block**: 
Let's the vectors "talk" to each other. It is resposible for the meanings of the word.

*What does it refer to in a scentens*

**Multi Layer Perceptron(MLP) /Residual Block /FFL**
Feed forward layer. A linear Layer , a ReLU and a liner Layer. The resulting vector is added to the original vector(Residual connetction).

these two layers with some normalizaition are repeated until the transformer has enough capacity to execute its task.
[[ResNet description](<https://databasecamp.de/ki/resnet#:~:text=Residual%20Neural%20Networks%20(kurz%3A%20ResNet,noch%20geringe%20Fehlerraten%20hervorrufen%20kann.>)]

**Unembedding matrix**: Assigns the best fitting token to a vektor

## Attention block [[Attention Is All You Need](https://arxiv.org/pdf/1706.03762)]
The "talking" between the vector happens with querys and keys. The query ask a question and the keys answer

*Nouns ask if there are any adjectivs infront of it and the adjectivs answer*

At the end, a value is also used to further process the data.

# CLIP [[Paper](https://arxiv.org/pdf/2103.00020)]
CLIP is a pretrained model for telling you how well a given image and a given text fit together. It is special because of its zero-shot capability.

It consists of a text encoder and an image encoder. The text encoder is a transformer and the image encoder is a ViT.

[[Easy explanation](https://medium.com/one-minute-machine-learning/clip-paper-explained-easily-in-3-levels-of-detail-61959814ad13)]

[[Further explanation](https://medium.com/@paluchasz/understanding-openais-clip-model-6b52bade3fa3)]

[[ViT Paper](https://arxiv.org/pdf/2010.11929)]

Dataset: 400 M

# ALIGN [[Paper](http://proceedings.mlr.press/v139/jia21b.html)]

Similar to CLIP but trained on a larger but noisier Dataset (1.8B).

ALIGN follows the natural distribution of image-text pairs from the
raw alt-text data, while CLIP collects the dataset by first
constructing an allowlist of high-frequency visual concepts
from English Wikipedia. 

Image Encoder: EfficientNet with global pooling 

Text Encoder: BERT

# TinyCLIP [[Paper](https://arxiv.org/pdf/2309.12314)]

TinyCLIP is a method to destill large-scale language models such as CLIP.
It consists of 3 components.

## Affinity mimicking

It introduces 2 affinity distillaition losses: image-to-language loss and language-to-image loss.

## Weight inheritance

Weight inheritance is a techinque which inherits the important weights from well-trained teacher models to smaller student models.
In they paper they propose 2 methods:

### Manual inheritance

Based on observaitions of the authors. Text encoder have most redundancy in depth (layer-wise), image encoder in width (channel-wise). They select $k$ layers and channels of a branch which will function as initialization weights. To select the most important weights prior knowledge is required

### Automatic inheritance

The Authors introduce a learnable mask to identify weight importance.



## Multi-stage progressiv distillaiton 

Pruning the model over multiple stages. For each stage the authors use a modest degree of compression (e.g. 25%). Each stage includes weight inheritance and affinity mimicking.

# Optimizing LLM [[Video](https://www.youtube.com/watch?v=UcwDgsMgTu4&t=359s)]

## Quantization

Quantize the values of a LLM(e.g. from FP32 to INT8 with Zero-point Quantizaition). Quantization should be determaned after inspecting the hardware in which the model should be runned.

There are 2 ways to Quantize a network:
### Weigth quantizaition
Store weights in INT8,dequantize into FP32 when running it. Not faster inference but saves space.

### Activation quantizaition
Convert inputs and outputs to INT8 and do computaition in INT8. Need calibraition (static or dynamic) to determine scale factors for data in each layer.

## Pruning
Remove some conections in the network which results in a sparse network which is easier to store.
To use a sparse network, one has to use same sort of sparse execution engine (e.g. sparse MatMul).

### Magnitude pruning

Pick pruning factor X. In each layer, set the lowest X% of the weights (by absolute value) to zero.
Optional: retrain the model to recover accuracy

### Structured Pruning

Removing random connections in a network is called unstructured pruning.
Structured pruning is when one enforces more structure on which weights one is allowed to set to zero.
#### 2:4 Structured pruning

For each block of 4 consecutive matric values only 2 are allowed to be 0.

## Knowledge distillation (Model distillation)

First train a larger teacher network. After the teacher network has been trained, one then starts training a smaller student network to predict the output of the teacher network. This works because the output of the teacher network has more informaition than the original label.

### Advantages
- Can modify the architecture of student models (e.g. fewer layers)
- Biggest potential gain in speed
### Disadvantages
- Need to setup training data, need to run teacher model while training student
- Relatively expensive (typically 5-10% of the teacher training compute)

## Engineeting optimizations

# 
