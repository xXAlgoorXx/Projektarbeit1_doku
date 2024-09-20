# Lookup terms
**Singel-modal network**: NN learning from ine source *e.g. bloodpressure*

**Multi-modal network**: Multiple sources *e.g. heartrate, bloodpressure, temprature*

**Cross-modal networks**: Levereges from one modality to another 
*image to text*

**Knowledge distillation**: Transfering knowledge from a larger model to another smaller model

**Zero-shot capability**: A learner(model) observers a sample from classes which were not observed during training

**Cosine simularity**: Used to check, if 2 vectors point in the same direction. To do this, the cosine of the angel between the vectors must be calculated.

**Gaussian Error Linear Unit(GeLU)**:
A unit which introduces non linearity in most transformers. It is rounder than ReLU and has some negativ values.

**BLEU Score**:
BLEU is an algorithm for evaluating the quality of text which has been machine-translated from one language to another.

# Scene Understanding [[Paper](https://www.mdpi.com/2076-3417/9/10/2110)]

Scene Understanding is something that to understand a scene. For instance, iPhone has function that help eye disabled person to take a photo by discribing what the camera sees. This is an example of Scene Understanding.

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

**Multi Layer Perceptron(MLP) / Residual Block**:
Feed forward layer. A linear Layer , a ReLU and a liner Layer. The resulting vector is added to the original vector(Residual connetction).

these two layers with some normalizaition are repeated until the transformer has enough capacity to execute its task.
[[ResNet description](<https://databasecamp.de/ki/resnet#:~:text=Residual%20Neural%20Networks%20(kurz%3A%20ResNet,noch%20geringe%20Fehlerraten%20hervorrufen%20kann.>)]

**Unembedding matrix**: Assigns the best fitting token to a vektor

## Attention block [[Attention Is All You Need](https://arxiv.org/pdf/1706.03762)]
The "talking" between the vector happens with querys and keys. The query ask a question and the keys answer

*Nouns ask if there are any adjectivs infront of it and the adjectivs answer*

At the end, a value is also used to further process the data.

# CLIP [[Paper](https://arxiv.org/pdf/2103.00020)]
CLIP is a pretrained model for telling you how well a given image and a given text fit together.

It consists of a text encoder and an image encoder. The text encoder is a transformer and the image encoder is a ViT.

[[Easy explanation](https://medium.com/one-minute-machine-learning/clip-paper-explained-easily-in-3-levels-of-detail-61959814ad13)]

[[Further explanation](https://medium.com/@paluchasz/understanding-openais-clip-model-6b52bade3fa3)]

[[ViT Paper](https://arxiv.org/pdf/2010.11929)]

# TinyCLIP [[Paper](https://arxiv.org/pdf/2309.12314)]

TinyCLIP is a method to destill large-scale language models such as CLIP.
It consists of 3 components.

### Affinity mimicking

It introduces 2 affinity distillaition losses: image-to-language loss and language-to-image loss.

### Weight inheritance

Weight inheritance is a techinque which inherits the important weights from well-trained teacher models to smaller student models.
In they paper they propose 2 methods:

- Manual inheritance
- Automatic inheritance

### Multi-stage progressiv distillaiton 


