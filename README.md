<img src="https://www.francetelevisions.fr/sites/default/files/styles/ftvisi_img_bandeau_x_large/public/images/2018/04/17/LEDERNIERMONET_1200.jpg?itok=OtCKJYZc" height="200" />

# I’m Something of a Painter Myself
**Objectif** : Construire un GAN qui transforme des photos en peintures de Monet. Ce travail a été réalisé dans le cadre d'une compétition Kaggle "I’m Something of a Painter Myself" à partir d'un jeu de données de 300 oeuvres de Monet au format de taille 256x256 px et de 7028 photos de même taille.

Ce projet a été exécuté sur un TPU mis à disposition par Kaggle.

## CycleGAN
Un **cycleGAN** est une architecture particulière d'un GAN. Son utilité se résume à pouvoir transférer le style d'une image X à une autre Y à partir d'image non appariées. Par exemple à partir de deux jeux de données : une banque d'images de zèbres et de chevaux, le cycleGAN permet de transformer un cheval en zèbre ou inversement. Plus concrètement, les cycleGANs sont capables de transformer une image en jour/nuit, changer la saison d'une image, rendre des photos réalistes à partir de contours, ajout ou suppression d'éléments sur une image.

La principale différence d'un cycleGAN par rapport à un GAN est qu'il possède **deux générateurs et deux discriminateurs** pour chaque style d'image. Le premier générateur Gx prend donc en entrée une image (et non plus un bruit aléatoire) et génère une image dans le style de Y. Le discriminateur Dy de style Y tente alors de classer si l'image générée est réelle (échantillonnée) ou fausse (généré par Gx). Le cycleGAN introduit donc un second générateur Fy capable de transformer une image du style Y vers le style X et son propre discriminateur Dx. Le cycleGAN encourage ainsi la **cohérence de cycle**, c'est-à-dire que lorsque qu'on traduit une image x vers un autre style yhat à partir G, alors F doit être capable de traduire yhat vers x. Par exemple, si on transforme une phrase du français vers l'anglais, puis qu'on transforme à nouveau la traduction vers le français alors on doit retrouver la phrase d'entrée. Le cycleGAN encourage la cohérence de cycle à partir de l'erreur de cohérence du cycle qui mesure la perte entre l'entrée et la sortie d'une cycle.

<img src="https://www.tensorflow.org/tutorials/generative/images/cycle_loss.png" height="150" />

La **fonction d'erreur totale** du CycleGAN s'écrit :

<img src="https://render.githubusercontent.com/render/math?math=L(G, F, D_X, D_Y)=L_{GAN}(G,D_Y, X, Y) %2B L_{GAN}(F,D_X, Y, X) %2B \lambda L_{cyc}(G, F) %2B 0.5\lambda L_{identity}(G, F)">

où Lgan désignent les pertes antagonistes des générateurs G et F vers Dx et Dy, Lcyc la perte de cohérence de cycle et Lidentity la perte d'identité qui permet de préserver les couleurs et les teintes des images transcrites. Lamdba permet d'attribuer plus de poids aux deux dernières pertes et est fixé à 10.

## Générateur
<img src="https://github.com/dimitreOliveira/MachineLearning/blob/master/Kaggle/I%E2%80%99m%20Something%20of%20a%20Painter%20Myself/generator_architecture.png?raw=true" height="300" />

J'ai adopté l'architecture du générateur à partir de celle de Johnson et al. J'utilise **6 blocs résiduels** pour les images d'entraînement de 128 × 128 et 9 blocs résiduels pour les images d'entraînement de 256 × 256 ou de plus haute résolution. 

Il se décompose en **trois parties** : un codeur, un transformateur et un décodeur. L'image d'entrée est envoyée directement dans le codeur, qui réduit la taille de la représentation tout en augmentant le nombre de canaux. Le codeur est composé de trois couches de convolution. L'activation qui en résulte est ensuite transmise au transformateur, une série de six à neuf blocs résiduels. Elle est ensuite élargie à nouveau par le décodeur, qui utilise deux convolutions de transposition pour agrandir la taille de représentation, et une couche de sortie pour produire l'image finale en RVB.

## Discriminateur

<img src="https://github.com/dimitreOliveira/MachineLearning/blob/master/Kaggle/I%E2%80%99m%20Something%20of%20a%20Painter%20Myself/discriminator_architecture.png?raw=true" height="300" />

J'utilise ici un **70x70 patchGAN** qui correspond à un réseau de neurones convolutif utilisé par Isola et al dans Image-to-Image Translation with Conditional Adversarial Nets. Cette méthode est à la fois plus efficace sur le plan du calcul et elle permet également au discriminateur de se concentrer sur des caractéristiques plus superficielles, comme la texture, qui est souvent le genre de choses qui sont modifiées lors d'une tâche de traduction d'image.

Le patchGAN est formé de **quatres couches convolutionnelles** qui divisent par deux la taille de la représentation et doublent le nombre de canaux jusqu'à ce que la taille de sortie souhaitée soit atteinte.

## Résultats

<img src="https://github.com/jordangyte/CycleGAN/blob/main/result.png?raw=true" height="450" />
Ci-dessus, trois exemples d'images générées dans le style des peintures de Monet. 
La métrique d'évaluation est la Memorization-informed Fréchet Inception Distance (MiFID) qui est une modification de la Fréchet Inception Distance. Plus cette dernière est faible, meilleur est la génération. Mon score est de **43.46** et je suis classé 43ème sur 155 au 13/11/2020 (top 28%). 

Kaggle restreint l'usage de ses TPU à 3h d'affilés maximum, ce qui limite l'apprentissage.

## Pistes d'amélioration

Voici quelques expérimentations que j'ai pu tester au cours de ce projet : 

* Architecture du générateur : ResNet plutôt que uNet [+]
* Augmentation des données [++]
* Utiliser BatchNorm plutôt que InstanceNorm [--]
* Ajouter des "skip-connexions" [+]
* Batchsize égal à 1 (SGD) [+-]
* Baisse du taux d'apprentissage après un certain nombre d'itérations [+-]
* Rognage à 128x128 avec ResNet 6 [+-]

## Références

[Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks - Original paper](https://arxiv.org/pdf/1703.10593.pdf)
[Monet CycleGAN tutorial](https://www.kaggle.com/amyjang/monet-cyclegan-tutorial)
[CycleGAN implementation in Keras](https://keras.io/examples/generative/cyclegan/)
[CycleGAN implementation in TF](https://www.tensorflow.org/tutorials/generative/cyclegan)
[CycleGAN: Learning to Translate Images (Without Paired Training Data)](https://towardsdatascience.com/cyclegan-learning-to-translate-images-without-paired-training-data-5b4e93862c8d)
[A Gentle Introduction to Cycle Consistent Adversarial Networks](https://towardsdatascience.com/a-gentle-introduction-to-cycle-consistent-adversarial-networks-6731c8424a87)
[How to Implement GAN Hacks in Keras to Train Stable Models](https://machinelearningmastery.com/how-to-code-generative-adversarial-network-hacks/)
[CycleGAN with Better Cycles](https://ssnl.github.io/better_cycles/report.pdf)
[Improving CycleGAN - Monet paintings](https://www.kaggle.com/dimitreoliveira/improving-cyclegan-monet-paintings)


