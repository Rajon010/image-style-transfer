# Exploring Maximum-Mean-Discrepancy-Based Descriptive Neural Image Style Transfer Using PyTorch

## How to run

`./download_models.sh` provided by [9].  
`./download_images.sh`  
`python3 neural_image_style_transfer.py` built upon [9].

## Abstract

Since Gatys et al. proposed an image style transfer algorithm using convolution neural network (CNN) in 2015 ([1.1], [1.2]), many proceeding works extend their method and produce various results. Jing et al ([2]) suggested that these methods be divided into 3 categories: Maximum-Mean-Discrepancy(MMD)-based Descriptive Methods, Markov-Random-Field(MRF)-based Descriptive Methods, and Generative Methods. In this project we focus on MMD-based methods, which achieves style transfer by gradient descent on an output image. We reproduce the result of [1.1], implement the idea "activation shift" in [3] to improve transfering quality, and then give furthur insights into the arguments in [4] by both mathematical proofs and experiments.

## Summary of Gatys et al.'s method

We briefly review the algorithm in [1.1] in order to set up the mathematical notations, which is mainly adapted from [4] and [5], and will be used throughout this project.

Given a content image x<sub>c</sub> and a style image x<sub>s</sub>, we would like to generate an output image x<sub>o</sub> consisting of the content of x<sub>c</sub> and the style of x<sub>s</sub>. x<sub>o</sub> shares the same height and width with x<sub>c</sub> and is initialized to contain the same pixel values as x<sub>c</sub>. Let x<sub>c</sub>, x<sub>s</sub>, and x<sub>o</sub> be passed through a pretrained VGG19 CNN ([6]). We denote the heights of feature maps of x<sub>c</sub>, x<sub>s</sub>, and x<sub>o</sub> in layer l of the CNN by h<sub>c</sub><sup>l</sup>, h<sub>s</sub><sup>l</sup>, and h<sub>o</sub><sup>l</sup>, respectively, and the widths are denoted by w<sub>c</sub><sup>l</sup>, w<sub>s</sub><sup>l</sup>, and w<sub>o</sub><sup>l</sup>. Let m<sub>z</sub><sup>l</sup> = h<sub>z</sub><sup>l</sup> &times; w<sub>z</sub><sup>l</sup>, &forall; z &isin; {c, s, o}. Also let n<sup>l</sup> be the number of filters in layer l. We then denote the rearranged feature maps of x<sub>c</sub>, x<sub>s</sub>, and x<sub>o</sub> in layer l by matrices P<sup>l</sup> &isin; M<sub>n<sup>l</sup>&times;m<sub>c</sub><sup>l</sup></sub>, S<sup>l</sup> &isin; M<sub>n<sup>l</sup>&times;m<sub>s</sub><sup>l</sup></sub>, and F<sup>l</sup> &isin; M<sub>n<sup>l</sup>&times;m<sub>o</sub><sup>l</sup></sub>, respectively. P<sup>l</sup><sub>i,(w<sub>c</sub><sup>l</sup>&times;k+q+1)</sub>, 0 &le; k < h<sub>c</sub><sup>l</sup> and 0 &le; q < w<sub>c</sub><sup>l</sup>, is the value of (k+1)-th row and (q+1)-th column of i-th feature map of x<sub>c</sub> in layer l. S<sup>l</sup> and F<sup>l</sup> are viewed similarly. We minimize the loss function L = L<sub>c</sub> + L<sub>s</sub> to achieve style transfer by back propagation through CNN and gradient descent on x<sub>o</sub>, where  
L<sub>c</sub> = &Sigma;<sub>l</sub><sup></sup> <sup>a<sup>l</sup></sup>&frasl;<sub>n<sup>l</sup>m<sub>c</sub><sup>l</sup></sub> &Sigma;<sub>i=1</sub><sup>n<sup>l</sup></sup> &Sigma;<sub>k=1</sub><sup>m<sub>c</sub><sup>l</sup></sup> (F<sup>l</sup><sub>ik</sub> - P<sup>l</sup><sub>ik</sub>)<sup>2</sup> is the content loss, and  
L<sub>s</sub> = &Sigma;<sub>l</sub><sup></sup> <sup>b<sup>l</sup></sup>&frasl;<sub>(n<sup>l</sup>)<sup>2</sup></sub> &Sigma;<sub>i=1</sub><sup>n<sup>l</sup></sup> &Sigma;<sub>j=1</sub><sup>n<sup>l</sup></sup> (G<sup>l</sup><sub>ij</sub> - A<sup>l</sup><sub>ij</sub>)<sup>2</sup> is the style loss, where  
A<sup>l</sup> = <sup>1</sup>&frasl;<sub>m<sub>s</sub><sup>l</sup></sup></sub>(S<sup>l</sup>)(S<sup>l</sup>)<sup>T</sup> and G<sup>l</sup> = <sup>1</sup>&frasl;<sub>m<sub>o</sub><sup>l</sup></sub>(F<sup>l</sup>)(F<sup>l</sup>)<sup>T</sup> are averaged Gramian matrices, and  
a<sup>l</sup> and b<sup>l</sup> are user-specified weights.

In the context below, we omit the superscript l when not needed.

The meaning of Gramian matrices is clearly pointed out in [7], section 1: A<sub>ij</sub> = <sup>1</sup>&frasl;<sub>m<sub>s</sub></sup></sub> &Sigma;<sub>k=1</sub><sup>m<sub>s</sub></sub></sup> S<sub>ik</sub>S<sub>jk</sub> indicates how often the i-th and j-th features appears together. G is understood similarly.

The content loss takes care of "where" a feature appears in a feature map. On the other hand, the style loss emphasizes "how often" distinct features appear together but does not care "where" they co-occur.

### Remarks

1. Though [1.1] assumes that the heights and weights of x<sub>c</sub> and x<sub>s</sub> are identical, they are allowed to differ. There is no such requirement in [5].
2. In literature, Gramian matrix comes in two forms: A = SS<sup>T</sup> (others) and A = S<sup>T</sup>S ([5]). The former is more consistent with APIs of deep learning libraries while the latter is the standard form ([10]). We use the former one.

## Images

In this project we use 3 content images and 3 style images as inputs and get 3 &times; 3 = 9 output images in each experiment.

### Content images ([C.1][C.2][C.3])

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/Beethoven.jpg/1200px-Beethoven.jpg" height="200"> <img src="http://www.newlooktravel.tn/photo/3-80-NewLookTravel_Paris_Cathedrale-Notre-Dame-nuit.jpg" height="200"> <img src="Images/fate.jpg" height="200">

### Style images ([S.1][S.2][S.3])

<img src="https://raw.githubusercontent.com/leongatys/PytorchNeuralStyleTransfer/master/Images/vangogh_starry_night.jpg" height="200"> <img src="http://www.neurosoup.com/wp-content/uploads/2016/08/2012-fawkes-randal-roberts_580.jpg" height="200"> <img src="http://eskipaper.com/images/ice-5.jpg" height="200">

## Part 0: Reproduce Gatys et al.'s results

Gatys kindly provides his implementation of [1.1] on [9]. **All the extensions below are built upon Gatys' code.**

In this part, we simply run Gatys' code to generate style-transferred images. It can be observed that there is "ghosting" in some results, as stated in [8].

<img src="Results/beethoven_starry_night_gramian_0.0.jpg" height="200"> <img src="Results/church_starry_night_gramian_0.0.jpg" height="200"> <img src="Results/fate_starry_night_gramian_0.0.jpg" height="200">
<img src="Results/beethoven_face_gramian_0.0.jpg" height="200"> <img src="Results/church_face_gramian_0.0.jpg" height="200"> <img src="Results/fate_face_gramian_0.0.jpg" height="200">
<img src="Results/beethoven_ice_gramian_0.0.jpg" height="200"> <img src="Results/church_ice_gramian_0.0.jpg" height="200"> <img src="Results/fate_ice_gramian_0.0.jpg" height="200">

## Part 1: Remove ghosting using activation shift

Risser et al. argue that ghosting occurs because there are multiple sets of pixel values that can produce alike feature maps when passed through CNN, and some of the sets looks like ghosting ([8]). Novak et al. give related argument in [3], section 3.3 and suggest that using "activation shift" can reduce the ambiguity of candidate sets of pixel values. Their modification is that: instead of letting  
A = <sup>1</sup>&frasl;<sub>m<sub>s</sub></sup></sub>SS<sup>T</sup> and G = <sup>1</sup>&frasl;<sub>m<sub>o</sub></sub>FF<sup>T</sup>,  
let  
A = <sup>1</sup>&frasl;<sub>m<sub>s</sub></sup></sub>(S + sU)(S + sU)<sup>T</sup> and G = <sup>1</sup>&frasl;<sub>m<sub>o</sub></sub>(F + sU)(F + sU)<sup>T</sup>, where  
s is a user-specified scalar and U the all one matrix (with size varying to meet the need).

The explanation of this modification is provided in [3], which we are not going to restate here. Here we simply implement it and examine its performance with s varying. It can be ovserved that the ghosting disappears when |s| increases. **Activation shift removes ghosting.**

### Gramian matrix with activation shift. Value of s from top to bottom: -600, -500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500, 600.

<img src="Results/beethoven_starry_night_gramian_-600.0.jpg" height="75"> <img src="Results/church_starry_night_gramian_-600.0.jpg" height="75"> <img src="Results/fate_starry_night_gramian_-600.0.jpg" height="75"> <img src="Results/beethoven_face_gramian_-600.0.jpg" height="75"> <img src="Results/church_face_gramian_-600.0.jpg" height="75"> <img src="Results/fate_face_gramian_-600.0.jpg" height="75"> <img src="Results/beethoven_ice_gramian_-600.0.jpg" height="75"> <img src="Results/church_ice_gramian_-600.0.jpg" height="75"> <img src="Results/fate_ice_gramian_-600.0.jpg" height="75">  
<img src="Results/beethoven_starry_night_gramian_-500.0.jpg" height="75"> <img src="Results/church_starry_night_gramian_-500.0.jpg" height="75"> <img src="Results/fate_starry_night_gramian_-500.0.jpg" height="75"> <img src="Results/beethoven_face_gramian_-500.0.jpg" height="75"> <img src="Results/church_face_gramian_-500.0.jpg" height="75"> <img src="Results/fate_face_gramian_-500.0.jpg" height="75"> <img src="Results/beethoven_ice_gramian_-500.0.jpg" height="75"> <img src="Results/church_ice_gramian_-500.0.jpg" height="75"> <img src="Results/fate_ice_gramian_-500.0.jpg" height="75">  
<img src="Results/beethoven_starry_night_gramian_-400.0.jpg" height="75"> <img src="Results/church_starry_night_gramian_-400.0.jpg" height="75"> <img src="Results/fate_starry_night_gramian_-400.0.jpg" height="75"> <img src="Results/beethoven_face_gramian_-400.0.jpg" height="75"> <img src="Results/church_face_gramian_-400.0.jpg" height="75"> <img src="Results/fate_face_gramian_-400.0.jpg" height="75"> <img src="Results/beethoven_ice_gramian_-400.0.jpg" height="75"> <img src="Results/church_ice_gramian_-400.0.jpg" height="75"> <img src="Results/fate_ice_gramian_-400.0.jpg" height="75">  
<img src="Results/beethoven_starry_night_gramian_-300.0.jpg" height="75"> <img src="Results/church_starry_night_gramian_-300.0.jpg" height="75"> <img src="Results/fate_starry_night_gramian_-300.0.jpg" height="75"> <img src="Results/beethoven_face_gramian_-300.0.jpg" height="75"> <img src="Results/church_face_gramian_-300.0.jpg" height="75"> <img src="Results/fate_face_gramian_-300.0.jpg" height="75"> <img src="Results/beethoven_ice_gramian_-300.0.jpg" height="75"> <img src="Results/church_ice_gramian_-300.0.jpg" height="75"> <img src="Results/fate_ice_gramian_-300.0.jpg" height="75">  
<img src="Results/beethoven_starry_night_gramian_-200.0.jpg" height="75"> <img src="Results/church_starry_night_gramian_-200.0.jpg" height="75"> <img src="Results/fate_starry_night_gramian_-200.0.jpg" height="75"> <img src="Results/beethoven_face_gramian_-200.0.jpg" height="75"> <img src="Results/church_face_gramian_-200.0.jpg" height="75"> <img src="Results/fate_face_gramian_-200.0.jpg" height="75"> <img src="Results/beethoven_ice_gramian_-200.0.jpg" height="75"> <img src="Results/church_ice_gramian_-200.0.jpg" height="75"> <img src="Results/fate_ice_gramian_-200.0.jpg" height="75">  
<img src="Results/beethoven_starry_night_gramian_-100.0.jpg" height="75"> <img src="Results/church_starry_night_gramian_-100.0.jpg" height="75"> <img src="Results/fate_starry_night_gramian_-100.0.jpg" height="75"> <img src="Results/beethoven_face_gramian_-100.0.jpg" height="75"> <img src="Results/church_face_gramian_-100.0.jpg" height="75"> <img src="Results/fate_face_gramian_-100.0.jpg" height="75"> <img src="Results/beethoven_ice_gramian_-100.0.jpg" height="75"> <img src="Results/church_ice_gramian_-100.0.jpg" height="75"> <img src="Results/fate_ice_gramian_-100.0.jpg" height="75">  
<img src="Results/beethoven_starry_night_gramian_0.0.jpg" height="75"> <img src="Results/church_starry_night_gramian_0.0.jpg" height="75"> <img src="Results/fate_starry_night_gramian_0.0.jpg" height="75"> <img src="Results/beethoven_face_gramian_0.0.jpg" height="75"> <img src="Results/church_face_gramian_0.0.jpg" height="75"> <img src="Results/fate_face_gramian_0.0.jpg" height="75"> <img src="Results/beethoven_ice_gramian_0.0.jpg" height="75"> <img src="Results/church_ice_gramian_0.0.jpg" height="75"> <img src="Results/fate_ice_gramian_0.0.jpg" height="75">  
<img src="Results/beethoven_starry_night_gramian_100.0.jpg" height="75"> <img src="Results/church_starry_night_gramian_100.0.jpg" height="75"> <img src="Results/fate_starry_night_gramian_100.0.jpg" height="75"> <img src="Results/beethoven_face_gramian_100.0.jpg" height="75"> <img src="Results/church_face_gramian_100.0.jpg" height="75"> <img src="Results/fate_face_gramian_100.0.jpg" height="75"> <img src="Results/beethoven_ice_gramian_100.0.jpg" height="75"> <img src="Results/church_ice_gramian_100.0.jpg" height="75"> <img src="Results/fate_ice_gramian_100.0.jpg" height="75">  
<img src="Results/beethoven_starry_night_gramian_200.0.jpg" height="75"> <img src="Results/church_starry_night_gramian_200.0.jpg" height="75"> <img src="Results/fate_starry_night_gramian_200.0.jpg" height="75"> <img src="Results/beethoven_face_gramian_200.0.jpg" height="75"> <img src="Results/church_face_gramian_200.0.jpg" height="75"> <img src="Results/fate_face_gramian_200.0.jpg" height="75"> <img src="Results/beethoven_ice_gramian_200.0.jpg" height="75"> <img src="Results/church_ice_gramian_200.0.jpg" height="75"> <img src="Results/fate_ice_gramian_200.0.jpg" height="75">  
<img src="Results/beethoven_starry_night_gramian_300.0.jpg" height="75"> <img src="Results/church_starry_night_gramian_300.0.jpg" height="75"> <img src="Results/fate_starry_night_gramian_300.0.jpg" height="75"> <img src="Results/beethoven_face_gramian_300.0.jpg" height="75"> <img src="Results/church_face_gramian_300.0.jpg" height="75"> <img src="Results/fate_face_gramian_300.0.jpg" height="75"> <img src="Results/beethoven_ice_gramian_300.0.jpg" height="75"> <img src="Results/church_ice_gramian_300.0.jpg" height="75"> <img src="Results/fate_ice_gramian_300.0.jpg" height="75">  
<img src="Results/beethoven_starry_night_gramian_400.0.jpg" height="75"> <img src="Results/church_starry_night_gramian_400.0.jpg" height="75"> <img src="Results/fate_starry_night_gramian_400.0.jpg" height="75"> <img src="Results/beethoven_face_gramian_400.0.jpg" height="75"> <img src="Results/church_face_gramian_400.0.jpg" height="75"> <img src="Results/fate_face_gramian_400.0.jpg" height="75"> <img src="Results/beethoven_ice_gramian_400.0.jpg" height="75"> <img src="Results/church_ice_gramian_400.0.jpg" height="75"> <img src="Results/fate_ice_gramian_400.0.jpg" height="75">  
<img src="Results/beethoven_starry_night_gramian_500.0.jpg" height="75"> <img src="Results/church_starry_night_gramian_500.0.jpg" height="75"> <img src="Results/fate_starry_night_gramian_500.0.jpg" height="75"> <img src="Results/beethoven_face_gramian_500.0.jpg" height="75"> <img src="Results/church_face_gramian_500.0.jpg" height="75"> <img src="Results/fate_face_gramian_500.0.jpg" height="75"> <img src="Results/beethoven_ice_gramian_500.0.jpg" height="75"> <img src="Results/church_ice_gramian_500.0.jpg" height="75"> <img src="Results/fate_ice_gramian_500.0.jpg" height="75">  
<img src="Results/beethoven_starry_night_gramian_600.0.jpg" height="75"> <img src="Results/church_starry_night_gramian_600.0.jpg" height="75"> <img src="Results/fate_starry_night_gramian_600.0.jpg" height="75"> <img src="Results/beethoven_face_gramian_600.0.jpg" height="75"> <img src="Results/church_face_gramian_600.0.jpg" height="75"> <img src="Results/fate_face_gramian_600.0.jpg" height="75"> <img src="Results/beethoven_ice_gramian_600.0.jpg" height="75"> <img src="Results/church_ice_gramian_600.0.jpg" height="75"> <img 
src="Results/fate_ice_gramian_600.0.jpg" height="75">

## Part 2: On theoretical part of [4]

Li et al. and Risser et al. regard each column A.<sub>k</sub> of A and G.<sub>k</sub> of G as generated from "style" probability distributions D<sub>s</sub> and D<sub>o</sub>, respectively ([4][8]). Minimizing the Gramian-matrix-based style loss L<sub>s</sub> is a way to match D<sub>o</sub> to D<sub>s</sub>.

Li et al. furthur argues that minimizing L<sub>s</sub> can be interpreted as minimizing MMD with a quadratic kernel. We slightly modify their proof and present it here.

<sup>1</sup>&frasl;<sub>n<sup>2</sup></sub> &Sigma;<sub>i=1</sub><sup>n</sup> &Sigma;<sub>j=1</sub><sup>n</sup> (G<sub>ij</sub> - A<sub>ij</sub>)<sup>2</sup>  
\= <sup>1</sup>&frasl;<sub>n<sup>2</sup></sub> &Sigma;<sub>i=1</sub><sup>n</sup> &Sigma;<sub>j=1</sub><sup>n</sup> ((<sup>1</sup>&frasl;<sub>m<sub>o</sub></sup></sub>FF<sup>T</sup>)<sub>ij</sub> - (<sup>1</sup>&frasl;<sub>m<sub>s</sub></sup></sub>SS<sup>T</sup>)<sub>ij</sub>)<sup>2</sup>  
\= <sup>1</sup>&frasl;<sub>n<sup>2</sup></sub> &Sigma;<sub>i=1</sub><sup>n</sup> &Sigma;<sub>j=1</sub><sup>n</sup> ((<sup>1</sup>&frasl;<sub>m<sub>o</sub></sup></sub> &Sigma;<sub>k=1</sub><sup>m<sub>o</sub></sub></sup> F<sub>ik</sub>F<sub>jk</sub>) - (<sup>1</sup>&frasl;<sub>m<sub>s</sub></sup></sub> &Sigma;<sub>k=1</sub><sup>m<sub>s</sub></sub></sup> S<sub>ik</sub>S<sub>jk</sub>))<sup>2</sup>  
\= <sup>1</sup>&frasl;<sub>n<sup>2</sup></sub> &Sigma;<sub>i=1</sub><sup>n</sup> &Sigma;<sub>j=1</sub><sup>n</sup> ((<sup>1</sup>&frasl;<sub>m<sub>o</sub></sup></sub> &Sigma;<sub>k=1</sub><sup>m<sub>o</sub></sub></sup> F<sub>ik</sub>F<sub>jk</sub>)<sup>2</sup> + (<sup>1</sup>&frasl;<sub>m<sub>s</sub></sup></sub> &Sigma;<sub>k=1</sub><sup>m<sub>s</sub></sub></sup> S<sub>ik</sub>S<sub>jk</sub>)<sup>2</sup> - 2(<sup>1</sup>&frasl;<sub>m<sub>o</sub></sup></sub> &Sigma;<sub>k=1</sub><sup>m<sub>o</sub></sub></sup> F<sub>ik</sub>F<sub>jk</sub>)(<sup>1</sup>&frasl;<sub>m<sub>s</sub></sup></sub> &Sigma;<sub>k=1</sub><sup>m<sub>s</sub></sub></sup> S<sub>ik</sub>S<sub>jk</sub>))  
\= <sup>1</sup>&frasl;<sub>n<sup>2</sup></sub> &Sigma;<sub>i=1</sub><sup>n</sup> &Sigma;<sub>j=1</sub><sup>n</sup> ((<sup>1</sup>&frasl;<sub>m<sub>o</sub><sup>2</sup></sup></sub> &Sigma;<sub>k<sub>1</sub>=1</sub><sup>m<sub>o</sub></sub></sup> &Sigma;<sub>k<sub>2</sub>=1</sub><sup>m<sub>o</sub></sub></sup> F<sub>ik<sub>1</sub></sub>F<sub>jk<sub>1</sub></sub>F<sub>ik<sub>2</sub></sub>F<sub>jk<sub>2</sub></sub>) + (<sup>1</sup>&frasl;<sub>m<sub>s</sub><sup>2</sup></sup></sub> &Sigma;<sub>k<sub>1</sub>=1</sub><sup>m<sub>s</sub></sub></sup> &Sigma;<sub>k<sub>2</sub>=1</sub><sup>m<sub>s</sub></sub></sup> S<sub>ik<sub>1</sub></sub>S<sub>jk<sub>1</sub></sub>S<sub>ik<sub>2</sub></sub>S<sub>jk<sub>2</sub></sub>) - (<sup>2</sup>&frasl;<sub>m<sub>o</sub>m<sub>s</sub></sup></sub> &Sigma;<sub>k<sub>1</sub>=1</sub><sup>m<sub>o</sub></sub></sup> &Sigma;<sub>k<sub>2</sub>=1</sub><sup>m<sub>s</sub></sub></sup> F<sub>ik<sub>1</sub></sub>F<sub>jk<sub>1</sub></sub>S<sub>ik<sub>2</sub></sub>S<sub>jk<sub>2</sub></sub>))  
\= (<sup>1</sup>&frasl;<sub>m<sub>o</sub><sup>2</sup></sup></sub> &Sigma;<sub>k<sub>1</sub>=1</sub><sup>m<sub>o</sub></sub></sup> &Sigma;<sub>k<sub>2</sub>=1</sub><sup>m<sub>o</sub></sub></sup> <sup>1</sup>&frasl;<sub>n<sup>2</sup></sub> &Sigma;<sub>i=1</sub><sup>n</sup> &Sigma;<sub>j=1</sub><sup>n</sup> F<sub>ik<sub>1</sub></sub>F<sub>jk<sub>1</sub></sub>F<sub>ik<sub>2</sub></sub>F<sub>jk<sub>2</sub></sub>) + (<sup>1</sup>&frasl;<sub>m<sub>s</sub><sup>2</sup></sup></sub> &Sigma;<sub>k<sub>1</sub>=1</sub><sup>m<sub>s</sub></sub></sup> &Sigma;<sub>k<sub>2</sub>=1</sub><sup>m<sub>s</sub></sub></sup> <sup>1</sup>&frasl;<sub>n<sup>2</sup></sub> &Sigma;<sub>i=1</sub><sup>n</sup> &Sigma;<sub>j=1</sub><sup>n</sup> S<sub>ik<sub>1</sub></sub>S<sub>jk<sub>1</sub></sub>S<sub>ik<sub>2</sub></sub>S<sub>jk<sub>2</sub></sub>) - (<sup>2</sup>&frasl;<sub>m<sub>o</sub>m<sub>s</sub></sup></sub> &Sigma;<sub>k<sub>1</sub>=1</sub><sup>m<sub>o</sub></sub></sup> &Sigma;<sub>k<sub>2</sub>=1</sub><sup>m<sub>s</sub></sub></sup> <sup>1</sup>&frasl;<sub>n<sup>2</sup></sub> &Sigma;<sub>i=1</sub><sup>n</sup> &Sigma;<sub>j=1</sub><sup>n</sup> F<sub>ik<sub>1</sub></sub>F<sub>jk<sub>1</sub></sub>S<sub>ik<sub>2</sub></sub>S<sub>jk<sub>2</sub></sub>)  
\= (<sup>1</sup>&frasl;<sub>m<sub>o</sub><sup>2</sup></sup></sub> &Sigma;<sub>k<sub>1</sub>=1</sub><sup>m<sub>o</sub></sub></sup> &Sigma;<sub>k<sub>2</sub>=1</sub><sup>m<sub>o</sub></sub></sup> (<sup>1</sup>&frasl;<sub>n</sub> &Sigma;<sub>i=1</sub><sup>n</sup> F<sub>ik<sub>1</sub></sub>F<sub>ik<sub>2</sub></sub>)<sup>2</sup>) + (<sup>1</sup>&frasl;<sub>m<sub>s</sub><sup>2</sup></sup></sub> &Sigma;<sub>k<sub>1</sub>=1</sub><sup>m<sub>s</sub></sub></sup> &Sigma;<sub>k<sub>2</sub>=1</sub><sup>m<sub>s</sub></sub></sup> (<sup>1</sup>&frasl;<sub>n</sub> &Sigma;<sub>i=1</sub><sup>n</sup> S<sub>ik<sub>1</sub></sub>S<sub>ik<sub>2</sub></sub></sub>)<sup>2</sup>) - (<sup>2</sup>&frasl;<sub>m<sub>o</sub>m<sub>s</sub></sup></sub> &Sigma;<sub>k<sub>1</sub>=1</sub><sup>m<sub>o</sub></sub></sup> &Sigma;<sub>k<sub>2</sub>=1</sub><sup>m<sub>s</sub></sub></sup> (<sup>1</sup>&frasl;<sub>n</sub> &Sigma;<sub>i=1</sub><sup>n</sup> F<sub>ik<sub>1</sub></sub>S<sub>ik<sub>2</sub></sub>)<sup>2</sup>)  
\= (<sup>1</sup>&frasl;<sub>m<sub>o</sub><sup>2</sup></sup></sub> &Sigma;<sub>k<sub>1</sub>=1</sub><sup>m<sub>o</sub></sub></sup> &Sigma;<sub>k<sub>2</sub>=1</sub><sup>m<sub>o</sub></sub></sup> (<sup>1</sup>&frasl;<sub>n</sub> F.<sub>k<sub>1</sub></sub><sup>T</sup>F.<sub>k<sub>2</sub></sub>)<sup>2</sup>) + (<sup>1</sup>&frasl;<sub>m<sub>s</sub><sup>2</sup></sup></sub> &Sigma;<sub>k<sub>1</sub>=1</sub><sup>m<sub>s</sub></sub></sup> &Sigma;<sub>k<sub>2</sub>=1</sub><sup>m<sub>s</sub></sub></sup> (<sup>1</sup>&frasl;<sub>n</sub> S.<sub>k<sub>1</sub></sub><sup>T</sup>S.<sub>k<sub>2</sub></sub></sub>)<sup>2</sup>) - (<sup>2</sup>&frasl;<sub>m<sub>o</sub>m<sub>s</sub></sup></sub> &Sigma;<sub>k<sub>1</sub>=1</sub><sup>m<sub>o</sub></sub></sup> &Sigma;<sub>k<sub>2</sub>=1</sub><sup>m<sub>s</sub></sub></sup> (<sup>1</sup>&frasl;<sub>n</sub> F.<sub>k<sub>1</sub></sub><sup>T</sup>S.<sub>k<sub>2</sub></sub>)<sup>2</sup>)  
\= (<sup>1</sup>&frasl;<sub>m<sub>o</sub><sup>2</sup></sup></sub> &Sigma;<sub>k<sub>1</sub>=1</sub><sup>m<sub>o</sub></sub></sup> &Sigma;<sub>k<sub>2</sub>=1</sub><sup>m<sub>o</sub></sub></sup> K(F.<sub>k<sub>1</sub></sub>, F.<sub>k<sub>2</sub></sub>, 2)) + (<sup>1</sup>&frasl;<sub>m<sub>s</sub><sup>2</sup></sup></sub> &Sigma;<sub>k<sub>1</sub>=1</sub><sup>m<sub>s</sub></sub></sup> &Sigma;<sub>k<sub>2</sub>=1</sub><sup>m<sub>s</sub></sub></sup> K(S.<sub>k<sub>1</sub></sub>, S.<sub>k<sub>2</sub></sub>, 2)) - (<sup>2</sup>&frasl;<sub>m<sub>o</sub>m<sub>s</sub></sup></sub> &Sigma;<sub>k<sub>1</sub>=1</sub><sup>m<sub>o</sub></sub></sup> &Sigma;<sub>k<sub>2</sub>=1</sub><sup>m<sub>s</sub></sub></sup> K(F.<sub>k<sub>1</sub></sub>, S.<sub>k<sub>2</sub></sub>, 2)) -------- (1), where  
K(v, u, p) = (<sup>1</sup>&frasl;<sub>n</sub>v<sup>T</sup>u)<sup>p</sup> is the averaged power kernel.

Theoretically we can use any positive number as the power p of the kernel, but in practice, we need to compute S<sup>T</sup>S and F<sup>T</sup>F, which are of size m<sub>s</sub><sup>2</sup> and m<sub>o</sub><sup>2</sup>, in order to computing MMD. This exhausts memory. Thus, we need to convert the MMD side to the Gramian side if we want to use a power kernel with another p. We now show that MMD with kernel with integer power are easy to convert. The following is the general version of equation (1).

Let p be a nonnegative integer.  
(<sup>1</sup>&frasl;<sub>m<sub>o</sub><sup>2</sup></sup></sub> &Sigma;<sub>k<sub>1</sub>=1</sub><sup>m<sub>o</sub></sub></sup> &Sigma;<sub>k<sub>2</sub>=1</sub><sup>m<sub>o</sub></sub></sup> K(F.<sub>k<sub>1</sub></sub>, F.<sub>k<sub>2</sub></sub>, p)) + (<sup>1</sup>&frasl;<sub>m<sub>s</sub><sup>2</sup></sup></sub> &Sigma;<sub>k<sub>1</sub>=1</sub><sup>m<sub>s</sub></sub></sup> &Sigma;<sub>k<sub>2</sub>=1</sub><sup>m<sub>s</sub></sub></sup> K(S.<sub>k<sub>1</sub></sub>, S.<sub>k<sub>2</sub></sub>, p)) - (<sup>2</sup>&frasl;<sub>m<sub>o</sub>m<sub>s</sub></sup></sub> &Sigma;<sub>k<sub>1</sub>=1</sub><sup>m<sub>o</sub></sub></sup> &Sigma;<sub>k<sub>2</sub>=1</sub><sup>m<sub>s</sub></sub></sup> K(F.<sub>k<sub>1</sub></sub>, S.<sub>k<sub>2</sub></sub>, p))  
\= (<sup>1</sup>&frasl;<sub>m<sub>o</sub><sup>2</sup></sup></sub> &Sigma;<sub>k<sub>1</sub>=1</sub><sup>m<sub>o</sub></sub></sup> &Sigma;<sub>k<sub>2</sub>=1</sub><sup>m<sub>o</sub></sub></sup> (<sup>1</sup>&frasl;<sub>n</sub> F.<sub>k<sub>1</sub></sub><sup>T</sup>F.<sub>k<sub>2</sub></sub>)<sup>p</sup>) + (<sup>1</sup>&frasl;<sub>m<sub>s</sub><sup>2</sup></sup></sub> &Sigma;<sub>k<sub>1</sub>=1</sub><sup>m<sub>s</sub></sub></sup> &Sigma;<sub>k<sub>2</sub>=1</sub><sup>m<sub>s</sub></sub></sup> (<sup>1</sup>&frasl;<sub>n</sub> S.<sub>k<sub>1</sub></sub><sup>T</sup>S.<sub>k<sub>2</sub></sub></sub>)<sup>p</sup>) - (<sup>2</sup>&frasl;<sub>m<sub>o</sub>m<sub>s</sub></sup></sub> &Sigma;<sub>k<sub>1</sub>=1</sub><sup>m<sub>o</sub></sub></sup> &Sigma;<sub>k<sub>2</sub>=1</sub><sup>m<sub>s</sub></sub></sup> (<sup>1</sup>&frasl;<sub>n</sub> F.<sub>k<sub>1</sub></sub><sup>T</sup>S.<sub>k<sub>2</sub></sub>)<sup>p</sup>)  
\= (<sup>1</sup>&frasl;<sub>m<sub>o</sub><sup>2</sup></sup></sub> &Sigma;<sub>k<sub>1</sub>=1</sub><sup>m<sub>o</sub></sub></sup> &Sigma;<sub>k<sub>2</sub>=1</sub><sup>m<sub>o</sub></sub></sup> (<sup>1</sup>&frasl;<sub>n</sub> &Sigma;<sub>i=1</sub><sup>n</sup> F<sub>ik<sub>1</sub></sub>F<sub>ik<sub>2</sub></sub>)<sup>p</sup>) + (<sup>1</sup>&frasl;<sub>m<sub>s</sub><sup>2</sup></sup></sub> &Sigma;<sub>k<sub>1</sub>=1</sub><sup>m<sub>s</sub></sub></sup> &Sigma;<sub>k<sub>2</sub>=1</sub><sup>m<sub>s</sub></sub></sup> (<sup>1</sup>&frasl;<sub>n</sub> &Sigma;<sub>i=1</sub><sup>n</sup> S<sub>ik<sub>1</sub></sub>S<sub>ik<sub>2</sub></sub></sub>)<sup>p</sup>) - (<sup>2</sup>&frasl;<sub>m<sub>o</sub>m<sub>s</sub></sup></sub> &Sigma;<sub>k<sub>1</sub>=1</sub><sup>m<sub>o</sub></sub></sup> &Sigma;<sub>k<sub>2</sub>=1</sub><sup>m<sub>s</sub></sub></sup> (<sup>1</sup>&frasl;<sub>n</sub> &Sigma;<sub>i=1</sub><sup>n</sup> F<sub>ik<sub>1</sub></sub>S<sub>ik<sub>2</sub></sub>)<sup>p</sup>)  
\= (<sup>1</sup>&frasl;<sub>m<sub>o</sub><sup>2</sup></sup></sub> &Sigma;<sub>k<sub>1</sub>=1</sub><sup>m<sub>o</sub></sub></sup> &Sigma;<sub>k<sub>2</sub>=1</sub><sup>m<sub>o</sub></sub></sup> <sup>1</sup>&frasl;<sub>n<sup>p</sup></sub> &Sigma;<sub>i<sub>1</sub>=1</sub><sup>n</sup> &Sigma;<sub>i<sub>2</sub>=1</sub><sup>n</sup> ... &Sigma;<sub>i<sub>p</sub>=1</sub><sup>n</sup> (&Pi;<sub>q=1</sub><sup>p</sup> F<sub>i<sub>q</sub>k<sub>1</sub></sub>)(&Pi;<sub>q=1</sub><sup>p</sup> F<sub>i<sub>q</sub>k<sub>2</sub></sub>)) + (<sup>1</sup>&frasl;<sub>m<sub>s</sub><sup>2</sup></sup></sub> &Sigma;<sub>k<sub>1</sub>=1</sub><sup>m<sub>s</sub></sub></sup> &Sigma;<sub>k<sub>2</sub>=1</sub><sup>m<sub>s</sub></sub></sup> <sup>1</sup>&frasl;<sub>n<sup>p</sup></sub> &Sigma;<sub>i<sub>1</sub>=1</sub><sup>n</sup> &Sigma;<sub>i<sub>2</sub>=1</sub><sup>n</sup> ... &Sigma;<sub>i<sub>p</sub>=1</sub><sup>n</sup> (&Pi;<sub>q=1</sub><sup>p</sup> S<sub>i<sub>q</sub>k<sub>1</sub></sub>)(&Pi;<sub>q=1</sub><sup>p</sup> S<sub>i<sub>q</sub>k<sub>2</sub></sub>)) - (<sup>2</sup>&frasl;<sub>m<sub>o</sub>m<sub>s</sub></sup></sub> &Sigma;<sub>k<sub>1</sub>=1</sub><sup>m<sub>o</sub></sub></sup> &Sigma;<sub>k<sub>2</sub>=1</sub><sup>m<sub>s</sub></sub></sup> <sup>1</sup>&frasl;<sub>n<sup>p</sup></sub> &Sigma;<sub>i<sub>1</sub>=1</sub><sup>n</sup> &Sigma;<sub>i<sub>2</sub>=1</sub><sup>n</sup> ... &Sigma;<sub>i<sub>p</sub>=1</sub><sup>n</sup> (&Pi;<sub>q=1</sub><sup>p</sup> F<sub>i<sub>q</sub>k<sub>1</sub></sub>)(&Pi;<sub>q=1</sub><sup>p</sup> S<sub>i<sub>q</sub>k<sub>2</sub></sub>))  
\= <sup>1</sup>&frasl;<sub>n<sup>p</sup></sub> &Sigma;<sub>i<sub>1</sub>=1</sub><sup>n</sup> &Sigma;<sub>i<sub>2</sub>=1</sub><sup>n</sup> ... &Sigma;<sub>i<sub>p</sub>=1</sub><sup>n</sup> ((<sup>1</sup>&frasl;<sub>m<sub>o</sub><sup>2</sup></sup></sub> &Sigma;<sub>k<sub>1</sub>=1</sub><sup>m<sub>o</sub></sub></sup> &Sigma;<sub>k<sub>2</sub>=1</sub><sup>m<sub>o</sub></sub></sup> (&Pi;<sub>q=1</sub><sup>p</sup> F<sub>i<sub>q</sub>k<sub>1</sub></sub>)(&Pi;<sub>q=1</sub><sup>p</sup> F<sub>i<sub>q</sub>k<sub>2</sub></sub>)) + (<sup>1</sup>&frasl;<sub>m<sub>s</sub><sup>2</sup></sup></sub> &Sigma;<sub>k<sub>1</sub>=1</sub><sup>m<sub>s</sub></sub></sup> &Sigma;<sub>k<sub>2</sub>=1</sub><sup>m<sub>s</sub></sub></sup> (&Pi;<sub>q=1</sub><sup>p</sup> S<sub>i<sub>q</sub>k<sub>1</sub></sub>)(&Pi;<sub>q=1</sub><sup>p</sup> S<sub>i<sub>q</sub>k<sub>2</sub></sub>)) - (<sup>2</sup>&frasl;<sub>m<sub>o</sub>m<sub>s</sub></sup></sub> &Sigma;<sub>k<sub>1</sub>=1</sub><sup>m<sub>o</sub></sub></sup> &Sigma;<sub>k<sub>2</sub>=1</sub><sup>m<sub>s</sub></sub></sup> (&Pi;<sub>q=1</sub><sup>p</sup> F<sub>i<sub>q</sub>k<sub>1</sub></sub>)(&Pi;<sub>q=1</sub><sup>p</sup> S<sub>i<sub>q</sub>k<sub>2</sub></sub>)))  
\= <sup>1</sup>&frasl;<sub>n<sup>p</sup></sub> &Sigma;<sub>i<sub>1</sub>=1</sub><sup>n</sup> &Sigma;<sub>i<sub>2</sub>=1</sub><sup>n</sup> ... &Sigma;<sub>i<sub>p</sub>=1</sub><sup>n</sup> ((<sup>1</sup>&frasl;<sub>m<sub>o</sub></sup></sub> &Sigma;<sub>k=1</sub><sup>m<sub>o</sub></sub></sup> &Pi;<sub>q=1</sub><sup>p</sup> F<sub>i<sub>q</sub>k</sub>)<sup>2</sup> + (<sup>1</sup>&frasl;<sub>m<sub>s</sub></sup></sub> &Sigma;<sub>k=1</sub><sup>m<sub>s</sub></sub></sup> &Pi;<sub>q=1</sub><sup>p</sup> S<sub>i<sub>q</sub>k</sub>)<sup>2</sup> - 2(<sup>1</sup>&frasl;<sub>m<sub>o</sub></sup></sub> &Sigma;<sub>k=1</sub><sup>m<sub>o</sub></sub></sup> &Pi;<sub>q=1</sub><sup>p</sup> F<sub>i<sub>q</sub>k</sub>)(<sup>1</sup>&frasl;<sub>m<sub>s</sub></sup></sub> &Sigma;<sub>k=1</sub><sup>m<sub>s</sub></sub></sup> &Pi;<sub>q=1</sub><sup>p</sup> S<sub>i<sub>q</sub>k</sub>))  
\= <sup>1</sup>&frasl;<sub>n<sup>p</sup></sub> &Sigma;<sub>i<sub>1</sub>=1</sub><sup>n</sup> &Sigma;<sub>i<sub>2</sub>=1</sub><sup>n</sup> ... &Sigma;<sub>i<sub>p</sub>=1</sub><sup>n</sup> ((<sup>1</sup>&frasl;<sub>m<sub>o</sub></sup></sub> &Sigma;<sub>k=1</sub><sup>m<sub>o</sub></sub></sup> &Pi;<sub>q=1</sub><sup>p</sup> F<sub>i<sub>q</sub>k</sub>) - (<sup>1</sup>&frasl;<sub>m<sub>s</sub></sup></sub> &Sigma;<sub>k=1</sub><sup>m<sub>s</sub></sub></sup> &Pi;<sub>q=1</sub><sup>p</sup> S<sub>i<sub>q</sub>k</sub>))<sup>2</sup> -------- (2).

Equation (2) can be interpreted as: **MMD with kernel with power p is equivalent to mean squared error of frequencies of co-occurrences of all possible permutations of p features.** The case with p = 3 is also mentioned in [3], section 4.5.

Although we can implement the generalized Gramian side for all nonnegative integer p, it still runs out of memory for p &ge; 2. Unfortunately, only p = 1 and 2 (using Gramian matrices) are practical.

For p = 1, we can furthur simplify equation (2).  
<sup>1</sup>&frasl;<sub>n<sup>p</sup></sub> &Sigma;<sub>i<sub>1</sub>=1</sub><sup>n</sup> &Sigma;<sub>i<sub>2</sub>=1</sub><sup>n</sup> ... &Sigma;<sub>i<sub>p</sub>=1</sub><sup>n</sup> ((<sup>1</sup>&frasl;<sub>m<sub>o</sub></sup></sub> &Sigma;<sub>k=1</sub><sup>m<sub>o</sub></sub></sup> &Pi;<sub>q=1</sub><sup>p</sup> F<sub>i<sub>q</sub>k</sub>) - (<sup>1</sup>&frasl;<sub>m<sub>s</sub></sup></sub> &Sigma;<sub>k=1</sub><sup>m<sub>s</sub></sub></sup> &Pi;<sub>q=1</sub><sup>p</sup> S<sub>i<sub>q</sub>k</sub>))<sup>2</sup>  
\= <sup>1</sup>&frasl;<sub>n</sub> &Sigma;<sub>i=1</sub><sup>n</sup> ((<sup>1</sup>&frasl;<sub>m<sub>o</sub></sup></sub> &Sigma;<sub>k=1</sub><sup>m<sub>o</sub></sub></sup> F<sub>ik</sub>) - (<sup>1</sup>&frasl;<sub>m<sub>s</sub></sup></sub> &Sigma;<sub>k=1</sub><sup>m<sub>s</sub></sub></sup> S<sub>ik</sub>))<sup>2</sup>  
\= <sup>1</sup>&frasl;<sub>n</sub> &Sigma;<sub>i=1</sub><sup>n</sup> (mean(F<sub>i</sub>.) - mean(S<sub>i</sub>.))<sup>2</sup> -------- (3).  
**MMD with linear kernel is equivalent to mean squared error of frequencies of occurrence of every feature.**  
This is much more easier to compute than what equation (2) suggests. We call this approach "mean vector" as opposed to the original Gramian matrix.

## Part 3: Experiment of part 2

In this part we show that the style loss using mean vectors does capture some aspects of the style. 

### Mean vector

<img src="Results/beethoven_starry_night_mean_0.0.jpg" height="200"> <img src="Results/church_starry_night_mean_0.0.jpg" height="200"> <img src="Results/fate_starry_night_mean_0.0.jpg" height="200">
<img src="Results/beethoven_face_mean_0.0.jpg" height="200"> <img src="Results/church_face_mean_0.0.jpg" height="200"> <img src="Results/fate_face_mean_0.0.jpg" height="200">
<img src="Results/beethoven_ice_mean_0.0.jpg" height="200"> <img src="Results/church_ice_mean_0.0.jpg" height="200"> <img src="Results/fate_ice_mean_0.0.jpg" height="200">

## Part 4: Link between activation shift and MMD

By substituting each variable z in equation (2) for (z + sU), we have  
<sup>1</sup>&frasl;<sub>n<sup>p</sup></sub> &Sigma;<sub>i<sub>1</sub>=1</sub><sup>n</sup> &Sigma;<sub>i<sub>2</sub>=1</sub><sup>n</sup> ... &Sigma;<sub>i<sub>p</sub>=1</sub><sup>n</sup> ((<sup>1</sup>&frasl;<sub>m<sub>o</sub></sup></sub> &Sigma;<sub>k=1</sub><sup>m<sub>o</sub></sub></sup> &Pi;<sub>q=1</sub><sup>p</sup> (F<sub>i<sub>q</sub>k</sub> + sU)) - (<sup>1</sup>&frasl;<sub>m<sub>s</sub></sup></sub> &Sigma;<sub>k=1</sub><sup>m<sub>s</sub></sub></sup> &Pi;<sub>q=1</sub><sup>p</sup> (S<sub>i<sub>q</sub>k</sub> + sU)))<sup>2</sup>  
\= (<sup>1</sup>&frasl;<sub>m<sub>o</sub><sup>2</sup></sup></sub> &Sigma;<sub>k<sub>1</sub>=1</sub><sup>m<sub>o</sub></sub></sup> &Sigma;<sub>k<sub>2</sub>=1</sub><sup>m<sub>o</sub></sub></sup> K(F.<sub>k<sub>1</sub></sub> + sU, F.<sub>k<sub>2</sub></sub> + sU, p)) + (<sup>1</sup>&frasl;<sub>m<sub>s</sub><sup>2</sup></sup></sub> &Sigma;<sub>k<sub>1</sub>=1</sub><sup>m<sub>s</sub></sub></sup> &Sigma;<sub>k<sub>2</sub>=1</sub><sup>m<sub>s</sub></sub></sup> K(S.<sub>k<sub>1</sub></sub> + sU, S.<sub>k<sub>2</sub></sub> + sU, p)) - (<sup>2</sup>&frasl;<sub>m<sub>o</sub>m<sub>s</sub></sup></sub> &Sigma;<sub>k<sub>1</sub>=1</sub><sup>m<sub>o</sub></sub></sup> &Sigma;<sub>k<sub>2</sub>=1</sub><sup>m<sub>s</sub></sub></sup> K(F.<sub>k<sub>1</sub></sub> + sU, S.<sub>k<sub>2</sub></sub> + sU, p)) -------- (4).

Letting p = 2, we find that the LHS is the style loss using Gramian matrices with activation shift s, and the RHS is MMD with "shifted" quadratic kernel. Thus we can reinterpret the conclusion in part 1: **style loss using MMD with shifted quadratic kernel removes ghosting.**

## Part 5: Style loss using variance vector and covariance matrix

In part 2, we mention that minimizing the Gramian-matrix-based style loss is a way to match D<sub>o</sub> to D<sub>s</sub>. There are other ways to describe D<sub>o</sub> and D<sub>s</sub> and therefore are other ways to match them. In this part, we try two descriptions: variance vector and covariance matrix, as shown in the following.

L<sub>s</sub> = &Sigma;<sub>l</sub><sup></sup> <sup>b<sup>l</sup></sup>&frasl;<sub>n<sup>l</sup></sub> &Sigma;<sub>i=1</sub><sup>n<sup>l</sup></sup> (var(F<sub>i</sub>.) - (var(S<sub>i</sub>.))<sup>2</sup>, and  
L<sub>s</sub> = &Sigma;<sub>l</sub><sup></sup> <sup>b<sup>l</sup></sup>&frasl;<sub>(n<sup>l</sup>)<sup>2</sup></sub> &Sigma;<sub>i=1</sub><sup>n<sup>l</sup></sup> &Sigma;<sub>j=1</sub><sup>n<sup>l</sup></sup> (cov(D<sub>o</sub>)<sub>ij</sub> - (cov(D<sub>s</sub>)<sub>ij</sub>)<sup>2</sup>, respectively.

We now examine their performance. It can be observed that the **style loss using covariance matrices creates great results without ghosting** while thie one using variance vectors creates ghosting easily. It can also be observed that **some styles are easy to transfer (producing less ghosting) while some are not**. For example, Vincent van Gogh's the starry night is an easier one.

### Variance vector
<img src="Results/beethoven_starry_night_variance_0.0.jpg" height="200"> <img src="Results/church_starry_night_variance_0.0.jpg" height="200"> <img src="Results/fate_starry_night_variance_0.0.jpg" height="200">
<img src="Results/beethoven_face_variance_0.0.jpg" height="200"> <img src="Results/church_face_variance_0.0.jpg" height="200"> <img src="Results/fate_face_variance_0.0.jpg" height="200">
<img src="Results/beethoven_ice_variance_0.0.jpg" height="200"> <img src="Results/church_ice_variance_0.0.jpg" height="200"> <img src="Results/fate_ice_variance_0.0.jpg" height="200">

### Covariance matrix
<img src="Results/beethoven_starry_night_covariance_0.0.jpg" height="200"> <img src="Results/church_starry_night_covariance_0.0.jpg" height="200"> <img src="Results/fate_starry_night_covariance_0.0.jpg" height="200">
<img src="Results/beethoven_face_covariance_0.0.jpg" height="200"> <img src="Results/church_face_covariance_0.0.jpg" height="200"> <img src="Results/fate_face_covariance_0.0.jpg" height="200">
<img src="Results/beethoven_ice_covariance_0.0.jpg" height="200"> <img src="Results/church_ice_covariance_0.0.jpg" height="200"> <img src="Results/fate_ice_covariance_0.0.jpg" height="200">

## Duscussion

We see in the experiments above that there are two style losses which create great results: the one using shifted Gramian matrics and the one using covariance matrices. We regard the latter one as a more elegant way since in the former one, we need to decide one more argument, namely, the activation shift s. However, there are some cases in which the former one outperforms the latter.

When doing this project, we find that striking the among the weights, a<sub>l</sub> and b<sub>l</sub>, of the losses is a nontrivial work. There is no solid theory that guides us to tune them, and we feel that the balance is just occasionally struck. This may be why people seek other approaches such as feed-forward methods.

## References

### Papers

[1.1] L. A. Gatys, A. S. Ecker, and M. Bethge. Image style transfer using convolutional neural networks. 2016.  
[1.2] L. A. Gatys, A. S. Ecker, and M. Bethge. A neural algorithm of artistic style. 2015. Preprint version of [1.1].  
[2] Y. Jing, Y. Yang, Z. Feng, J. Ye, and M. Song. Neural Style Transfer: A Review. 2017.  
[3] R. Novak and Y. Nikulin. Improving the neural algorithm of artistic style. 2016.  
[4] Y. Li, N. Wang, J. Liu, and X. Hou. Demystifying neural style transfer. 2017.  
[5] L. A. Gatys, A. S. Ecker, M. Bethge, A. Hertzmann, and E. Shechtman. Controlling perceptual factors in neural style transfer. 2016.  
[6] K. Simonyan and A. Zisserman. Very Deep Convolutional Networks for Large-Scale Image Recognition. 2014.  
[7] Y. Nikulin and R. Novak. Exploring the neural algorithm of artistic style. 2016.  
[8] E. Risser, P. Wilmot, and C. Barnes. Stable and controllable neural texture synthesis and style transfer using histogram losses. 2017.

### Source code

[9] PyTorch implementation of [1.1] by Gatys. https://github.com/leongatys/PytorchNeuralStyleTransfer

### Source images

#### Content images

[C.1] Ludwig van Beethoven. https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/Beethoven.jpg/1200px-Beethoven.jpg  
[C.2] Notre-Dame de Paris. http://www.newlooktravel.tn/photo/3-80-NewLookTravel_Paris_Cathedrale-Notre-Dame-nuit.jpg  
[C.3] Opening of symphony no. 5 by Ludwig van Beethoven. https://www.bostonglobe.com/arts/music/2012/11/17/beethoven-fifth-symphony-warhorse-for-all-times/UjG7gHBTD5z5qB843gYZZO/story.html

#### Style images

[S.1] The starry night by Vincent van Gogh. https://raw.githubusercontent.com/leongatys/PytorchNeuralStyleTransfer/master/Images/vangogh_starry_night.jpg  
[S.2] Fawkes by Randal Roberts. http://www.neurosoup.com/wp-content/uploads/2016/08/2012-fawkes-randal-roberts_580.jpg  
[S.3] Ice. http://eskipaper.com/images/ice-5.jpg

### Other references

[10] Gramian matrix. https://en.wikipedia.org/wiki/Gramian_matrix
