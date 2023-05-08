Download Link: https://assignmentchef.com/product/solved-statsm232a-project-5-generator-and-descriptor
<br>



<h1>1           Generator: real inference</h1>

The model has the following form:

(1)

N(0<em>,σ</em><sup>2</sup><em>I<sub>D</sub></em>)<em>, d &lt; D.                                                   </em>(2)

<em>f</em>(<em>Z</em>;<em>W</em>) maps latent factors into image <em>Y </em>, where <em>W </em>collects all the connection weights and bias terms of the ConvNet.

Adopting the language of the EM algorithm, the complete data model is given by

log<em>p</em>(<em>Y,Z</em>;<em>W</em>) = log[<em>p</em>(<em>Z</em>)<em>p</em>(<em>Y </em>|<em>Z,W</em>)]                                                                           (3)

+ const<em>.                          </em>(4)

The observed-data model is obtained by intergrating out <em>Z</em>: <em>p</em>(<em>Y </em>;<em>W</em>) = <sup>R </sup><em>p</em>(<em>Z</em>)<em>p</em>(<em>Y </em>|<em>Z,W</em>)<em>dZ</em>. The posterior distribution of <em>Z </em>is given by <em>p</em>(<em>Z</em>|<em>Y,W</em>) = <em>p</em>(<em>Y,Z</em>;<em>W</em>)<em>/p</em>(<em>Y </em>;<em>W</em>) ∝ <em>p</em>(<em>Z</em>)<em>p</em>(<em>Y </em>|<em>Z,W</em>) as a function of <em>Z</em>.

We want to minimize the observed-data log-likelihood, which is

. The gradient of <em>L</em>(<em>W</em>) can be calculated according to the fol-

lowing well-known fact that underlies the EM algorithm:

;<em>W</em>)<em>dZ                                    </em>(5)

<em>.                                   </em>(6)

The expectation with respect to <em>p</em>(<em>Z</em>|<em>Y,W</em>) can be approximated by drawing samples from <em>p</em>(<em>Z</em>|<em>Y,W</em>) and then compute the Monte Carlo average.

The Langevin dynamics for sampling <em>Z </em>∼ <em>p</em>(<em>Z</em>|<em>Y,W</em>) iterates

<em>,                          </em>(7)

where <em>τ </em>denotes the time step for the Langevin sampling, <em>δ </em>is the step size, and <em>U<sub>τ </sub></em>denotes a random vector that follows N(0<em>,I<sub>d</sub></em>).

The stochastic gradient algorithm can be used for learning, where in each iteration, for each <em>Z<sub>i</sub></em>, only a single copy of <em>Z<sub>i </sub></em>is sampled from <em>p</em>(<em>Z<sub>i</sub></em>|<em>Y<sub>i</sub>,W</em>) by running a finite number of steps of Langevin dynamics starting from the current value of <em>Z<sub>i</sub></em>, i.e., the warm start. With {<em>Z<sub>i</sub></em>} sampled in this manner, we can update the parameter <em>W </em>based on the gradient <em>L</em><sup>0</sup>(<em>W</em>), whose Monte Carlo approximation is:

)                                                                (8)

(9)

<em>.                                    </em>(10)

Algorithm 1 describes the details of the learning and sampling algorithm.

<strong>Algorithm 1 </strong>Generator: real inference <strong>Input:</strong>

<ul>

 <li>training examples {<em>Y<sub>i</sub>,i </em>= 1<em>,…,n</em>},</li>

 <li>number of Langevin steps <em>l</em>, (3) number of learning iterations <em>T</em>.</li>

</ul>

<strong>Output:</strong>

<ul>

 <li>learned parameters W,</li>

 <li>inferred latent factors {<em>Z<sub>i</sub>,i </em>= 1<em>,…,n</em>}.</li>

</ul>

1: Let <em>t </em>← 0, initialize W.

2: Initialize <em>Z<sub>i</sub></em>, for <em>i </em>= 1<em>,…,n</em>.

3: <strong>repeat</strong>

4: <strong>Inference step</strong>: For each <em>i</em>, run <em>l </em>steps of of Langevin dynamics to sample <em>Z<sub>i </sub></em>∼ <em>p</em>(<em>Z<sub>i</sub></em>|<em>Y<sub>i</sub>,W</em>) with warm start, i.e., starting from the current <em>Z<sub>i</sub></em>, each step follows equation 7.

5: <strong>Learning step</strong>: Update <em>W </em>← <em>W </em>+<em>γ<sub>t</sub>L</em><sup>0</sup>(<em>W</em>), where <em>L</em><sup>0</sup>(<em>W</em>) is computed according to equation 10, with learning rate <em>γ<sub>t</sub></em>.

6:            Let <em>t </em>← <em>t </em>+ 1.

7: <strong>until </strong><em>t </em>= <em>T</em>

<h2>1.1         TO DO</h2>

For the lion-tiger category, learn a model with 2-dim latent factor vector. Fill the blank part of ./GenNet/GenNet.py. <strong>Show</strong>:

<ul>

 <li>Reconstructed images of training images, using the inferred <em>z </em>from training images.</li>

 <li>Randomly generated images, using randomly sampled <em>z</em>.</li>

 <li>Generated images with linearly interpolated latent factors from (−2<em>,</em>2) to (−2<em>,</em>2). For example, you inperlolate 8 points from (−2<em>,</em>2) for each dimension of <em>z</em>. Then you will get a 8 × 8 panel of images. You should be able to seee that tigers slight change to lion.</li>

 <li>Plot of loss over iteration.</li>

</ul>

<h1>2           Descriptor: real sampling</h1>

The descriptor model is as follows:

<em>,                                                       </em>(11)

where <em>p</em><sub>0</sub>(<em>Y </em>) is the reference distribution such as Gaussian white noise

(12)

The scoring function <em>f<sub>θ</sub></em>(<em>Y </em>) is defined by a bottom-up ConvNet whose parameters are denoted by <em>θ</em>. The normalizing constant <em>Z</em>(<em>θ</em>) = <sup>R </sup>exp[<em>f<sub>θ</sub></em>(<em>Y </em>)]<em>p</em><sub>0</sub>(<em>Y </em>)<em>dY </em>is analytically intractable. The energy function is

<em>.                                                            </em>(13)

<em>p<sub>θ</sub></em>(<em>Y </em>) is an exponential tilting of <em>p</em><sub>0</sub>.

Suppose we observe training examples {<em>Y<sub>i</sub>,i </em>= 1<em>,…,n</em>} from an unknown data distribution <em>P</em><sub>data</sub>(<em>Y </em>). The maximum likelihood learning seeks to maximize the log-likelihood function

<em>.                                                             </em>(14)

If the sample size <em>n </em>is large, the maximum likelihood estimator minimizes the KullbackLeibler divergence KL(<em>P</em><sub>data</sub>k<em>p<sub>θ</sub></em>) from the data distribution <em>P</em><sub>data </sub>to the model distribution <em>p<sub>θ</sub></em>. The gradient of <em>L</em>(<em>θ</em>) is

<em> ,                                           </em>(15)

where E<em><sub>θ </sub></em>denotes the expectation with respect to <em>p<sub>θ</sub></em>(<em>Y </em>). The key to the above identity is that <em><sub>∂θ</sub><u><sup>∂ </sup></u></em>log<em>Z</em>(<em>θ</em>) = E<em><sub>θ</sub></em>[<em><sub>∂θ</sub><u><sup>∂ </sup></u>f<sub>θ</sub></em>(<em>Y </em>)].

The expectation in equation (15) is analytically intractable and has to be approximated by MCMC, such as Langevin dynamics, which iterates the following step:

<em>,                                      </em>(16)

where <em>τ </em>indexes the time steps of the Langevin dynamics, <em>δ </em>is the step size, and <em>U<sub>τ </sub></em>∼ N(0<em>,I</em>) is Gaussian white noise. The Langevin dynamics relaxes <em>Y<sub>τ </sub></em>to a low energy region, while the noise term provides randomness and variability. A Metropolis-Hastings step may be added to correct for the finite step size <em>δ</em>. We can also use Hamiltonian Monte Carlo for sampling the generative ConvNet.

We can run ˜<em>n </em>parallel chains of Langevin dynamics according to (16) to obtain the synthesized examples {<em>Y</em>˜<em><sub>i</sub>,i </em>= 1<em>,…,n</em>˜}. The Monte Carlo approximation to <em>L</em><sup>0</sup>(<em>θ</em>) is

(17)

<em>,</em>

which is used to update <em>θ</em>.

To make Langevin sampling easier, we use mean images of training images as the sampling starting point. That is, we down-sampled each training image to a 1×1 patch, and up-sample this patch to the size of training image. We use cold start for Langevin sampling, i.e., at each iteration, we start sampling from mean images.

Algorithm 2 describes the details of the learning and sampling algorithm.

<strong>Algorithm 2 </strong>Descriptor: real sampling <strong>Input:</strong>

<ul>

 <li>training examples {<em>Y<sub>i</sub>,i </em>= 1<em>,…,n</em>},</li>

 <li>number of Langevin steps <em>l</em>, (3) number of learning iterations <em>T</em>.</li>

</ul>

<strong>Output:</strong>

<ul>

 <li>estimated parameters <em>θ</em>,</li>

 <li>synthesized examples {<em>Y</em>˜<em><sub>i</sub>,i </em>= 1<em>,…,n</em>}.</li>

</ul>

1: Let <em>t </em>← 0, initialize <em>θ</em>.

2: <strong>repeat</strong>

3:               For <em>i </em>= 1<em>,…,n</em>, initialize <em>Y</em><sup>˜</sup><em><sub>i </sub></em>to be the mean image of <em>Y<sub>i</sub></em>.

4:            Run <em>l </em>steps of Langevin dynamics to evolve <em>Y</em><sup>˜</sup><em><sub>i</sub></em>, each step following equation (16).

5: Update <em>θ<sub>t</sub></em><sub>+1 </sub>= <em>θ<sub>t </sub></em>+<em>γ<sub>t</sub>L</em><sup>0</sup>(<em>θ<sub>t</sub></em>), with step size <em>γ<sub>t</sub></em>, where <em>L</em><sup>0</sup>(<em>θ<sub>t</sub></em>) is computed according to equation (17).

6:            Let <em>t </em>← <em>t </em>+ 1.

7: <strong>until </strong><em>t </em>= <em>T</em>

<h2>2.1         TO DO</h2>

For the egret category, learn a descriptor model. Fill the blank part of ./DesNet/DesNet.py. <strong>Show</strong>:

<ul>

 <li>Synthesized images.</li>

 <li>Plot of training loss over iteration.</li>

</ul>