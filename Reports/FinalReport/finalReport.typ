#import "psu-colors.typ": *
#import "@preview/lilaq:0.3.0" as lq

#set math.equation(numbering: "(1)")
#show table.cell.where(y: 0): set text(weight: "bold")
#show smallcaps: set text(font: "Latin Modern Roman Caps")

#let title = "Organ pipe voicing"
#let authors = (
  "Meghana Cyanam",
  "Hai Nguyen",
  "Gabriel Pinochet-Soto",
)

#set document(
  title: title,
  author: authors,
  description: "Final report for the CADES Lab consulting project on organ pipe voicing.",
  keywords: (
    "organ pipe voicing",
    "acoustics",
    "CADES Lab",
    "consulting project",
  ),
)

#set page(
  paper: "a4",
  fill: psu-white,
  numbering: "1",
  supplement: [p.],
  header: [
    #smallcaps[Organ pipe voicing]
    #h(1fr)
  ],
)

#set text(
  font: "Times New Roman",
  size: 12pt,
  fill: psu-black
)

#set par(
  justify: true
)

//-----------------------------------------------------------------------------
#{
  set page(
    numbering: none,
    header: [],
  )
  set align(center)
  v(3cm)
  text(
    size: 20pt,
    weight: "bold",
  )[#title]
  v(0.5cm)
  [by]
  v(0.5cm)
  text(
    size: 14pt,
    weight: "bold",
  )[
    #{
      for author in authors {
        author
        linebreak()
      }
    }
  ]
  v(2fr)
  columns(2)[
    #align(left)[
      #upper[ Professors: ]
      #linebreak()
      Daniel Taylor-Rodriguez
      #linebreak()
      Jacob Schultz
    ]

    #colbreak()

    #align(right)[
      #upper[ Client: ]
      #linebreak()
      Arno Patin
    ]
  ]
  v(1fr)
  [
    Portland State University
    #linebreak()
    CADES Lab
  ]
  pagebreak()
}

//-----------------------------------------------------------------------------

#{
  set page(
    numbering: "i",
    header: [],
  )
  set align(center)
  text(
    size: 14pt,
    weight: "bold",
  )[Executive Summary]
  set align(left)
  [
    We found some _a priori_ trends in the data that suggest a bias in the
    measurements; we suggest that the data is not sufficient to fit larger models.
    We present some techniques to interpret optimal partial distribution.
    Deep Neural Networks (DNN) are used to predict the partial ratios
    distribution, that take into account the airflow rate, and use the Ising
    number as a constraint.
    Finally, we propose prospective future work; we recommend the collection of
    evenly distributed data and the use of the DNN model as a _companion_ tool
    that can be enhanced with new data.
  ]
  pagebreak()
}

#{
  set page(
    numbering: "i",
    header: [],
  )
  outline()
  pagebreak()
  outline(
    title: "List of Figures",
    target: figure.where(kind: image),
  )
  pagebreak()
}


//-----------------------------------------------------------------------------

#set heading(numbering: "1.1.1")

= Introduction

Pipe organs have been around for hundreds of years, and making them sound just
right is a special job called intonation, or _voicing_.
This means adjusting each pipe to create the best sound for the organ.
The way a pipe looks and is built --like its size, shape, and material-- can
change how it sounds.
These details are decided first, and then the pipes are fine-tuned before being
added to the organ.

Our goal is to understand how the *Ising number* --a non-dimensional quantity designed
to describe the proportions of the pipe concisely-- is related to the voicing
of the organ pipes _upon modification of the airflow_.

- Make the Ising number more accurate.
- Include the flow rate in the Ising number.
- Create a model that helps us design pipes before we start intonating them.

== Project objectives and goals

The main objective of this project concerns the improvement of the organ pipe voicing
process.
As the organ pipe design is a complex process, and usually is not a reversible
process, it is important to ensure that the design of the pipes is done
correctly; this presents a key constraint for the intonation process.
The parameters involved are various, which makes learning about this complex
system a difficult task; the understanding of these parameters and their
interactions would facilitate the harmonization process.

A particular goal of this project is to improve the Ising
formula @1971Ising-1, which reads
$
sans(I)
= frac(v,omega) sqrt(frac(d,h^3))
= frac(1,omega) sqrt(frac(2 P d,rho h^3)),
$ <ising>
where $sans(I)$ is the Ising number, $omega$ is the target frequency of the
pipe, $v$ is the jet initial velocity, $d$ is the jet initial thickness, $h$ is
the cut-up height (or the length of the mouth), $P$ is the blowing pressure
--where we make use of Bernoulli's Law, cf. @2025Lilj-1)--, and $rho$ is the density
of the air.
In @1971Ising-1 and @2025Lilj-1, it is stated that the Ising number should be
a number close to $sans(I) approx 2$ in order to have optimal sound.
We propose two modifications to this formula, as it does not take into account
the airflow rate, which is a crucial parameter in the organ pipe design.

A second goal of this project is to provide a model that can be used to
predict the Ising number and the partials of the pipe, while also integrating
the airflow rate and all the remaining control parameters of the pipe.
For this, we will use a Deep Neural Network (DNN) model that can be trained
on the data provided by the client.
This model will be able to predict the Ising number and the partials of the pipe
based on the parameters of the pipe, such as the diameter of the toe, the cut-up
height, the fundamental frequency, the acoustic intensity, and the
airflow rate.

== Summary of results

Our main results correspond to a better understanding of the correlation of the data,
the limitations that the data itself presents, the distribution of the partials --in terms of
what would be an optimal distribution--, and exploratory analysis of the relationship
between the Ising number, the parameters of the organ pipe, and the airflow rate.

We also provide a Deep Neural Network (DNN) model that can be used to predict the Ising number
while also predicting the partials of the pipe, but integrating the airflow rate and all the remaining
control parameters of the pipe.
This document will summarize some instructions on how to use the model and interpret its results.

The reminder of this document is structured as follows: 
In @dataDescription, we describe the data that was provided by the client and the
observations that we made during the data analysis.
In @methodology, we describe the methodology that we used to analyze the data
and to build the model that predicts the Ising number and the partials of the
pipe; we also present the results of the model.
In @conclusions, we summarize the main findings of the project and present some
potential alternatives to the approach that we took to solve the problem of organ pipe voicing.
We include some appendices with additional information:
- @alternatives[Appendix] presents some potential alternatives to the approach that we took
  to solve the problem of organ pipe voicing.
- @dnn_implementation[Appendix] presents some implementation details of the DNN model that we used to
  predict the Ising number and the partials of the pipe.

#pagebreak()

= Data Description and Data Analysis <dataDescription>

In this section, we describe the data that was provided by the client and
the observations that we made during the data analysis.

== Data description

The main source of data is provided by the client.
The data is currently available in a `csv` file, and it contains the
following columns:
 - `isBourdon` (`boolean`) -- Indicates if the pipe is a Bourdon pipe.
 - `flueDepth` (`float`) -- The depth of the flue.
 - `frequency` (`float`) -- The frequency of the pipe.
 - `cutUpHeight` (`float`) -- The cut-up height of the pipe.
 - `diameterToe` (`float`) -- The diameter of the toe.
 - `acousticIntensity` (`float`) -- The acoustic intensity of the pipe.
 - `partialN` (`float`) -- $N$~th partial of the pipe.
        The number of partials is not fixed, and it can vary from 1 to 8.
        This value is bounded between 0 and 100.

We also acknowledge new data that was provided by the client during the
project that pertains to new measurements with the same protocol, yet displaying
a maximum observed partial `maxPartialN` and a minimum observed partial `minPartialN`.
For the ease of the analysis, we merged the two datasets into a single one, where
each new observation represents two rows in the dataset, one for each of the
minimum and maximum partials.

Finally, the client provided a detailed description of the dimensioning of the pipes
and of the system that allows the air circulation through the pipes.
Due to time constraints, we were not able to fully explore this data, but we will
mention some potential avenues for future work in the conclusion section.

== Data analysis

We first start with the process of data wrangling.
We do not include constants or parameters that are given by the client --say,
the blowing pressure, the density of the air, or the jet initial velocity-- in
the Ising number formula (@ising) as they are not being modified during the
voicing process.
Rather, we focus on the observable parameters that can be measured, as described
above.
Recall that these quantities are:
- `isBourdon` -- Indicates if the pipe is a Bourdon pipe.
- `flueDepth` -- The depth of the flue.
- `frequency` -- The frequency of the pipe.
- `cutUpHeight` -- The cut-up height of the pipe.
- `diameterToe` -- The diameter of the toe.
- `acousticIntensity` -- The acoustic intensity of the pipe.
- `partialN` -- The $N$ th partial of the pipe.
We assume standard units (metric system) for all the quantities.
The remaining environmental parameters are assumed to be constant, as provided by
the client, and are not included in the dataset.

The remaining quantities are computed under the required formulae (Ising number,
velocities, circular areas, etc.) and are _not_ included in the dataset.

We begin by computing the Ising number for each observation in the dataset;
we correlate the predictors associated with the organ pipe parameters with themselves,
and then, we correlate the predictors with the responses (the Ising number
and the partials). See @corrPP and @corrPR for the correlation matrices
of the predictors and the predictors with the response, respectively.

We notice that `isBourdon` is not as much correlated with the other predictors, yet it
highly correlated with `partial2`.
The main dimensioning parameters of the pipe, such as `cutUpHeight` and `flueDepth`, 
are highly correlated with the fundamental frequency of the pipe, `frequency`.
The Ising number does not have high correlation with the other predictors.


#figure(
  image(
    "PP.png",
    format: "png",
    width: 13cm,
  ),
  caption: "Correlation matrix of the predictors.",
) <corrPP>

#figure(
  image(
    "PR.png",
    format: "png",
    width: 13cm,
  ),
  caption: "Correlation matrix of the predictors and the response.",
) <corrPR>

#{
  let allOrgan = csv("../../Data/allOrgan.csv", row-type: dictionary)
  let nObservations = allOrgan.len()
  let frequency = ()
  let cutUpHeight = ()
  let diameterToe = ()
  for i in range(nObservations) {
    frequency.push(float(allOrgan.at(i).frequency))
    cutUpHeight.push(float(allOrgan.at(i).cutUpHeight))
    diameterToe.push(float(allOrgan.at(i).diameterToe))
  }

  figure(
    scale(x: 130%, y: 130%)[
      #v(1cm)
      #lq.diagram(
        lq.scatter(
          frequency,
          cutUpHeight,
          color: diameterToe,
          norm: t => calc.min(calc.max(t , -0.01), 0.01),
        ),
        xlabel: [Frequency (Hz)],
        yscale: "log",
        ylabel: [Cut-Up Height ($log$ m)],
      )
      #v(1cm)
    ],
    caption:"Cut-Up Height vs. Frequency, colored by Diameter Toe."
  ) 
} <cutUpHeightVsFrequency>

@cutUpHeightVsFrequency shows a *strong bias* in the data, as the selection of
the parameters is not evenly distributed in the parameter space.
In order to have a better understanding of the data, we suggest that the client
attempts to collect _evenly distributed_ data in the parameter space.

== A note on data formatting

We decided to use standard `csv` files to store the data, as they are easy to
read and write with packages like
#link("https://pandas.pydata.org/", "Pandas") or
#link("https://www.pola.rs/", "Polars").

We suggest using the following labels for the columns in the data files:
- `isBourdon`: a boolean (represented with an integer 0 or 1) indicating whether the pipe is a bourdon pipe (1) or not (0).
- `flueDepth`: the depth of the flue of the pipe, in meters.
- `frequency`: the frequency of the fundamental of the pipe, in Hertz.
- `cutUpHeight`: the height of the cut-up of the pipe, in meters.
- `diameterToe`: the diameter of the toe-hole of the pipe, in meters.
- `acousticIntensity`: the acoustic intensity of the pipe, in decibels.
- `partial1`, `partial2`, ..., `partial8`: the partials of the pipe, adimensional values.

We have most of the data in the `/Data` folder of the 
#link("https://github.com/homeomorfismo/STAT409Project")[GitHub repository].
See, for instance, `allOrgan.csv` for a complete dataset.
```csv
isBourdon,flueDepth,frequency,cutUpHeight,diameterToe,acousticIntensity,
partial1,partial2,partial3, partial4,partial5,partial6,partial7,partial8
1,0.1,440.0,0.05,0.02,80.0,99.0,97.0,97.0,81.0,64.0,45.0,30.0,12.0
0,0.15,220.0,0.07,0.03,85.0,34.0,99.0,17.0,11.0,59.0,28.0,0.0,32.0
```

Despite the Ising number is adimensional, it is *unit-dependent*.
Make sure to use the same units for all the data in the dataset:
the range of "intonation" values may vary with different units.

#pagebreak()

= Methodology <methodology>

In this section, we describe the methodology that we used to analyze the data
and to build the model that predicts the Ising number and the partials of the
pipe.

== Linear regression analysis for partials distribution

We used linear regression to study the relationship between the frequencies and
their partials --which are normalized intensities.
Three heuristic shape functions are used to fit the partials distribution:
$
frak(p)_sans("lin")(omega; a, b) & = a omega + b, \ 
frak(p)_sans("exp")(omega; a, b) & = a e^omega + b, \
frak(p)_sans("log")(omega; a, b) & = a log(omega) + b,
$
where $frak(p)$ is the partial (ratio), $omega$ is the frequency, and $a, b in bb(R)$ are shape
constants for fitting the data.

As the data is not evenly distributed, linear regression is not the best
approach to fit the data.
The _a priori_ known correlation between the second partial and the Bourdon pipes
(see, e.g., @2012RosFle-1) suggest that fitting these *monotonic* functions 
might not be the best approach.

== Proposed modifications to the Ising number

Two modifications to the Ising number formula are proposed, in order to
take into account the airflow rate.

=== Naive approach

A natural heuristic is the following: provided no airflow, there should be no need to modify the Ising number; in case of airflow, we expect to increase the Ising number, as the pipe is expected to _overblow_ at some point.

Let $sans(S)$ be the airflow of the wind jet on the toe-hole of the pipe.
We propose a first *naive modified Ising number* as follows:
$
sans(I)_sans("naive") = sans(I) ( 1 + frac(1,2) sans(S) ).
$

This approach is simple, but *breaks the dimensional analysis*, as it forces the factor $frac(1,2)$ to have dimensions of the reciprocal of the airflow.

More statistical analysis was performed with this proposed modification, but it was not conclusive.
See, e.g., @accIntVsModIsing for a plot of the acoustic intensity vs. the modified Ising number.

#{
  let accIntModIsing = csv("../../Data/accousticIntensityModIsing.csv", row-type: dictionary)
  let accInt = ()
  let modIsing = ()
  for i in range(accIntModIsing.len()) {
    accInt.push(float(accIntModIsing.at(i).acousticIntensity))
    modIsing.push(float(accIntModIsing.at(i).modifiedIsing))
  }
  figure(
    lq.diagram(
      lq.scatter(
        accInt,
        modIsing,
      ),
      xlabel: [Acoustic Intensity (dB)],
      ylabel: [Modified Ising number],
    ),
    caption:"Acoustic Intensity vs. Modified Ising number.",
  ) 
} <accIntVsModIsing>

=== Flow-informed Ising number

A more sophisticated approach is to use the assumption that, on low velocities, the airflow is non-compressible.
We can use conservation of the airflow to modify the Ising number as follows: given two transversal areas $A_1$ and $A_2$ of the pipe system, and two velocities $v_1$ and $v_2$ of the wind jet, we have:
$
v_1 A_1 = v_2 A_2.
$

Thus, by assuming _the velocity from the system is known_, we can modify the Ising number as follows:
Let $ v_serif("toe") = frac(A_serif("system"), A_serif("toe")) v_serif("system") $ be the velocity of the wind jet at the toe-hole of the pipe;
then the *flow-informed Ising number* is defined as:
$
sans(I)_sans("flow-informed") = frac(v_serif("system") A_serif("system"),omega A_serif("system") ) sqrt(frac(d,h^3))
= frac(A_serif("system"),omega A_serif("toe") ) sqrt(frac(2 P d,rho h^3)).
$

== Deep Neural Network model

A deep neural network (DNN) is a type of machine learning model that
can learn complex relationships between inputs and outputs.
The JAX framework @2018Jax-1, @2020Optax-1, and @2024Flax-1 was used to implement
the model.
The architecture of the DNN was made simple due to the limited amount of data
available.
We defer technical details of the implementation to @dnn_implementation[Appendix].

For installation instructions, see the 
#link("https://github.com/homeomorfismo/STAT409Project/blob/main/Codes/README.md", `README.md`) on GitHub.

=== Architecture of the model

In @dnn, the diagram shows the architecture of the DNN model that we used to
predict the Ising number and the partials of the pipe.

In a more technical way, the DNN model corresponds to a function
$sans("DNN"): bb(R)^6 -> bb(R)^9$.
An input vector $bold(x) in bb(R)^6$ corresponds to the six parameters of the
pipe `isBourdon`, `flueDepth`, `frequency`, `cutUpHeight`, `diameterToe`, and
`acousticIntensity`.
The output vector $bold(y) in bb(R)^9$ corresponds to the Ising number and the
partials ratios of the pipe.
The first component of the output vector, $bold(y)_0 = sans("DNN")_sans("I")(bold(x))$,
is the Ising number prediction, and the remaining components
$(bold(y)_1, dots, bold(y)_8) = sans("DNN")_sans("P")(bold(x))$
are the partials ratios predictions.

#figure(
  scale(x: 100%, y: 100%)[
    #include "dnn.typ"
    #v(1cm)
  ],
  caption:"A Deep Neural Network (DNN) approach to predict the Ising number.",
) <dnn>

== Utilization of the model

A command line interface (CLI) is provided to interact with the DNN model.
The CLI allows users to perform three main tasks:
- Run the DNN model for training or evaluation.
- Classify samples to generate a rating of "optimal" partials ratios.
- Convert units of the parameters using Pint @2025Pint-1.

Usually, on a terminal, the code would be executed as follows:
```bash
run_model \
    --learning_rate <initial_learning_rate> \
    --epoch <epochs> \
    --data_path <path_to_data_in_csv>
```
where `run_model` is the name of the script that runs the DNN model, `--learning_rate`
is the initial learning rate for the model, `--epoch` is the number of epochs to
train the model, and `--data_path` is the path to the data in `csv` format.

=== Other features

Other features were included in our Python package, such as:
- Unit conversion `convert_units`
- Sample classification `classify_samples`
The client can get a description of the utilization of these applications by
running `convert_units --help` or `classify_samples --help`.

An example of how to classify samples based on a fundamental frequency of 440 Hz (A4 note), is as follows:
```bash
classify_samples \
  --frequency 440 \
  --output-dir ./audio_samples \
  --duration 2.0 \
  --samplerate 44100 \
  --samples 10 \
  --save-samples \
  --plot-samples
```

== Results of the model

We have trained the model using 300 epochs, with an initial learning rate of 0.01.
Under the provided hyperparameters (see @dnn-hyperparams), the model converged
we report a final total average loss of \(0.0393\).
Other statistics are displayed on screen upon running the model.
A verification step on a test set of the data is performed, and the results are
displayed on screen.

The model displays difficulties in learning the second partial distribution, accurately, despite being able to have access to the model.

#pagebreak()

= Conclusions <conclusions>

While our findings provide some valuable insights into the correlation between the parameters defining the organ pipes
and the Ising number, the role of airflow rate remains fairly uncertain.
@cutUpHeightVsFrequency shows a strong bias in the data, as the selection of
the parameters is not evenly distributed in the parameter space; we suggest that the client
attempts to collect more data in an evenly distributed manner in the parameter space.
We understand that this procedure may be counterintuitive --even contraproductive-- to the
voicing process from a production point of view (as it may require constructing pipes that are not intended to be used),
yet we believe that this would allow to have a better understanding of the system itself.

We propose some way to integrate the new available data concerning the airflow and dimension of the wind jet system.
In @alternatives[Appendix], we present some potential alternatives to the approach that we took; in particular,
the modification to the Ising number utilizing conservation of the airflow seems to be the most promising one to pursue,
provided that the airflow rates are available.

Finally, we present a Deep Neural Network (DNN) model.
The model is trained on the data provided by the client, and it can be used to predict the Ising number
and the partials of the pipe.
This model is not robust enough to be used as a standalone tool, but it can be used as a
_companion_ tool that can be enhanced with new data.
In @alternatives[Appendix], we highlight that the inclusion of the airflow rate in the model
could be a promising avenue for future work.
In @dnn_implementation[Appendix], we present some implementation details of the DNN model that we used to
predict the Ising number and the partials of the pipe.

//------------------------------------------------------------------------------

// Appendices

#pagebreak()

#set heading(numbering: "A.1")
#counter(heading).update(0)

= Potential alternatives <alternatives>

Here we present some potential alternatives to the approach that we took
to solve the problem of organ pipe voicing.

== Finite Element Method

The Finite Element Method (FEM) is a numerical method for solving partial
differential equations (PDEs) and integral equations.
As the governing equations of the organ pipe are PDEs (such as the wave
equation or the Helmholtz equation), it is plausible to use the FEM to solve
them.

Standard techniques and discretization methods for the FEM
are available, and we will not delve into those details here.
This approach could be very rewarding if implemented correctly, as the client could
_virtually_ model the organ pipe --using software such as CAD or OpenCascade-- and
then use the FEM to simulate the behavior of the pipe under different conditions.

For more details on the FEM, we refer the reader to the literature on numerical
methods, as @2021ErnGue-1 and @2021ErnGue-2.
As we are concerned with wave propagation in the _whole space_, a truncation to a
computational domain is required: this can be done using a perfectly matched layer
(PML), more details can be found in @2019VazKeeDem-1.

== Generalized additive models

Generalized Additive Models (GAMs) offer a robust statistical framework
for modeling complex, nonlinear interactions between the multiple pipe parameters
provided by the client.
GAMs allow for flexible modeling of different relationships between the
data; one promising approach is to use GAMs to model the relationship between the
data and the partial distribution.
Some of this exploratory work is available in our
#link("https://github.com/homeomorfismo/STAT409Project")[GitHub repository].

== Partial distribution in natural scales

Currently, the data is decoupling the partials (which are ratios) from the
acoustic intensity.
We propose to use the partials in their natural scales, i.e., the partials
should be amplified by the acoustic intensity.

This means the partials keep their physical meaning, and a different type of
normalization can be used, that doesn't force a partial to be (close to) 1.

== Integrating the modified Ising number in the DNN model

Currently, the DNN model is trained to predict the Ising number and the
partials of the pipe, and uses the diameter of the toe as a predictor.
There is a potential to integrate one of the suggested modifications to the
Ising number formula (@ising) into the DNN model.

#pagebreak()

= Deep Neural Network model implementation details <dnn_implementation>

We explain some implementation details of the DNN model that we used to
predict the Ising number and the partials of the pipe.
Most of these decision were heuristic.

+ The data was rescaled _on the fly_ during the training of the DNN model: we consider values between 0 and 1 for partial ratios.
+ A Batch Normalization layer is used to normalize the inputs of the DNN model. This means that the inputs are normalized to have a mean of 0 and a standard deviation of 1.
+ The DNN model was trained using the Adam optimizer @2014Kin-1. See @2020Optax-1 for implementation details.
  - We use gradient clipping to avoid exploding gradients.
  - We impose weights decay to avoid overfitting.
  - A cosine decay function is used as a learning rate schedule @2016LosHut-1.
+ Different activation functions are used in the DNN model:
  - Hidden layers use $sans("tanh")(x) = frac(e^x - e^(-x), e^x + e^(-x))$ due to be zero-centered and smooth.
  - The output layer is a combination of two activation functions:
    - The Ising number requires a $sans("softplus")(x) = log(1 + e^x)$ activation function, as it is always positive.
    - The partial ratios use a $sans("softmax")(bold(x))_i = frac(e^(x_i), sum_(j)e^(x_j))$ activation function, as the rations must be in the range $[0, 1]$.
+ A weighted mean squared error (MSE) loss function is used to train the DNN model. See @dnn-loss for details.
  - The Ising number prediction is weighted by a hyperparameter $lambda^2$ that controls the importance of the Ising number prediction.
+ The main hyperparameters used in the DNN model are summarized in @dnn-hyperparams.

The loss function is defined as follows:
given observational data ${(bold(x)_i, bold(p)_i)}_{i=1}^N$ where $bold(x)_i in bb(R)^6$ and $bold(p)_i in [0, 1]^8$ are the six parameters of the pipe and the partial ratios, respectively, the loss function is defined as
$
cal(L)_sans("DNN")(bold(W), bold(b); lambda)
=
frac(1,N) limits(sum)_(i=1)^N ( lambda^2 ( sans("DNN")_sans("I")(bold(x)_i) - sans("I")(bold(x)_i) )^2 + sum_(j=1)^8 ( sans("DNN")_sans("P")(bold(x)_i)_j - bold(p)_(i,j) )^2 ),
$ <dnn-loss>
where $cal(L)_sans("DNN")$ is the loss function, $bold(W)$ and $bold(b)$ are the weights and biases of the DNN model, $lambda$ is a hyperparameter that controls the importance of the Ising number prediction, $N$ is the number of training samples, and $sans("DNN")_sans("I")$ and $sans("DNN")_sans("P")$ are the DNN model predictions for the Ising number and the partials, respectively.

#figure(
  table(
    columns: 3,
    align: (left, left, left),
    fill: (_, y) => if calc.odd(y) { psu-electric-green.lighten(90%) },
    stroke: none,
    table.header[Hyperparameter][Value][Type],
    [$N_sans("hidden")$ (number of hidden layers)],[2],[`int`],
    [$d_sans("hidden")$ (dimension of hidden layers)],[16],[`int`],
    [$sigma$ (standard deviation bias initialization)],[0.325],[`float`],
    [$lambda$ (importance of Ising number prediction)],[2.5],[`float`],
    [$n_sans("decay")$ (number of decay steps)],[10],[`int`],
    [$alpha_sans("decay")$ (decay convex factor)],[0.875],[`float`],
    [$p_sans("decay")$ (decay power factor)],[1.125],[`float`],
    [$epsilon$ (division regularization)],[1.0e-8],[`float`],
    [$beta_{1,sans("Adam")}$ (Adam first moment)],[0.95],[`float`],
    [$beta_{2,sans("Adam")}$ (Adam second moment)],[0.9995],[`float`],
    [$tau_sans("clip")$ (gradient clipping factor)],[1.0],[`float`],
    [$tau_sans("weight")$ (weight decay factor)],[1.0e-2],[`float`],
    [$rho$ (weight regularization factor)],[1.0e-4],[`float`],
  ),
  caption: "Hyperparameters used in the DNN model.",
) <dnn-hyperparams>

//------------------------------------------------------------------------------

#pagebreak()

#set heading(numbering: none)

#bibliography("refs.bib", style: "chicago-author-date")
