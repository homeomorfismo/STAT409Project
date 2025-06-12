#import "psu-colors.typ": *
#import "psu-beamer.typ": *

#import "@preview/lilaq:0.3.0" as lq
#import "@preview/cetz:0.3.4" as cetz

#show: psu-theme.with(
  title: "Organ Pipe Voicing and Flow-Informed Ising Number",
  author: "Meghana Cyanam, Hai Nguyen, and Gabriel Pinochet-Soto",
  institute: "Portland State University",
  date: datetime.today().display(),
)

#psu-title-slide(
  title: "Organ pipe voicing and flow-informed Ising number",
  subtitle: "Team C - CADES Lab",
  author: "Meghana Cyanam, Hai Nguyen, and Gabriel Pinochet-Soto",
  institute: "Portland State University", 
  date: datetime.today().display(),
  left-logo: image("PSU_logo_white_transparent.png", width: 10cm),
)

#pagebreak()

= Table of Contents

#psu-outline()

#pagebreak()

= Introduction

== On the art of organ voicing

+ Pipe organs have been around for hundreds of years, and making them sound just right is a special job called intonation, or _voicing_.
+ The way a pipe looks and is built -- like its size, shape, and material-- can change how it sounds.
+ These details are decided first, and then the pipes are fine-tuned before being added to the organ.

== Main goals

#columns(2)[
  #psu-block(title: "Ising Number as a predictor", fill-color: psu-purple)[
    - Ideally, we would like to use *the Ising number* to predict the intonation of a pipe.
    - Using a data-driven approach, we aim to correct the formulation of the Ising number, hopefully to integrate the flow of the wind jet in the pipe and improve the prediction of the intonation.
  ]

  #colbreak()

  #psu-block(title: "Predict good distribution of partials", fill-color: psu-purple)[
    - In general, it is desirable to have an understanding of *the distribution of the (intensity of the) partials of a pipe*.
    - Being able to model and predict the distribution of the partials of a pipe is a key aspect of voicing.
  ]
]

#pagebreak()

== Good voicing vs. bad voicing

#let freqs = range(8)
#let goods = (99, 97, 97, 81, 64, 45, 30, 12)
#let bads = (34, 99, 17, 11, 59, 28, 0, 32)

#columns(2)[
  #lq.diagram(
    width: 9cm,
    xaxis: (
      ticks: ("1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th").enumerate(),
      subticks: none,
    ),
    lq.bar(freqs, goods, fill: psu-electric-green),
  )

  #psu-block(title: "Good voicing", fill-color: psu-electric-green)[
    Good voicing is when the pipe *sounds good*, and the sound of the fundamental is clear and pleasant.
    It often has a monotone decay.
  ]

  #colbreak()

  #lq.diagram(
    width: 9cm,
    xaxis: (
      ticks: ("1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th").enumerate(),
      subticks: none,
    ),
    lq.bar(freqs, bads, fill: psu-red),
  )

  #psu-block(title: "Bad voicing", fill-color: psu-red)[
    Bad voicing is when the pipe *sounds off*, and the sound of the fundamental is not clear.
    It often has a noisy decay, and the fundamental gets _shifted_ to a different frequency.
  ]
]

#pagebreak()

== The Ising number

#columns(2)[
  Define the *Ising number* of a pipe as follows:
  $
  sans(I) = frac(v,omega) sqrt(frac(d,h^3))
  = frac(1,omega) sqrt(frac(2 P d,rho h^3)).
  $
  Here:
  - $v$ is the velocity of the wind jet,
  - $omega$ is the frequency of the fundamental,
  - $d$ is the initial jet diameter,
  - $h$ is the cut-up height of the pipe,
  - $P$ is the pressure of the wind jet, and
  - $rho$ is the density of the air.

  #colbreak()

  // Add Tikz-like picture.
  #figure(
    scale(x: 130%, y: 130%)[
      #cetz.canvas({
        import cetz.draw: *
        // Mouth
        line((3, 0.5), (0, 0.5))
        line((0, 0.5), (1, 0))
        line((1, 0), (3, 0))
        // Cut-up
        line((-2, 0), (-1, 0))
        line((-1, 0), (-1, 0.5))
        line((-1, 0.5), (-2, 0.5))
        // Remaining body
        line((-2, 0.75), (-1, 0.75))
        line((-1, 0.75), (-1, 1.75))
        line((-1, 1.75), (3, 1.75))
        // Distances
        line((-1, -0.2), (0, -0.2), stroke: (paint: psu-purple, dash: "dashed"), name: "cutUpHeight")
        line((-1, -0.1), (-1, -0.3), stroke: (paint: psu-purple))
        line((0, -0.1), (0, -0.3), stroke: (paint: psu-purple))
        content("cutUpHeight.mid", [$h$], anchor: "north")
      })
      #v(1cm)
    ],
    caption:"Diagram of a transversal section of an (squared) organ pipe. The remaining unlabeled opening corresponds to the jet opening.",
  )
]

#pagebreak()

= Semi-analytical methods

We introduce some formula-based approaches to modify the Ising number to account for the flow of the wind jet in the pipe.

== Naive approach

A natural heuristic is the following: provided no airflow, there should be no need to modify the Ising number; in case of airflow, we expect to increase the Ising number, as the pipe is expected to _overblow_ at some point.

#psu-block(title: "Naive modified Ising number")[
  Let $sans(S)$ be the airflow of the wind jet on the toe-hole of the pipe.
  We propose a first *naive modified Ising number* as follows:
  $
  sans(I)_sans("naive") = sans(I) ( 1 + frac(1,2) sans(S) ).
  $
]

This approach is simple, but *breaks the dimensional analysis*, as it forces the factor $frac(1,2)$ to have dimensions of the reciprocal of the airflow.

#pagebreak()

== Flow-informed Ising number

A more sophisticated approach is to use the assumption that, on low velocities, the airflow is non-compressible.
We can use conservation of the airflow to modify the Ising number as follows: given two transversal areas $A_1$ and $A_2$ of the pipe system, and two velocities $v_1$ and $v_2$ of the wind jet, we have:
$
v_1 A_1 = v_2 A_2.
$

Thus, by assuming _the velocity from the system is known_, we can modify the Ising number as follows:
#psu-block(title: "Flow-informed modified Ising number")[
  Let $ v_serif("toe") = frac(A_serif("system"), A_serif("toe")) v_serif("system") $ be the velocity of the wind jet at the toe-hole of the pipe;
  then the *flow-informed Ising number* is defined as:
  $
  sans(I)_sans("flow-informed") = frac(v_serif("system") A_serif("system"),omega A_serif("toe") ) sqrt(frac(d,h^3))
  = frac(A_serif("system"),omega A_serif("toe") ) sqrt(frac(2 P d,rho h^3)).
  $
]

#pagebreak()

= Exploratory data analysis

#let allOrgan = csv("../../Data/allOrgan.csv", row-type: dictionary)
#let flowIsingData = csv("../../Data/flowIsingData.csv", row-type: dictionary)
#let freqCutUpHeightDiam = csv("../../Data/freqCutUpHeightDiam.csv" , row-type: dictionary)
#let accousticIntensityModIsing = csv("../../Data/accousticIntensityModIsing.csv", row-type: dictionary)

#columns(2)[
  #{
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
  }

  #colbreak()

  #psu-alert-block(title: "Observations")[
    - The cut-up height is *logarithmic* in the frequency.
    - The diameter toe is *almost constant* across the observations.
    - The data is *not very dense*, and has a bias.
  ]

  #psu-block(title: "Variability of the data", fill-color: psu-purple)[
    We would like to see a more evenly distributed dataset, with more variability in the parameters.
    In this way, we can train a model that is more robust to different conditions, and identify _good *and* bad_ voicing
    patterns.
  ]
]

#pagebreak()

#columns(2)[
  #figure(
    image(
      "PP.png",
      format: "png",
      width: 13cm,
    ),
    caption: "Correlation matrix of the predictors.",
  )

  #colbreak()

  #figure(
    image(
      "PR.png",
      format: "png",
      width: 13cm,
    ),
    caption: "Correlation matrix of the predictors and the response.",
  )
]

// #{
//   let corMatPredPred = csv("../../Data/CorMatPredPred.csv", row-type: dictionary)
//   let rowPP = ()
//   let dataPP = ()
//   for i in range(corMatPredPred.len()) {
//     rowPP.push(corMatPredPred.at(i).label)
//   }
//   for i in range(corMatPredPred.len()) {
//     let temp = ()
//     for j in rowPP {
//       temp.push(float(corMatPredPred.at(i).at(j)))
//     }
//     dataPP.push(temp)
//   }
// 
//   figure(
//     scale(x: 130%, y: 130%)[
//       #v(1cm)
//       #lq.diagram(
//         lq.colormesh(
//           range(rowPP.len()),
//           range(rowPP.len()),
//           dataPP,
//           interpolation: "pixelated",
//         ),
//         xlabel: [Predictors],
//         ylabel: [Predictors],
//       )
//       #v(1cm)
//     ],
//     caption:"Correlation matrix of the predictors.",
//   )
// }
// #{
//   let corMatPredResp = csv("../../Data/CorMatPredResp.csv", row-type: dictionary)
//   let rowPR = ()
//   let colPR = ()
//   let dataPR = ()
// }

#pagebreak()

= A deep neural network approach

== Glance into its architecture

#v(1.5cm)
#figure(
  scale(x: 130%, y: 130%)[
    #include "dnn.typ"
    #v(1cm)
  ],
  caption:"A Deep Neural Network (DNN) approach to predict the Ising number.",
)

#pagebreak()

== Utilization

For installation instructions, see the 
#link("https://github.com/homeomorfismo/STAT409Project/blob/main/Codes/README.md", `README.md`) on GitHub.

#columns(2)[
  #psu-alert-block(title: "Running the model")[
    ```bash
    run_model --help
    run_model --mode train \
      --data_path <path_to_data> \
      --save_path <path_to_save_model> \
      --load_path <path_to_load_model>
    run_model --mode evaluate \
      --data_path <path_to_data> \
      --load_path <path_to_load_model>
    ```
    The model is trained on the data provided in the `data_path` and the weights (i.e., the information that the model learns) are saved in the `save_path`.
    The model can be evaluated on the data provided in the `data_path` and the weights are loaded from the `load_path`.
  ]

  #colbreak()

  #psu-block(title: "Under the hood")[
    Essentially, we would like to fit a model that verifies the approximations:
    $
    sans(I)_i approx sans("DNN")_sans("I")_i, text(" and ")
    p_i approx sans("DNN")_sans("P")_i,
    $
    for all observations $i$ in the dataset.
    Thus, *the distance between what the model predicts and the observed partials and Ising numbers is minimized*.
  ]

  #psu-block(title: "Where to find this code", fill-color: psu-purple)[
    Check the code in #link("https://github.com/homeomorfismo/STAT409Project", [*our GitHub repository*]).
    There are some instructions in the `Codes/README.md` file.
  ]
]

#pagebreak()

#columns(2)[
  #psu-block(title: "Other commands")[
    - You can check some units conversion: `convert_units --help`
    - You can classify different spectum distributions: `classify_samples --help`
  
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
  ]

  #colbreak()

  == Formatting the data

  We decided to use standar `csv` files to store the data, as they are easy to read and write with packages like #link("https://pandas.pydata.org/", "Pandas") or #link("https://www.pola.rs/", "Polars").
  We suggest to use the following labels for the columns in the data files:
  - `isBourdon`: a boolean (represented with an integer 0 or 1) indicating whether the pipe is a bourdon pipe (1) or not (0).
  - `flueDepth`: the depth of the flue of the pipe, in meters.
  - `frequency`: the frequency of the fundamental of the pipe, in Hertz.
  - `cutUpHeight`: the height of the cut-up of the pipe, in meters.
  - `diameterToe`: the diameter of the toe-hole of the pipe, in meters.
  - `acousticIntensity`: the acoustic intensity of the pipe, in decibels.
  - `partial1`, `partial2`, ..., `partial8`: the partials of the pipe, adimensional values.
]

#pagebreak()

#psu-block(title: "Example of a data file")[
  We have most of the data in the `/Data` folder of the repository.
  See, for instance, `allOrgan.csv` for a complete dataset.
  ```csv
  isBourdon,flueDepth,frequency,cutUpHeight,diameterToe,acousticIntensity,partial1,partial2,partial3,
  partial4,partial5,partial6,partial7,partial8
  1,0.1,440.0,0.05,0.02,80.0,99.0,97.0,97.0,81.0,64.0,45.0,30.0,12.0
  0,0.15,220.0,0.07,0.03,85.0,34.0,99.0,17.0,11.0,59.0,28.0,0.0,32.0
  ```
]

#psu-alert-block(title: "Standarize your units!")[
  Despite the Ising number is adimensional, it is *unit-dependent*.
  Make sure to use the same units for all the data in the dataset:
  the range of "intonation" values may vary with different units.
]

=== How to enhance the model

It will be convenient to:
- Use a larger dataset, with more observations and *variability* in the parameters.
- Concatenate _new_ observations to the previous ones, to improve the model.

#pagebreak()

= Final remarks

== Potential further work

- Purely physical approach to the computation of the intensity: we want to model the partials without creating a pipe. Using FEM or CFD methods to compute the intensity of the partials.
- Explore flow-informed Ising number and cross-validate with the data: there are new measurements related to the opening of the channels and pallets.
