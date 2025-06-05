#import "psu-colors.typ": *
#import "@preview/cetz:0.3.4": canvas, draw
#import "@preview/cetz-plot:0.1.1": plot
#import draw: line, circle, content, group, bezier, translate, rect

#canvas({
  let layer-sep = 2.5 // Horizontal separation between layers
  let node-sep = 1.0 // Vertical separation between nodes (smaller for more nodes)
  let arrow-style = (mark: (end: "stealth", scale: 0.25), stroke: none, fill: none)

  // Helper function to draw a neuron
  let neuron(pos, fill: psu-white, label: none, name: none, size: 0.4) = {
    content(
      pos,
      if label != none { text(size: 7pt)[$#label$] },
      frame: "circle",
      fill: fill,
      stroke: none,
      radius: size,
      padding: 2pt,
      name: name,
    )
  }

  // Helper function to draw input labels
  let input-label(pos, text-content) = {
    content(
      (pos.at(0) - 1.8, pos.at(1)),
      text(size: 7pt)[#text-content],
      anchor: "east"
    )
  }

  // Helper function to draw batch norm layer
  let batch-norm(pos, name: none) = {
    rect(
      (pos.at(0) - 0.3, pos.at(1) - 2.5),
      (pos.at(0) + 0.3, pos.at(1) + 2.5),
      fill: psu-purple.lighten(60%),
      stroke: none,
      name: name,
    )
    content(
      (pos.at(0), pos.at(1) - 3),
      text(size: 7pt)[`BatchNorm`]
    )
  }

  // Helper function to calculate line angle and shift along it
  let line-shift(start, end, dist) = {
    let dx = end.at(0) - start.at(0)
    let dy = end.at(1) - start.at(1)
    let len = calc.sqrt(dx * dx + dy * dy)
    return (
      x: dist * dx / len,
      y: dist * dy / len,
    )
  }

  // Helper function to draw a weight label
  let weight-label(start, end, ii, jj, offset: 0) = {
    let mid-x = (start.at(0) + end.at(0)) / 2
    let mid-y = (start.at(1) + end.at(1)) / 2

    let shift = if offset != 0 {
      let s = line-shift(start, end, offset * 0.3)
      (s.x, s.y)
    } else { (0, 0) }

    content(
      (mid-x + shift.at(0), mid-y + shift.at(1)),
      text(size: 4pt)[#calc.round(0.35 * ii - jj * 0.15, digits: 2)],
      frame: "rect",
      fill: psu-white,
      stroke: none,
      padding: 1pt,
    )
  }

  // Input layer labels
  let input-labels = (
    `isBourdon`,
    `flueDepth`, 
    `frequency`,
    `cutUpHeight`,
    `diameterToe`,
    `acousticIntensity`
  )

  // Draw regular DNN
  group(
    name: "regular",
    {
      // Input layer (6 nodes)
      for ii in range(6) {
        let y-pos = (2.5 - ii) * node-sep + 1
        neuron(
          (0, y-pos),
          fill: psu-sand,
          label: $x_ii$,
          name: "i" + str(ii+ 1),
        )
        input-label((0, y-pos), input-labels.at(ii))
      }

      // Batch normalization
      batch-norm((layer-sep * 0.6, 1), name: "bn")

      // Hidden layer 1 (5 nodes shown)
      for ii in range(5) {
        let y-pos = (ii - 2) * node-sep + 1
        neuron(
          (layer-sep * 1.2, y-pos),
          fill: psu-orange.lighten(50%),
          name: "h1" + str(ii + 1),
        )
      }

      // Hidden layer 2 (5 nodes shown)
      for ii in range(5) {
        let y-pos = (ii - 2) * node-sep + 1
        neuron(
          (layer-sep * 2, y-pos),
          fill: psu-orange.lighten(50%),
          name: "h2" + str(ii + 1),
        )
      }

      // Hidden layer 3 (5 nodes shown)
      for ii in range(5) {
        let y-pos = (ii - 2) * node-sep + 1
        neuron(
          (layer-sep * 2.8, y-pos),
          fill: psu-orange.lighten(50%),
          name: "h3" + str(ii + 1),
        )
      }
      
      // Add tanh activation label
      content(
        (layer-sep * 2, -2.5),
        text(size: 7pt)[$sans("tanh")(bold(W)^ell bold(x)^(ell-1) + bold(b)^ell)$]
      )

      // Output layer - Ising number
      neuron(
        (layer-sep * 3.6, (node-sep + 1) * 1.5),
        fill: psu-light-blue,
        label: $sans("DNN")_sans("I")$,
        name: "o_ising",
      )
      content(
        (layer-sep * 3.6 + 0.8, (node-sep + 1) * 1.5),
        text(size: 7pt)[`isingNumber`],
        anchor: "west"
      )
      content(
        (layer-sep * 3.6, (node-sep + 1) * 1.5 - 1.0),
        text(size: 7.5pt)[$sans("softplus")(bold(W)_I^L bold(x)^(L-1) + bold(b)_I^L)$]
      )

      // Output layer - Partials (4 nodes shown)
      // for ii in range(3) {
      //   //let y-pos = (ii - 1.5) * node-sep * 0.7
      //   let y-pos = (ii - 1.5) * node-sep
      //   neuron(
      //     (layer-sep * 3.6, y-pos),
      //     fill: psu-light-blue,
      //     label: $sans("DNN")_sans("P")_ii$,
      //     name: "o_p" + str(ii + 1),
      //   )
      // }

      neuron(
        (layer-sep * 3.6, 0.5 * node-sep),
        fill: psu-light-blue,
        label: $sans("DNN")_sans("P")_0$,
        name: "o_p1",
      )

      neuron(
        (layer-sep * 3.6, -0.5 * node-sep),
        fill: psu-light-blue,
        label: $dots.v$,
        name: "o_p2",
        size: 8.0,
      )

      neuron(
        (layer-sep * 3.6, -1.5 * node-sep),
        fill: psu-light-blue,
        label: $sans("DNN")_sans("P")_7$,
        name: "o_p3",
      )
      content(
        (layer-sep * 3.6 + 0.8, -0.5 * node-sep),
        text(size: 7pt)[`partial1, ..., partial8`],
        anchor: "west"
      )
      content(
        (layer-sep * 3.6, -0.5 * node-sep - 2.0),
        text(size: 7.5pt)[$sans("softmax")(bold(W)_P^L bold(x)^(L-1) + bold(b)_P^L)$]
      )

      // Connections with weights (simplified)
      // Input to batch norm
      for ii in range(6) {
        line("i" + str(ii + 1), "bn", ..arrow-style, stroke: psu-stone + 0.5pt)
      }

      // Batch norm to hidden layer 1
      for jj in range(5) {
        line("bn", "h1" + str(jj + 1), ..arrow-style, stroke: psu-stone + 0.5pt)
      }

      // Between hidden layers (sample connections with weights)
      for ii in range(5) {
        for jj in range(5) {
          line("h1" + str(ii + 1), "h2" + str(jj + 1), ..arrow-style, stroke: psu-stone + 0.3pt)
          line("h2" + str(ii + 1), "h3" + str(jj + 1), ..arrow-style, stroke: psu-stone + 0.3pt)
        }
      }

      // Hidden to output connections
      for ii in range(5) {
        line("h3" + str(ii + 1), "o_ising", ..arrow-style, stroke: psu-stone + 0.3pt)
        for jj in range(3) {
          line("h3" + str(ii + 1), "o_p" + str(jj + 1), ..arrow-style, stroke: psu-stone + 0.3pt)
        }
      }

      // Layer labels
      content((0, 4), text(weight: "bold", size: 8pt)[Input])
      content((layer-sep * 0.6, 4), text(weight: "bold", size: 8pt)[Norm])
      content((layer-sep * 2, 4), text(weight: "bold", size: 8pt)[Hidden Layers])
      content((layer-sep * 3.6, 4), text(weight: "bold", size: 8pt)[Output])
    },
  )
})
