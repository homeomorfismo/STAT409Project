// psu-beamer.typ
// Beamer-style theme for Portland State University in Typst

#import "psu-colors.typ": *

// Theme configuration
#let logo-size = 1cm
#let default-font = "Acumin Pro"

// Main theme function
#let psu-theme(
  title: none,
  subtitle: none,
  author: none,
  institute: none,
  date: none,
  left-logo: none,
  center-logo: none,
  right-logo: none,
  title-color: psu-electric-green,
  frame-title-color: psu-forest-green,
  body
) = {
  
  // Set document properties (only pass datetime objects to date)
  set document(
    title: title, 
    author: author,
  )
  
  // Set page layout
  set page(
    paper: "presentation-16-9",
    margin: (x: 2cm, y: 1.5cm),
    footer: [
      #line(length: 100%, stroke: 0.5pt + psu-forest-green)
      #v(-0.3em)
      #rect(
        fill: psu-forest-green,
        width: 100%,
        height: 1.5em,
        radius: 0pt
      )[
        #set text(fill: psu-white, size: 8pt)
        #grid(
          columns: (1fr, 1fr, 1fr, auto),
          align: (left, center, center, right),
          gutter: 1em,
          if title != none { title } else { "" },
          if institute != none { institute } else { "" },
          if author != none { author } else { "" },
          context [#counter(page).display() / #counter(page).final().first()]
        )
      ]
    ]
  )
  
  // Set text properties
  set text(
    font: default-font,
    size: 12pt,
    fill: psu-black
  )
  
  // Set heading styles
  show heading.where(level: 1): it => {
    set text(size: 16pt, weight: "bold", fill: frame-title-color)
    set align(center)
    v(0.5em)
    it.body
    v(0.25em)
    line(length: 100%, stroke: 1pt + frame-title-color)
    v(0.25em)
  }
  
  show heading.where(level: 2): it => {
    set text(size: 14pt, weight: "bold", fill: frame-title-color)
    v(0.3em)
    it.body
    v(0.2em)
  }

  show heading.where(level: 3): it => {
    set text(size: 12pt, weight: "bold", fill: frame-title-color)
    v(0.2em)
    it.body
    v(0.1em)
  }
  
  // Set list styling
  set list(marker: [•], indent: 1em)
  show list: it => {
    set text(fill: psu-forest-green)
    it
  }
  
  // Set enum styling  
  set enum(indent: 1em)
  show enum: it => {
    set text(fill: psu-forest-green)
    it
  }
  
  body
}

// Title slide function
#let psu-title-slide(
  title: none,
  subtitle: none,
  author: none,
  institute: none,
  date: none,
  left-logo: none,
  center-logo: none,
  right-logo: none,
  title-color: psu-electric-green,
  bg-color: psu-forest-green
) = {
  set page(footer: none, fill: bg-color)
  
  v(1cm)
  
  // Title
  if title != none {
    text(size: 24pt, weight: "bold", fill: title-color, style: "italic")[#title]
    v(0.5em)
  }
  
  // Subtitle
  if subtitle != none {
    text(size: 18pt, weight: "bold", fill: title-color)[#subtitle]
    v(1em)
  }
  
  // Author
  if author != none {
    text(size: 16pt, fill: title-color)[#author]
    v(0.5em)
  }
  
  // Institute
  if institute != none {
    text(size: 12pt, fill: title-color)[#institute]
    v(0.5em)
  }
  
  // Date
  if date != none {
    text(size: 10pt, fill: title-color)[#date]
    v(1em)
  }
  
  // Logos
  grid(
    columns: (1fr, auto, 1fr),
    align: (left, center, right),
    gutter: 0.5cm,
    if left-logo != none { left-logo } else { [] },
    if center-logo != none { center-logo } else { [] },
    if right-logo != none { right-logo } else { [] }
  )
}

// Block functions (equivalent to LaTeX blocks)
#let psu-block(title: none, body, fill-color: psu-stone) = {
  rect(
    fill: fill-color.lighten(90%),
    stroke: 1pt + fill-color,
    radius: 3pt,
    width: 100%
  )[
    #if title != none {
      rect(
        fill: fill-color,
        width: 100%,
        radius: (top: 3pt)
      )[
        #set text(fill: psu-white, weight: "bold")
        #h(0.5em)#title#h(0.5em)
      ]
      v(-0.3em)
    }
    #pad(0.5em)[#body]
  ]
}

#let psu-alert-block(title: none, body) = {
  psu-block(title: title, body, fill-color: psu-orange)
}

// Table of contents styling
#let psu-outline() = {
  show outline.entry.where(level: 1): it => {
    set text(fill: psu-forest-green, weight: "bold")
    it
  }
  
  show outline.entry.where(level: 2): it => {
    set text(fill: psu-forest-green)
    it
  }
  
  outline(indent: auto)
}
