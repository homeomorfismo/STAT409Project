// psu-thesis.typ
// Portland State University Thesis/Dissertation Document Template
// Copyright (c) 2025

#import "psu-colors.typ": *

// Thesis configuration state
#let thesis-state = state("thesis-config", (
  thesis-type: "thesis",
  degree: none,
  department: none,
  graduation-month: none,
  graduation-year: none,
  chair: none,
  member-one: none,
  member-two: none,
  member-three: none,
  member-four: none,
  document-format: "monograph",
  title: none,
  author: none,
))

// Main thesis template function
#let psu-thesis(
  title: none,
  author: none,
  thesis-type: "thesis", // "thesis" or "dissertation"
  degree: none,
  department: none,
  graduation-month: none,
  graduation-year: none,
  chair: none,
  member-one: none,
  member-two: none,
  member-three: none,
  member-four: none,
  document-format: "monograph", // "monograph" or "multi-paper"
  body
) = {
  
  // Update state
  thesis-state.update((
    thesis-type: thesis-type,
    degree: degree,
    department: department,
    graduation-month: graduation-month,
    graduation-year: graduation-year,
    chair: chair,
    member-one: member-one,
    member-two: member-two,
    member-three: member-three,
    member-four: member-four,
    document-format: document-format,
    title: title,
    author: author,
  ))
  
  // Set document metadata
  set document(
    title: title,
    author: author,
  )
  
  // Page setup (letter paper, 1-inch margins)
  set page(
    paper: "us-letter",
    margin: 1in,
    numbering: none, // Will be set per section
  )
  
  // Text setup
  set text(
    font: "Times New Roman",
    size: 12pt,
    fill: black
  )
  
  // Paragraph setup (double spacing equivalent)
  set par(
    leading: 1.5em,
    first-line-indent: 0.5in,
    justify: true
  )
  
  // Heading setup
  show heading.where(level: 1): it => {
    set text(size: 14pt, weight: "bold")
    set align(center)
    pagebreak(weak: true)
    v(1in)
    upper(it.body)
    v(0.5in)
  }
  
  show heading.where(level: 2): it => {
    set text(size: 12pt, weight: "bold")
    v(0.5in)
    it.body
    v(0.25in)
  }
  
  show heading.where(level: 3): it => {
    set text(size: 12pt, weight: "bold", style: "italic")
    v(0.25in)
    it.body
    v(0.125in)
  }
  
  // Figure and table captions
  show figure.caption: it => {
    set text(size: 10pt)
    set par(first-line-indent: 0pt)
    context {
      strong(it.supplement + " " + it.counter.display() + ". ") + it.body
    }
  }
  
  // Table setup
  show table: it => {
    set text(size: 10pt)
    it
  }
  
  body
}

// Title page function
#let psu-title-page(
  title: none,
  author: none,
  thesis-type: "thesis",
  degree: none,
  department: none,
  graduation-month: none,
  graduation-year: none,
  chair: none,
  member-one: none,
  member-two: none,
  member-three: none,
  member-four: none,
) = {
  set page(numbering: none)
  set align(center)
  
  v(1in)
  
  // Title
  text(size: 14pt, weight: "bold")[#title]
  
  v(2cm)
  
  [by]
  
  v(1cm)
  
  text(size: 12pt)[#author]
  
  v(1fr)
  
  // Thesis statement
  [A #thesis-type submitted in partial fulfillment of the\ 
   requirements for the degree of]
  
  v(1cm)
  
  text(size: 12pt, weight: "bold")[#degree]
  
  [ in ]
  
  text(size: 12pt, weight: "bold")[#department]
  
  v(2cm)
  
  // Committee
  upper[Dissertation Committee:]
  linebreak()
  [#chair, Chair]
  linebreak()
  [#member-one]
  linebreak()
  [#member-two]
  if member-three != none {
    linebreak()
    [#member-three]
  }
  if member-four != none {
    linebreak()
    [#member-four]
  }
  
  v(1fr)
  
  [Portland State University\ 
   #graduation-month #graduation-year]
  
  pagebreak()
}

// Copyright page
#let psu-copyright-page(author: none, graduation-year: none) = {
  set page(numbering: none)
  
  v(1fr)
  align(center)[© #graduation-year #author]
  v(1fr)
  
  pagebreak()
}

// Abstract environment
#let psu-abstract(body) = {
  set page(numbering: "i")
  counter(page).update(1)
  
  align(center)[
    #text(size: 14pt, weight: "bold")[Abstract]
  ]
  
  v(0.5in)
  
  body
  
  pagebreak()
}

// Table of contents
#let psu-table-of-contents() = {
  align(center)[
    #text(size: 14pt, weight: "bold")[Table of Contents]
  ]
  
  v(0.5in)
  
  show outline.entry.where(level: 1): it => {
    set text(weight: "bold")
    it
  }
  
  outline(
    title: none,
    indent: auto
  )
  
  pagebreak()
}

// List of figures
#let psu-list-of-figures() = {
  align(center)[
    #text(size: 14pt, weight: "bold")[List of Figures]
  ]
  
  v(0.5in)
  
  outline(
    title: none,
    target: figure.where(kind: image),
  )
  
  pagebreak()
}

// List of tables
#let psu-list-of-tables() = {
  align(center)[
    #text(size: 14pt, weight: "bold")[List of Tables]
  ]
  
  v(0.5in)
  
  outline(
    title: none,
    target: figure.where(kind: table),
  )
  
  pagebreak()
}

// Dedication page
#let psu-dedication(body) = {
  align(center)[
    #text(size: 14pt, weight: "bold")[Dedication]
  ]
  
  v(0.5in)
  
  align(center)[
    #text(style: "italic")[#body]
  ]
  
  pagebreak()
}

// Acknowledgments page
#let psu-acknowledgments(body) = {
  align(center)[
    #text(size: 14pt, weight: "bold")[Acknowledgments]
  ]
  
  v(0.5in)
  
  body
  
  pagebreak()
}

// Preface page
#let psu-preface(body) = {
  align(center)[
    #text(size: 14pt, weight: "bold")[Preface]
  ]
  
  v(0.5in)
  
  body
  
  pagebreak()
}

// Glossary page
#let psu-glossary(body) = {
  align(center)[
    #text(size: 14pt, weight: "bold")[Glossary]
  ]
  
  v(0.5in)
  
  body
  
  pagebreak()
}

// Start main body (switch to Arabic numbering)
#let psu-start-body() = {
  set page(numbering: "1")
  counter(page).update(1)
}

// Chapter heading page for multi-paper format
#let psu-chapter-heading-page(title: none, subtitle: none) = {
  pagebreak()
  
  align(center)[
    #text(size: 14pt, weight: "bold")[#title]
  ]
  
  if subtitle != none {
    v(0.5in)
    align(center)[
      #text(style: "italic")[#subtitle]
    ]
  }
  
  v(0.5in)
}

// References section
#let psu-references(title: "References", body) = {
  align(center)[
    #text(size: 14pt, weight: "bold")[#title]
  ]
  
  v(0.5in)
  
  // Reset paragraph formatting for bibliography
  show par: set par(first-line-indent: 0pt, hanging-indent: 0.5in)
  
  body
}

// Appendix function
#let psu-appendix(title, body) = {
  pagebreak()
  
  align(center)[
    #text(size: 14pt, weight: "bold")[Appendix #title]
  ]
  
  v(0.5in)
  
  body
}
