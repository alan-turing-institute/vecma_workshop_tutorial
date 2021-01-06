# Vecma Workshop Tutorial

Files for the VECMA/`mogp_emulator` tutorial.

## Overview

This repository contains documents describing a tutorial written for
the "Reliability and reproducibility in computational science: Implementing
verification, validation and uncertainty quantification in silico" workshop
held at the Alan Turing Institute on 24 January, 2020. It builds on
two software libraries written as part of two projects developing
tools for performing Uncertainty Quantification for computational science
tools: [Verified Exascale Computing for Multiscale Applications](https://www.vecma.eu) (VECMA),
funded by the European Union’s Horizon 2020 research and innovation programme
under grant agreement No 800925, and two
[Uncertainty Quantification](https://www.turing.ac.uk/research/research-projects/uncertainty-quantification-multi-scale-and-multi-physics-computer-models)
[projects](https://www.turing.ac.uk/research/research-projects/uncertainty-quantification-black-box-models)
funded by EPSRC grant EP/N510129/1HA to the Alan Turing Institute.

## Installation

The software and simulations covered by the tutorial can be accessed
most easily using the Dockerfile provided in this repository. Please follow
the instructions provided in the [docker_installation.md](docker_installation.md)
to access and run the docker image.

## Software Used in this Tutorial

* [mogp_emulator](https://github.com/alan-turing-institute/mogp-emulator),
  a software library for performing surrogate model based Uncertainty
  Quantification on complex computer simulations
* [FabSim3](https://github.com/djgroen/FabSim3), a simulation management
  tool to support reproducible computational workflows
* [fabmogp](https://github.com/alan-turing-institute/fabmogp), a FabSim3
  plugin to perform the simulations described in the tutorial
* [fdfault](https://github.com/edaub/fdfault), a high performance
  finite difference code for performing dynamic earthquake rupture simulations.

