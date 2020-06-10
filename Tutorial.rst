VECMA Workshop Tutorial
=======================

In this tutorial we perform `Uncertainty Quantification (UQ) <https://en.wikipedia.org/wiki/Uncertainty_quantification>`_ [Hoekstra2019]_ on an earthquake model, by
**training and applying a surrogate model**. To generate initial data for the surrogate model,
we perform an ensemble of many simulation runs of the earthquake, each with different input parameters.
To generate and draw the samples we use the
`Latin Hypercube technique <https://en.wikipedia.org/wiki/Latin_hypercube_sampling>`_ [Tang1993]_,
while we rely on the `FabSim3 <https://fabsim3.readthedocs.io>`_ tool in the VECMA toolkit to
automatically run ensembles and curate both the simulation inputs and outputs. In the final stages
of the tutorial we use the `mogp_emulator <https://mogp_emulator.readthedocs.io>`_ package to build a
Gaussian Process surrogate model, and use the surrogate model to examine the parameter space and determine
plausible inputs to the computational earthquake model.

In terms of components used, our Tube Map looks as follows:

.. figure:: FabMogpMap.png

We will use both `FabSim3 <https://fabsim3.readthedocs.io>`_ from the VECMA toolkit and the
`Mogp emulator <https://github.com/alan-turing-institute/mogp_emulator>`_ from the Turing institute.
Although we focus mainly on the Mogp emulator to do sampling in this tutorial, we will reflect on how the
same workflow could be established using an alternative tool, namely the
`EasyVVUQ component <http://easyvvuq.readthedocs.io>`_ in the VECMA toolkit.

   **Sidebar: About FabSim3**

   FabSim3 is an toolkit for user-developers to help automate computational workflows involving many simulations
   and remote resources. It has been used in a variety of disciplines, for instance to facilitate coupled atomistic /
   coarse-grained materials simulations and to perform large-scale sensitivity analysis of agent-based migration models. The
   tool is open-source (BSD 3-clause license) and one of the main components of the
   `VECMA toolkit <http://www.vecma-toolkit.eu>`_.

   **Sidebar: About mogp_emulator**

   mogp_emulator is a python package for performing uncertainty quantitification (UQ) workflows for complex computer
   simulations. The core component of the package is an efficient Gaussian Process emulator for fitting
   surrogate models, which will also include GPU and FPGA support for situations where high performance is required.
   It also contains tools for generating experimental designs and performing model calibration
   which are highlighted in the tutorial. The package is open source (MIT license) and under current
   development by the Turing Research Engineering Group as part of two projects on UQ.

In addition, we will perform tasks on only on your local host due to time constraints of this session, but we will provide clear instructions on how you can scale up various aspects of this approach, and use FabSim3 to run the same ensembles on remote machines such as supercomputers.

Setting up the environment and FabSim3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To make life easier, we provide a Docker image which contains an installation of FabSim3, as
well as the Earthquake simulation code ``fdfault`` and the ``mogp_emulator`` toolkit. Our tutorial
relies on a specific FabSim3 plugin that provides customisations for this application. The plugin
is called FabMogp, and you can find it at: https://github.com/edaub/fabmogp
To set up Docker, please refer to the documentation provided `here <https://www.docker.com/get-started>`_

   **Sidebar: code blocks in this tutorial**

   Throughout the tutorial, we will highlight how to run the computational
   workflow from the bash shell. These commands will be highlighted in code boxes with no preamble code comments.
   Additionally, we will illustrate the underlying mogp_emulator Python code in code blocks. These will include a
   comment preamble ``# mogp_emulator code`` to emphasize that this is python code that can be run in the Python
   interpreter. However, note that some of the code blocks will depend on previously defined variables or running some
   of the FabSim commands first, so you may get errors if you are not careful. If you have a question about
   running any of the code in the tutorial, please ask one of the facilitators.

To download the Docker image, you can use:

.. code:: bash

   docker pull ha3546/vecma_turing_workshop

then, create an empty folder on your PC, and login to the image by typing:

.. code:: bash

   docker run --rm -v <PATH_to_your_folder>:/home/root/turing_workshop/FabSim3/results -ti ha3546/vecma_turing_workshop

Within the container, you can start a Python interpreter using ``python3`` to
run some of the mogp_emulator routines. Alternatively we describe how to automate the entire workflow
from the shell using FabSim.

Setting up the model
~~~~~~~~~~~~~~~~~~~~

In Uncertainty Quantification (UQ) workflows, we would like to learn about a complex simulator that
describes the real world (imperfectly). These simulations are usually computationally
intensive and the outputs are very sensitive to the inputs, making it hard to use them directly to
compare with observations.

As a concrete example of one of these problems, we will examine a simplified version of an earthquake
simulation. In seismology, the most basic quantity that we can measure about an earthquake is its
size, quantified by the seismic moment. The seismic moment is proportional to the relative
displacement across the two sides of the fault (known as the slip) multiplied by the area of the
fault plane that experienced this slip. Larger earthquakes occur when either more slip occurs or
the area that slipped increases (in nature, these two quantities are correlated so earthquakes
get bigger by both increasing the slip and the area simulataneously).

   **Sidebar: fdfault**

   To run the earthquake simulations, we are using the fdfault application. fdfault is a high
   performance, parallelized finite difference code for simulation of frictional failure and
   wave propagation in elastic-plastic media. It features high order finite difference methods
   and is able to handle complex geometries through coordinate transformations and implements
   a provably stable method.

Physically, this slip occurs when the stress (or force) on the fault exceeds the fault strength.
Fault strength is determined by a friction law that compares the shear force on a patch of the
fault to the normal force acting on that patch of the fault. When this condition is met, the fault
slips on this local patch, which changes the forces acting on the other fault patches (this process
is described by the elastic wave equation). Thus, to make a physical model of an earthquake, we need
to specify the initial forces on the fault, the strength of the fault, and the elastic medium
surrounding the fault. In general, the initial forces on the fault cannot be determined
in the earth, and we will use a UQ workflow to try and estimate these quantities. A snapshot from
one of the simulations is shown in the figure below -- the bumpy line is the rough fault surface,
and the color scale shows the propagation of elastic waves away from the fault due to the slip on
the fault.

.. figure:: earthquake.png
   :width: 405 px
   :align: center

   Snapshot of an earthquake simulation. The bumpy line is the fault surface. The color
   scale represents the ground motions from the resulting earthquake as the elastic
   waves carry the stress changes from the slip propagate through the medium.

Complicating matters is the fact that earthquake faults are not smooth planes, but instead rough
bumpy surfaces with a fractal geometry. An important consequence of this is that the *smallest*
wavelength bumps have the largest effect on the resulting forces. This is what makes earthquake
problems challenging to model: at a given model resolution, you are omitting details that play an
important role. This small scale roughness that is left out of the model must instead be accounted
for when setting the strength of the fault. However, for this demonstration we will assume that
both the rough geometry of the fault and the fault strength are known in advance, and it is just the
initial stress (forces) that must be inferred. This tutorial will show how a UQ workflow can be
used to estimate the fault stresses for a given earthquake size.

The simulation requires us to specify the initial stress tensor acting on the earthquake fault in order
to run a simulation. For this case, we run a 2D plane strain simulation to reduce the
problem to a reasonable computational level such that it only takes a short amount of time to run.
In a plane strain model, the stress tensor has three components: two compressive and one shear.
One compressive component describes the normal force on the fault, and the other component describes
the normal force in the orthogonal direction. The shear component sets the shear force acting on
the fault. Note, however, that all three components matter because the fault is not a perfect plane,
and we must project the tensor into the local shear and normal components for a given patch on
the fault to determine the actual forces on the fault.

While we do not know the exact values of the stresses on earthquake faults, we do know a few general
things that we should incorporate into our simulations.

1. Pressure increases linearly with depth due to the weight of the rocks. This can be mediated by
   fluid pressure counterbalancing some of the overburden pressure, and earthquakes start at different
   depths, so we are not sure of the exact value. However, at typical depths where earthquakes start
   (5-10 km), this pressure is expected to be somewhere in the range of -80 MPa to -120 MPa (stress
   is assumed to be negative in compression). Therefore, we can use this range to choose values for one
   component, and then assume that the other component is similar (say +/- 10% of that value).

2. Shear stresses are below the failure level on the fault. This can be understood as simply reflecting
   that earthquakes tend to start in one place and then grow from there, and do not start in many
   places at once. Thus, we will assume that since the frictional strength of the fault in our
   simulation is 0.7 times the normal stress, the initial shear stress is between 0.1 and 0.4 of
   the normal stress.

Thus, we parametrize the simulations with three inputs: a normal stress that is uniformly distributed
from -120 MPa to -80 MPa, a shear to normal ratio uniformly distributed from 0.1 to 0.4, and a
ratio between the two normal stress components uniformly distribted from 0.9 to 1.1. These three
parameters can be sampled via Monte Carlo sampling and then transformed to the three correlated stress
components in order to run the simulation.


Creating samples
~~~~~~~~~~~~~~~~

While we can simply draw Monte Carlo samples for our simulation runs, we probably should be a bit
more careful about this since we only get a limited number of runs. It is probably a good idea that
some of our simulations sample low values of the inputs, some high values, and try and do a decent job
of mixing up the different values. This can be done by using a Latin Hypercube, which ensures that
samples are drawn from each quantile of the distribution of each parameter that is varied. The
``mogp_emulator`` package has a built-in class for generating these types of samples:

.. code:: python

   # mogp_emulator code

   import numpy as np
   import mogp_emulator

   ed = mogp_emulator.LatinHypercubeDesign([(-120., -80.), (0.1, 0.4), (0.9, 1.1)])

   seed = None
   sample_points = 20

   np.random.seed(seed)
   input_points = ed.sample(sample_points)

The input arguments to ``LatinHypercubeDesign`` can take several forms, but the simplest is if you
want your parameters to be uniformly distributed. In that case, you simply pass a list of tuples,
where each tuple gives the min/max value that each parameter should take. To create a design,
we simply use the ``sample`` method, which requires the number of points that should be included in
the design.

   **Sidebar: other sampling methods in mogp_emulator**

   mogp_emulator also implements Monte Carlo sampling and MICE (Mutual Information for Computer Experiments).
   MICE is a sequential design algorithm that chooses simulation points one at a time (or in batches) based
   on fitting a Gaussian Process to the intermediate results at each step. Usually, this additional overhead
   is small compared to the simulation time required for a complex computer model, so this gives an improvement
   in performance.

``input_points`` is a numpy array with shape ``(20, 3)`` as we
have 20 design points, each containing 3 parameters. We can iterate over this to get each successive
point where we need to run the simulation.

   **Sidebar: EasyVVUQ, an alternative tool for scalable sampling**

   In this tutorial we use Mogp for sampling, primarily because we train a surrogate model that relies on its Gaussian
   process emulation functionalities. For other applications, it's also possible to use EasyVVUQ for sampling and
   uncertainty quantification. Both tools complement each other, in that Mogp provides Gaussian process emulators, whereas
   EasyVVUQ has a stronger emphasis on providing sophisticated and scalable sampling and results collation (for instance for
   use with thousands or millions of jobs on a remote supercomputer). EasyVVUQ is part of the
   `VECMA toolkit <http://www.vecma-toolkit.eu>`_, has a documentation site `here <https://easyvvuq.readthedocs.io>`_, and a
   simple separate tutorial `here <https://colab.research.google.com/drive/1qD07_Ry2lOB9-Is6Z2mQG0vVWskNBHjr>`_.

Executing the simulations locally
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now we can actually run the simulations. First, we feed the input points
to `create_problem` to write the input files, call `run_simulation` to
actually simulate them, and compute_moment to load the data and compute
the earthquake size. The simulation is parallelized, so if you have
multiple cores available you can specify more processors to run the
simulation. Each simulation takes about 20 seconds on 4 processors on my
MacBook Pro, so the entire design will take several minutes to run.

.. code:: python

   # mogp_emulator code

   from earthquake import create_problem, run_simulation

   results = []
   counter = 1

   for point in input_points:
       name="simulation_{}".format(counter)
       create_problem(point, name=name)
       run_simulation(name=name, n_proc=4)
       result = compute_moment(name=name)
       results.append(result)
       counter += 1

   results = np.array(results)

While this procedure might be okay for this demo, in real situations these runs would be much more
expensive and need to be run on a supercomputer. Runs on a supercomputer will be much harder to
manage in this fashion, as jobs will need to be created and submitted separately according to the
submission details of the particular supercomputer, and then we would need to have a way to collect
all of the results to run the analysis below. This will be hard to manage even for a modest number of
simulations. Thus, we have automated this process using FabSim3 to show a better method for handling
ensembles of simulations in a UQ workflow.

Within FabSim you can do this on the command line using:

.. code:: bash

   fab localhost mogp_ensemble:demo,sample_points=20

You can set the random seed for the Latin Hypercube sampling by passing ``seed=<seed>`` along with the
number of sample points (separate any arguments with a comma). The ``mogp_ensemble`` workflow will
automatically sample the Latin Hypercube to create the desired number of points, set up all of the
necessary simulations, and run them. The advantage of using this approach over the manual approach
described above is that the runs are each performed in individual directories, with input, output and
environment curated accordingly. This makes it very easy to reproduce individual runs, and also helps
with the diagnostics in case some of the simulations exhibit unexpected behaviours.



Executing the simulations on a remote resource
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Although this tutorial session is too short to set up and showcase the application on a remote resource, we do want to explain how you can do this for your machine of choice.

Essentially, you need to do three steps:
1. Create a machine definition for your resource of interest in FabSim3 (if there is not one already). How to do this is explained here: https://github.com/djgroen/FabSim3/blob/master/doc/CreateMachineDefinitions.md
2. Adding your user_specific information (such as account name and home directory) to `machines_user.yml`.
3. Replace the 'localhost' part of your FabSim ensemble command with the name of your machine. For example, if your machine is "archer", then you could change `fabsim localhost mogp_ensemble:demo,sample_points=20` into `fabsim archer mogp_ensemble:demo,sample_points=20`.

    **Sidebar: where do I find a suitable larger resource?**

    Unfortunately the national `ARCHER supercomputer <http://www.archer.ac.uk/>`_ is about to be decommissioned, but
    there are a few alternatives available. Several UK universities have so-called *Tier-2* resources available, which
    can support runs using thousands of cores, and one can also choose to buy time on the Cloud. For larger needs, one
    will need to look abroad, for instance by writing a proposal for `PRACE (preparatory) access
    <http://www.prace-ri.eu/>`_ or contacting other foreign supercomputer centres. Of course this is not an ideal
    situation, so we as authors of this tutorial happily endorse any effort to try and establish more suitable
    large-scale resources here in the UK.

    **Sidebar: running large ensembles on large machines**

    Most clusters and supercomputers have queuing systems that enable users to run a small ensemble of 5-20 jobs.
    However, larger ensembles can be rejected by queuing systems due to policy constraints meant to prevent scheduler
    overload. To circumvent this issue, one can choose to use a Pilot Job framework. Within the VECMA toolkit we provide
    `QCG-PilotJob <https://github.com/vecma-project/QCG-PilotJob>`_, a component which supports all major machines using the
    SLURM scheduler.

Analysing the Results
~~~~~~~~~~~~~~~~~~~~~

Collecting the Results
----------------------

If the simulations were run within the Python interpreter we do not need to do anything to collect
the results; however if simulations were run using FabSim, then we need to fetch the results and
load them into the python interpreter. From the shell, to fetch the results we simply need to enter:

.. code:: bash
   fab localhost fetch_results

This will collate all of the results into a subdirectory of the ``results`` directory within the
FabSim installation (within the Docker container, this is likely to be ``demo_localhost_16``).
Once the results have been collected, to re-load the input points, results, and the
``LatinHypercubeDesign`` class that created them we have provided a convenience function
``load_results`` in the ``mogp_functions`` module:

.. code:: python

   # mogp_emulator code

   from mogp_functions import load_results

   results_dir = <path_to_results>/demo_localhost_16
   input_points, results, ed = load_results(results_dir)

You will need to set the appropriate directory where the results are collected. Fortunately, FabSim can
manage this for you when you run the analysis using the FabSim commands specified below.

Creating the surrogate model
----------------------------

Once we have run all of the input points, we can proceed with fitting the approximate model and analysing
the parameter space. We can fit a Gaussian Process to the results using the ``GaussianProcess`` class:

.. code:: python

   # mogp_emulator code

   gp = mogp_emulator.GaussianProcess(input_points, results)

This just creates the GP class. Gaussian Processes are a non-parametric model for regression that approximates
the complex simulator function as a multivariate normal distribution. In simple terms, a GP interpolates
between the known simulation points in a robust way and provides uncertainty estimates for any predictions
that it makes. Because it has an uncertainty estimate, it is commonly used in UQ workflows.

In order to make predictions, we need to fit the model to the data. The class has several methods of doing this,
but the simplest is to use the maximum marginal likelihood, which is easy to compute for a GP:

.. code:: python

   # mogp_emulator code

   gp.learn_hyperparameters()

This finds a set of correlations lengths, the hyperparameters of the GP, that maximises the marginal
log-likelihood and determines how the GP interpolates between unknown points. Once these parameters are
estimated, we can make predictions efficiently for unknown parameter values and have estimates of
the uncertainty.

   **Sidebar: other options in the Gaussian Process surrogate model**

   A Gaussian Process requires specification of a mean function and a covariance kernel in order to
   perform the necessary calculations. We have several built-in kernels (the popular squared exponential and
   Matern 5/2 kernels), though the user can easily define additional stationary kernels. The current tutorial
   uses a zero mean function, but an upcoming update to mogp_emulator will allow for flexible specification
   of mean functions.

   This tutorial fits the GP hyperparameters through maximum likelihood. We also have implemented weak prior
   MCMC sampling if a Bayesian specification of the emulator is desired. Future improvements will also allow for
   priors to be specified to enable MAP or full MCMC estimation of the hyperparameters.

Making Predictions
------------------

To analyse the full parameter space, we need to draw a large number of samples from the full space. As
before, we do this using our Latin Hypercube Design (which ensures that the points we choose are spread
out across the full parameter space), but since we do not need to run the computationally intensive
simulation for each one, we can draw many more samples (say, 10,000 in this case):

.. code:: python

   # mogp_emulator code

   analysis_points = 10000

   query_points = ed.sample(analysis_points)
   predictions = gp.predict(query_points)

The ``predictions`` holds the mean and variance of all 10,000 prediction points. We will need these
momentarily to analyse the input space.

History Matching
----------------

Once we have predictions for a large number of query points, it is straightforward to compare with
observations. History Matching is one way to perform this comparison -- in History Matching, we compute an
implausibility metric for each query point by determining the number of standard deviations
between the observation and the predicted mean from the approximate model. We can then "rule out"
points that are many standard deviations from the mean as being implausible given the observation
and all sources of error.

In real situations, there are three types of uncertainty that we need to account for when computing
implausibility:

1. Observational error, which is uncertainty in the observed value itself;
2. Uncertainty in the approximate model, which reflects the fact that we cannot query the full
   computational model at all points; and
3. Model discrepancy, which is uncertainty about the model itself, and measures how well the
   computational model represents reality.

In practice, 1. and 2. are straightforward to determine, while 3. is much trickier. However, many
studies have shown that not accounting for model discrepancy leads to `overconfident predictions
<https://doi.org/10.1111/1467-9868.00294>`_, so this is essential to consider to give a thorough
UQ treatment to a computational model. However, estimating model uncertainty is in itself a difficult
(and often subjective) task, and is beyond the scope of this tutorial, as it requires knowledge about
the approximations made in the simulation. Thus, we will restrict ourselves to only accounting for
uncertainty in the approximate model in this tutorial, but note that realistic UQ assessments
require careful scrutiny and awareness of the limitations of computational models.

   **Sidebar: other calibration techniques**

   An advantage of history matching is that it is conceptually simple and can still provide useful
   information even if the surrogate model is uncertain about parts of the parameter space. However,
   it has the disadvantage that it only tells you about what parts of the space can be ruled out,
   not what are better choices from within the space that has not been ruled out. For full Bayesian
   calibration, one needs to specify the priors for all model parameters and an emulator that
   can accurately estimate the simulator output at any point. Then one can use MCMC sampling or other
   Bayesian estimation techniques to determine the posterior distribution of the input parameters.
   Full Bayesian calibration technqiues are not currently implemented in mogp_emulator.

To compute the implausibility, we need to know the observation (which we will choose arbitrarily
here; reasonable values to consider range from 40 to 250) and the model predictions/uncertainties
(referred to as``expectations`` in the ``HistoryMatching`` class). These can be passed directly to
the ``HistoryMatching`` class when creating it (or prior to computing the implausibility):

.. code:: python

   # mogp_emulator code

   threshold = 3.
   known_value = 58.

   hm = mogp_emulator.HistoryMatching(obs=known_value, expectations=predictions,
                                      threshold=threshold)

   implaus = hm.get_implausibility()
   NROY = hm.get_NROY()

Once we have computed the implausibility, we can figure out which points can be ruled out
(known as NROY, Not Ruled Out Yet). We assume this threshold to be 3 standard deviations, though this could
be made larger if we would like to be more conservative. The ``NROY`` variable here is just a list of indices
that have not been ruled out yet from all of our sample points, we we can use the indexing capabilities of
numpy to get the NROY points. The NROY points provide us with one simple way to visualise
the results:

.. code:: python

   # mogp_emulator code

   import matplotlib.pyplot as plt

   plt.figure()
   plt.plot(query_points[NROY, 0], query_points[NROY, 1], 'o')
   plt.xlabel('Normal Stress (MPa)')
   plt.ylabel('Shear to Normal Stress Ratio')
   plt.xlim((-120., -80.))
   plt.ylim((0.1, 0.4))
   plt.title("NROY Points")
   plt.show()

.. figure:: nroy.png
   :width: 412px
   :align: center

   Points that have not been ruled out yet (NROY) projected into the normal and shear/normal
   plane of the parameter space. Note that the points are fairly tightly clustered along a line,
   showing that the earthquake size is very sensitive to the stress tensor components.

This shows the points that have not been ruled out projected onto a plane in 2 dimensions. You can try
other projections, though by far most of the predictive power in the model comes from knowing the
shear/normal stress and the normal stress (the moment is much less sensitive to the second normal
stress component). We can also make a pseudocolor plot showing the implausibility metric projected
into this plane:

.. code:: python

   # mogp_emulator code

   import matplotlib.tri

   plt.figure()
   tri = matplotlib.tri.Triangulation(-(query_points[:,0]-80.)/40., (query_points[:,1]-0.1)/0.3)
   plt.tripcolor(query_points[:,0], query_points[:,1], tri.triangles, implaus,
                 vmin = 0., vmax = 6., cmap="viridis_r")
   cb = plt.colorbar()
   cb.set_label("Implausibility")
   plt.xlabel('Normal Stress (MPa)')
   plt.ylabel('Shear to Normal Stress Ratio')
   plt.title("Implausibility Metric")
   plt.show()

.. figure:: implausibility.png
   :width: 400px
   :align: center

   Implausibility metric (number of standard deviations between the observation and the predictions
   of the surrogate model) in the parameter space projected into the normal and shear/normal plane.
   As with the NROY plot, this shows the sensitivity of the output to the stress components.

This illustrates that there is only a limited part of the parameter space that can produce a particular
seismic moment. This means that the sensitivity of the earthquake size to the stress is actually quite
a useful constraint, as there is only a small range of stress conditions that can produce an
earthquake of a particular size. However, note that many of the other things that were assumed to be
known here (friction, fault geometry, how the earthquake initiates) are in practice not well understood,
meaning that realistic applications of this sort will be much more uncertain once all of these other
aspects of the simulation are varied. However, this tutorial illustrates the essence of the UQ workflow
and how it can be used to constrain complex models with observations.

Automating the Analysis
-----------------------

We have provided two ways to run the above set of analysis commands and plotting commands. To
run the entire thing within the Python interpreter, import the ``run_mogp_analysis`` function
from the ``mogp_function`` module. This function requires 4 inputs:
``analysis_points``, ``known_value``, ``threshold``, and ``results_dir``
(all of these variables are defined above). This should run the analysis and create the plots.

Alternatively, we have set up a FabSim command to do this for you that accepts all of the
above options (default values are the ones provided above for everything except ``results_dir``,
which is likely to be ``demo_localhost_16`` for the docker container we have provided).
To run the analysis using FabSim, enter the following on the command line:

.. code:: bash

   fab localhost mogp_analysis:demo,demo_localhost_16

This will run the analysis and create the plots in the ``results`` directory within the FabSim
installation. You should be able to view these if you correctly mounted a shared directory between
your local machine and this directory in the container.

Running the whole thing automated from the command line:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


You can run the full simulation workflow by using:

.. code:: bash

   fab localhost mogp_ensemble:demo,sample_points=20
   fab localhost fetch_results
   fab localhost mogp_analysis:demo,demo_localhost_16

Further Investigation
~~~~~~~~~~~~~~~~~~~~~

Some things in the UQ workflow that you can vary to see how they effect the results:

* Change the number of sample points (note that you can only do this up to a limit given
  the number of simulations you have to run!)
* Change the random seed to draw a different set of samples for the Latin Hypercube samples
* Change the number of analysis points that are used in history matching
* Change the threshold for determining the NROY points
* Change the ``known_value`` of the seismic moment (try values from 40 to 250; outside of that
  range you are likely to rule out the entire space!)


References
~~~~~~~~~~
.. [Hoekstra2019] Hoekstra, Alfons G., Simon Portegies Zwart, and Peter V. Coveney. "Multiscale modelling, simulation and   computing: from the desktop to the exascale." (2019): 20180355.
.. [Tang1993] Tang, Boxin. "Orthogonal array-based Latin hypercubes." Journal of the American statistical association 88.424 (1993): 1392-1397.
.. [Groen2019] Groen, Derek, et al. "Introducing VECMAtk-Verification, Validation and Uncertainty Quantification for Multiscale and HPC Simulations." International Conference on Computational Science. Springer, Cham, 2019.
