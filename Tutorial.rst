VECMA Workshop Tutorial
=======================

In this tutorial we perform `Uncertainty Quantification (UQ) <https://en.wikipedia.org/wiki/Uncertainty_quantification>`_ [Hoekstra2019]_ on an earthquake model, **train and apply a surrogate model**. To generate initial data for the surrogate model, we perform an ensemble of many simulation runs, each with different input parameters. To generate and draw the samples we use the `Latin Hypercube technique <https://en.wikipedia.org/wiki/Latin_hypercube_sampling>`_ [Tang1993]_, while we rely on the `FabSim3 <https://fabsim3.readthedocs.io>`_ tool in the VECMA toolkit [Groen2019]_ to automatically run ensembles and curate both the simulation inputs and outputs. In the final stages of the tutorial we build and apply the surrogate model, and attempt to identify points that have been Not Ruled Out Yet **(Eric, can you explain more clearly what this means?)**  

In terms of components used, our Tube Map looks as follows:

.. figure:: FabMogpMap.png

We will use both `FabSim3 <https://fabsim3.readthedocs.io>`_ from the VECMA toolkit and the `Mogp emulator <https://github.com/alan-turing-institute/mogp_emulator>`_ from the Turing institute. Although we focus mainly on the Mogp emulator to do sampling in this tutorial, we will reflect on how the same workflow could be established using an alternative tool, namely the `EasyVVUQ component <http://easyvvuq.readthedocs.io>`_ in the VECMA toolkit.

In addition, we will perform tasks on only on your local host due to time constraints of this session, but we will provide clear instructions on how you can scale up various aspects of this approach, and use FabSim3 to run the same ensembles on remote machines such as supercomputers.

Setting up the environment and FabSim3
~~~~~~~~

To make life easier, we provide a Docker image which contains an installation of FabSim3, as well as the Earthquake simulation code and the Mogp toolkit. Our tutorial relies on a specific FabSim3 plugin that provides customisations for this application. The plugin is called FabMogp, and you can find it at: https://github.com/edaub/fabmogp
To set up Docker, please refer to the documentation provided `here <https://www.docker.com/get-started>`_

To download the Docker image, you can use:
::

    docker pull ha3546/vecma_turing_workshop

then, login to the image by typing:
::

    docker run --rm -ti ha3546/vecma_turing_workshop


Setting up the model
~~~~~~~~

This demo runs an earthquake simulation that depends on an input stress
tensor and outputs an earthquake size (related to the seismic moment,
which is something that can be measured). The stress tensor cannot be
measured in the earth, thus we need some kind of model to infer what
values of the stress tensor might be compatible with a given earthquake
size. The earthquake model is (in most cases) very expensive to run,
though for this example we have simplified things to enable the demo to
run relatively quickly, yet still maintain the complex nature of the
problem.

We have a broad understanding of the stress tensor values that might be
**(Eric: Indicate a range ?)**. Our simulation is a 2D plane strain model, so the stress tensor has
three components: 

- One in the nominal fault plane 
- One out of plane 
- One shear component

Because the fault is not a geometric plane, all three components combine
to determine the shear and normal components acting on the fault at a
given location.

Instead of running simulation points that we would like to query
directly, we will take a different approach. We will first run an
ensemble of simulations that are distributed throughout the space, fit
an approximate (surrogate) model to the ensemble of simulations, and then query the
surrogate model. This will allow us to efficiently query a large number of
points and ensure that we have robustly considered the entire input
space in our analysis.

We parametrize the stress tensor as follows:

   1. In-plane normal stress (assumed to be between -120 and -80 Mpa)
   2. Ratio of shear stress to in-plane normal stress (assumed to be
      between 0.1 and 0.4)
   3. Ratio of out-of-plane Normal Stress to in-plane normal stress
      (assumed to be between 0.9 and 1.1)


Creating samples
~~~~~~~~~~~~~~~~

This parameterization formulates the problem as three independent
components, so we can draw samples for these components independently.
we use the Latin HyperCube approach (i.e. the latinhypercubedesign class) to create the design, which
attempts to spread out the samples evenly across the entire space. The
implementation takes as input a list of tuples of lower/upper bounds on
the parameters:

::

       ed = mogp_emulator.LatinHypercubeDesign([(-120., -80.), (0.1, 0.4), (0.9, 1.1)])

We can now generate a design of 20 sample points by calling the sample
method:

::

       np.random.seed(157374)
       input_points = ed.sample(20)
       
Executing the simulations locally
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now we can actually run the simulations. First, we feed the input points
to `create_problem` to write the input files, call `run_simulation` to
actually simulate them, and compute_moment to load the data and compute
the earthquake size. The simulation is parallelized, so if you have
multiple cores available you can specify more processors to run the
simulation. Each simulation takes about 20 seconds on 4 processors on my
MacBook Pro, so the entire design will take several minutes to run.

::

   results = []
   counter = 1

   for point in input_points:
       name="simulation_{}".format(counter)
       try:
           result = compute_moment(name=name)
        except ModuleNotFoundError:
            create_problem(point, name=name)
            run_simulation(name=name, n_proc=4)
            result = compute_moment(name=name)
        results.append(result)
        counter += 1

    results = np.array(results)
    
Within FabSim you can also do this on the command line using:
::

    fabsim localhost mogp_ensemble:demo,sample_points=20
    

The advantage of using this approach is that the runs are each performed in individual directories, with input, output and environment curated accordingly. This makes it very easy to reproduce individual runs, and also helps with the diagnostics in case some of the simulations exhibit unexpected behaviors.
    

Executing the simulations on a remote resource
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Although this tutorial session is too short to set up and showcase the application on a remote resource, we do want to explain how you can do this for your machine of choice.

Essentially, you need to do three steps:
1. Create a machine definition for your resource of interest in FabSim3 (if there is not one already). How to do this is explained here: https://github.com/djgroen/FabSim3/blob/master/doc/CreateMachineDefinitions.md
2. Adding your user_specific information (such as account name and home directory) to `machines_user.yml`.
3. Replace the 'localhost' part of your FabSim ensemble command with the name of your machine. For example, if your machine is "archer", then you could change `fabsim localhost mogp_ensemble:demo,sample_points=20` into `fabsim archer mogp_ensemble:demo,sample_points=20`.

Creating a surrogate model
~~~~~~~~~~~~~~~~~~~~~~~~~~
Now fit a Gaussian Process to the input\_points and results to fit the
approximate model. We use the maximum marginal likelihood method to
estimate the GP hyperparameters

::

    gp = mogp_emulator.GaussianProcess(input_points, results)
    gp.learn_hyperparameters()

We can now make predictions for a large number of input points much more
quickly than running the simulation. For instance, let's sample 1000
points

::

    query\_points = ed.sample(1000) 
    predictions = gp.predict(query\_points)

Predictions contains both the mean values and variances from the
approximate model, so we can use this to quantify uncertainty given a
known value of the moment. Since we don't have an actual observation to
use, we will do a synthetic test by running an additional point so we
can evaluate the results from the known inputs.

::

    known_input = ed.sample(1)
    name="known_value"
    create_problem(known_input[0], name=name)
    run_simulation(name=name, n_proc=4)
    known_value = compute_moment(name=name)

One easy method for comparing a model with observations is known as
History Matching, where you compute an implausibility measure for many
sample points given all sources of uncertainty (observational error,
approximate model uncertainty, and "model discrepancy" which is a
measure of how good the model is at describing reality).

For simplicity here we will only consider the approximate model
uncertainty, but for real situations it is important to include all
three sources. The implausibility is then just the number of standard
deviations between the predicted value and the known value. To compute
the implausibility, we use the HistoryMatching class, which requires the
observation, query points (coords), and predicted values (expectations),
plus a threshold above which we can rule out a point

::

    hm = mogp_emulator.HistoryMatching(obs=known_value, coords=query_points, expectations=predictions, threshold=2.)

    implaus = hm.get_implausibility()

We can see which points have not been ruled out yet (NROY) based on the implausibility threshold.
**(Can we provide a literature reference to what this means?)** 

::

    print("Actual point:", known_input[0])
    print("NROY:")
    print(query_points[hm.get_NROY()])

Running the whole thing automated from the command line:
~~~~~~~~~~~~~~


You can run the full simulation workflow by using:
::

    fabsim localhost mogp_ensemble:demo,sample_points=20
    fabsim localhost fetch_results
    fabsim localhost mogp_analysis:demo,demo_localhost_16


References
##########
.. [Hoekstra2019] Hoekstra, Alfons G., Simon Portegies Zwart, and Peter V. Coveney. "Multiscale modelling, simulation and computing: from the desktop to the exascale." (2019): 20180355.
.. [Tang1993] Tang, Boxin. "Orthogonal array-based Latin hypercubes." Journal of the American statistical association 88.424 (1993): 1392-1397.
.. [Groen2019] Groen, Derek, et al. "Introducing VECMAtk-Verification, Validation and Uncertainty Quantification for Multiscale and HPC Simulations." International Conference on Computational Science. Springer, Cham, 2019.
