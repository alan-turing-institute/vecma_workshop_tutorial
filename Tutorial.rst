VECMA Workshop Tutorial
=======================

Overview
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
(?). Our simulation is a 2D plane strain model, so the stress tensor has
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

Now fit a Gaussian Process to the input\_points and results to fit the
approximate model. We use the maximum marginal likelihood method to
estimate the GP hyperparameters

::

    gp = mogp_emulator.GaussianProcess(input_points, results)
    gp.learn_hyperparameters()

We can now make predictions for a large number of input points much more
quickly than running the simulation. For instance, let's sample 1000
points query\_points = ed.sample(1000) predictions =
gp.predict(query\_points)

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

We can see which points have not been ruled out yet (NROY) based on the
implausibility threshold.

::

    print("Actual point:", known_input[0])
    print("NROY:")
    print(query_points[hm.get_NROY()])

