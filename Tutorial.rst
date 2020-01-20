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
three components: - One in the nominal fault plane - One out of plane -
One shear component

Because the fault is not a geometric plane, all three components combine
to determine the shear and normal components acting on the fault at a
given location.

Instead of running simulation points that we would like to query
directly, We will take a different approach. We will first run an
ensemble of simulations that are distributed throughout the space, fit
an approximate model to the ensemble of simulations, and then query the
approximate model. This will allow us to efficiently query many, many
points and ensure that we have robustly considered the entire input
space in our analysis.

We parametrize the stress tensor as follows:

   1. In-Plane Normal Stress (Assumed To Be Between -120 And -80 Mpa)
   2. Ratio Of Shear Stress To In-Plane Normal Stress (Assumed To Be
      Between 0.1 And 0.4)
   3. Ratio Of Out-Of-Plane Normal Stress To In-Plane Normal Stress
      (Assumed To Be Between 0.9 And 1.1)

This parameterization formluates the problem as three independent
components, so we can draw samples for these components independently.
we use the latinhypercubedesign class to create the design, which
attempts to spread out the samples evenly across the entire space. the
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
to create_problem to write the input files, call run_simulation to
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
           result = com
