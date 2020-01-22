import numpy as np
import matplotlib.pyplot as plt
import mogp_emulator
from earthquake import create_problem, run_simulation, compute_moment
import matplotlib.pyplot as plt
import matplotlib.tri

# This demo runs an earthquake simulation that depends on an input stress tensor and outputs
# an earthquake size (related to the seismic moment, which is something that can be measured).
# The stress tensor cannot be measured in the earth, thus we need some kind of model to infer
# what values of the stress tensor might be compatible with a given earthquake size. The
# earthquake model is (in most cases) very expensive to run, though for this example we have
# simplified things to enable the demo to run relatively quickly, yet still maintain the complex
# nature of the problem.

# We have a broad understanding of the stress tensor values that might be. Our simulation is
# a 2D plane strain model, so the stress tensor has three components: two compressive components
# (one in the nominal fault plane, one out of plane), and one shear component. Because the fault
# is not a geometric plane, all three components combine to determine the shear and normal
# components acting on the fault at a given location.

# Instead of running simulation points that we would like to query directly, we will take a
# different approach. We will first run an ensemble of simulations that are distributed throughout
# the space, fit an approximate model to the ensemble of simulations, and then query the approximate
# model. This will allow us to efficiently query many, many points and ensure that we have robustly
# considered the entire input space in our analysis.

# We parametrize the stress tensor as follows:

# 1. In-plane normal stress (assumed to be between -120 and -80 MPa)
# 2. Ratio of shear stress to in-plane normal stress (assumed to be between 0.1 and 0.4)
# 3. Ratio of out-of-plane normal stress to in-plane normal stress (assumed to be between 0.9 and 1.1)

# This parameterization formluates the problem as three independent components, so we can draw samples
# for these components independently. We use the LatinHypercubeDesign class to create the design, which
# attempts to spread out the samples evenly across the entire space. The implementation takes as input
# a list of tuples of lower/upper bounds on the parameters:

ed = mogp_emulator.LatinHypercubeDesign([(-120., -80.), (0.1, 0.4), (0.9, 1.1)])

# We can now generate a design of 20 sample points by calling the sample method:

np.random.seed(157374)

input_points = ed.sample(20)

# Now we can actually run the simulations. First, we feed the input points to create_problem to
# write the input files, call run_simulation to actually simulate them, and compute_moment to
# load the data and compute the earthquake size. The simulation is parallelized, so if you
# have multiple cores available you can specify more processors to run the simulation.

# Each simulation takes about 20 seconds on 4 processors on my MacBook Pro, so the entire design
# will take several minutes to run.

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

# Now fit a Gaussian Process to the input_points and results to fit the approximate model. We use
# the maximum marginal likelihood method to estimate the GP hyperparameters

gp = mogp_emulator.GaussianProcess(input_points, results)
gp.learn_hyperparameters()

# We can now make predictions for a large number of input points much more quickly than running the
# simulation. For instance, let's sample 10000 points

np.random.seed()

query_points = ed.sample(10000)
predictions = gp.predict(query_points)

# predictions contains both the mean values and variances from the approximate model, so we can use this
# to quantify uncertainty given a known value of the moment.

# Since we don't have an actual observation to use, we will do a synthetic test by running an additional
# point so we can evaluate the results from the known inputs.

known_value = 58.
threshold = 3.

# One easy method for comparing a model with observations is known as History Matching, where you
# compute an implausibility measure for many sample points given all sources of uncertainty
# (observational error, approximate model uncertainty, and "model discrepancy" which is a measure of
# how good the model is at describing reality). For simplicity here we will only consider the
# approximate model uncertainty, but for real situations it is important to include all three sources.
# The implausibility is then just the number of standard deviations between the predicted value and
# the known value.

# To compute the implausibility, we use the HistoryMatching class, which requires the observation,
# query points (coords), and predicted values (expectations), plus a threshold above which we can
# rule out a point

hm = mogp_emulator.HistoryMatching(obs=known_value, expectations=predictions,
                                   threshold=threshold)

implaus = hm.get_implausibility()
NROY = hm.get_NROY()

# now make a plot to visualize the space

# need to normalize outputs to triangulate plot

norm_points = np.zeros(query_points.shape)
norm_points[:,0] = -(query_points[:,0]-80.)/40.
norm_points[:,1] =  (query_points[:,1]-0.1)/0.3
norm_points[:,2] =  (query_points[:,2]-0.9)/0.2

# select the projection of input space that we want to view

axis1 = 1
axis2 = 0

# function to generate correct plot lables

def label_generator(axis):
    if axis == 0:
        label = "Fault Compressive Stress (MPa)"
    elif axis == 1:
        label = "Shear to Normal Stress Ratio"
    elif axis == 2:
        label = "Out of Plane to In Plane Stress ratio"
    else:
        raise ValueError("Bad value for axis")

    return label

def limits_generator(axis):
    if axis == 0:
        limits = (-120., -80.)
    elif axis == 1:
        limits = (0.1, 0.4)
    elif axis == 2:
        limits = (0.9, 1.1)
    else:
        raise ValueError("Bad value for axis")

    return limits

tri = matplotlib.tri.Triangulation(norm_points[:,axis1], norm_points[:,axis2])

plt.figure(figsize=(4,3))
plt.tripcolor(query_points[:,axis1], query_points[:,axis2], tri.triangles, implaus, vmin=0., vmax=6., cmap="viridis_r")
cb = plt.colorbar()
cb.set_label("Implausibility")
#plt.plot(known_input[0,axis1], known_input[0,axis2], "or")
plt.xlabel(label_generator(axis1))
plt.ylabel(label_generator(axis2))
plt.title("Implausibility Metric")
plt.savefig("implausibility.png", dpi=200, bbox_inches="tight")

plt.figure(figsize=(4,3))
plt.scatter(query_points[hm.get_NROY(), axis1], query_points[hm.get_NROY(), axis2])
plt.xlim(limits_generator(axis1))
plt.ylim(limits_generator(axis2))
plt.xlabel(label_generator(axis1))
plt.ylabel(label_generator(axis2))
plt.title("NROY points")
plt.savefig("nroy.png", dpi=200, bbox_inches="tight")
plt.show()