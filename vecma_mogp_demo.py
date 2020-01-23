import numpy as np
import matplotlib.pyplot as plt
import mogp_emulator
from earthquake import create_problem, run_simulation, compute_moment
import matplotlib.pyplot as plt
import matplotlib.tri

# This is a python script showing the mogp_emulator part of the demo, and contains the python
# code found in the tutorial. This code is included for illustration purposes, as it does not
# implement the workflow tools included with the FabSim plugin.

# create Latin Hypercube sample of the three stress parameters:
# first parameter is normal stress on the fault, which has a uniform distribution from -120 to -80 MPa
# (negative in compression)
# second parameter is the ratio between shear and normal stress, a uniform distribution from 0.1 to 0.4
# third is ratio between the two normal stress components, a uniform distribution between +/- 10%

ed = mogp_emulator.LatinHypercubeDesign([(-120., -80.), (0.1, 0.4), (0.9, 1.1)])

seed = None
sample_points = 20

np.random.seed(seed)
input_points = ed.sample(sample_points)

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

analysis_samples = 10000

analysis_points = ed.sample(analysis_samples)
predictions = gp.predict(analysis_points)

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

# now make two plots to visualize the space (projected into normal and shear/normal plane)

plt.figure()
plt.plot(analysis_points[NROY, 0], analysis_points[NROY, 1], 'o')
plt.xlabel('Normal Stress (MPa)')
plt.ylabel('Shear to Normal Stress Ratio')
plt.xlim((-120., -80.))
plt.ylim((0.1, 0.4))
plt.title("NROY Points")
plt.show()

plt.figure()
tri = matplotlib.tri.Triangulation(-(analysis_points[:,0]-80.)/40., (analysis_points[:,1]-0.1)/0.3)
plt.tripcolor(analysis_points[:,0], analysis_points[:,1], tri.triangles, implaus,
              vmin = 0., vmax = 6., cmap="viridis_r")
cb = plt.colorbar()
cb.set_label("Implausibility")
plt.xlabel('Normal Stress (MPa)')
plt.ylabel('Shear to Normal Stress Ratio')
plt.title("Implausibility Metric")
plt.show()