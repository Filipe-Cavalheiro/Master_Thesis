# MATLAB prototypes

Early MATLAB prototypes of the thesis "Cooperative Encirclement of Non-Cooperative Drones": a swarm of chaser drones
spreads over a sphere (or another surface) around a moving target, with decentralized control and collision
avoidance. The folders are self-contained. Run each script from its own folder (it uses relative paths and local
helper functions). Most folders are learning exercises or paper reproductions. The main line of development is
`fibonacci_sphere/` (v1-v9), continued in `voronoin/` (v10-v14).

## Folders

| Folder | Content | Entry point |
|---|---|---|
| `2D_chase` | Kim & Sugie (2007) cyclic pursuit target capturing, case 1: 9 agents encircle a moving target at distance R. Despite the name, the simulation is in 3D. `vary_thet_and_phi.m` is the same model with the desired angles varied. | `Cooperative_Control_forTarget_Capturing_Task.m` |
| `Attitude_and_angular_velocity_tracking_for_a_rigid_body_using_geometric_methods_on_the_two-sphere` | Reproduction of the numerical simulation of the paper named by the folder: reduced attitude (pointing direction q on S^2) tracking with the geometric error function and control law; the goal changes at Tend/2. `psi_r_vs_psi.m` plots the error function against the angle. This controller is the outer loop used from `fibonacci_sphere/control_sphere_v5` on. | `main_sim.m` |
| `capture-drone-models-main` | Drone models of Project CAPTURE (shuttle, target-atomic, target-blast, target-easyglider; MATLAB and Simulink). **Confidential Project CAPTURE material**, not to be used or reproduced without authorization. See its [readme.md](capture-drone-models-main/readme.md). | see its readme |
| `CBF_learning` | Control barrier function exercises, one script each: `car_R2_CBF.m` (unicycle, 2D obstacle), `drone_R3_CBF.m` / `drone_R3_HOCBF.m` (single / double integrator in R^3), `multi_obstacle_drone_R3_HOCBF.m`, `box_R3_HOCBF.m` (stay inside a box, `quadprog`), and on S^2 `drone_S2_CBF.m`, `drone_S2_HOCBF_cartisian.m`, `drone_S2_HOCBF_geodesic*.m` and `drone_S2_CLF_HOCBF_geodesic_{open_loop,closed_loop,inverted}.m` (CLF + HOCBF quadratic program, several obstacle positions; "inverted" changes the sign of the barrier). `AM_QP/` plans Bernstein polynomial trajectories for 4 drones with collision avoidance by alternating minimization (`solveS1.m` ... `solveS5.m`). `Lee+CBF/` runs the Lee geometric controller (FDCL code, `Lee_controller/`); its `R3_spherical_CBF.m` is unfinished. | the script of interest; `AM_QP/main.m`; `Lee+CBF/main.m` |
| `Consensus_learning` | Average consensus with a fully connected Laplacian: scalar values for 10 agents (`basic_fully_connected.m`), R^3 values (`basic_fully_connected_R3.m`), and a target moving on a circle measured with noise, filtered with an exponential moving average and agreed by consensus (`fully_connected_exponential_moving_targetR3.m`, figure `moving_target_consensus.pdf`). | any of the three scripts |
| `direct_computation_equal_points_method_comparison` | Box plot of the execution times of the direct (non-iterative) point distribution methods, from the `.mat` files of the other folders (Fibonacci, latitude-longitude, Poisson disk, Thomson). Output `fibonacci_exec_time.pdf`. | `compare_methods.m` |
| `fibonacci_sphere` | Fibonacci-sphere formation around a target, versions v1-v9 (table below). | see below |
| `geometric_transforms` | Animated morphing of a point set or mesh grid: disc -> cone -> sphere (`circle_2_cone_2_sphere.m`, `mesh_grid_circle_2_cone_2_sphere.m`). `pos_based_mesh_transform.m` morphs a Fibonacci disc into a cone and a sphere depending on the position (inside / border / outside), figures `mesh_controller_*.pdf`. | any of the three scripts |
| `iterative_computation_equal_points_method_comparison` | Box plot of the execution times of the iterative methods (Thomson, Lloyd with convex hull; Lloyd with half planes is commented out), from the `.mat` files in the folder. Output in `PDF/`. | `compare_methods.m` |
| `Lattitude and longitude` | Points on a sphere on a regular latitude-longitude grid, with `sphere_uniformity.m` metrics. Figures `latlon_*.png`. | `Lattitude_longitude.m` |
| `manifolds_projections` | Stereographic projection of a sphere onto a plane (`untitled.m`) and a PCA flattening function (`Principal Component Analysis.m`, defines `flatten_surface_pca`). Small exercises. | `untitled.m` |
| `mellinger+minimum_snap_traj` | Project CAPTURE drone simulation (full 3D dynamics + Mellinger controller, 3 drones) used as the base of the later versions. The minimum snap generator is defined in `drone_main_simul.m` but its call is commented out: the reference is a straight line. `sphere_points_with_distance.m` (fmincon) is not called. | `drone_main_simul.m` (calls `drone_init.m`; plots with `drone_show_data.m`) |
| `posissonDIskSampling` | Bridson Poisson disk sampling, in a 2D rectangle (`R2/`, timed over 100 runs) and on the unit sphere with a minimum angular distance (`two_sphere/`, with `sphere_uniformity.m` and figures `poisson_disk_*.pdf`). | `R2/main.m`, `two_sphere/main.m` |
| `thomson_minimum_energy_sphere` | Thomson problem: minimum Coulomb energy arrangement of points on the sphere (`thomson11.m`), 30 points, timed over 100 runs. Results in `thomson_execution_times*.mat` and `thomson_*.png`. | `main.m` |
| `voronoin` | Voronoi / CVT goal assignment on the sphere and other surfaces, versions v10-v14 (table below), and the `Learning/` exercises. | see below |

## `fibonacci_sphere` versions

All versions from v3 on use the Project CAPTURE drone simulation (`drone_init.m` with the parameters, `drone_main_simul.m`
the loop, Mellinger controller and full 3D dynamics). From v5 on, each chaser has an outer loop: its reference q is a
point on a sphere around the target, driven to its Fibonacci goal q_d by the S^2 attitude tracking controller of the
`Attitude_and_angular_velocity_...` folder (`linear_motion_ref.m`, named "laypunov" in the folders), and the drone tracks q.

| Version | What it adds | Entry point |
|---|---|---|
| `control_sphere_v1` | Fibonacci points on a sphere and the first kinematic encirclement: 9 agents driven in (r, theta, phi) to the Fibonacci angles around a moving target (`control_sphere.m`, `control_sphere_v1_5.m`). Subfolders try variants of the point set (mean distance, mean theta, mean theta with equal z) with figures. `fibonacci_sphere/points_on_sphere_fibonacci.m` times the generation and computes `sphere_uniformity.m`. `learning.m` is an unconstrained MPC exercise. `Spherical_law_of_cosines.m` checks distances on the sphere. | `control_sphere_v1_5.m` |
| `control_sphere_v2_solvers_0` | Fibonacci points moved by `fmincon` (`sphere_points_with_distance.m`) so that the K nearest neighbours keep the average distance. No drones. | `control_sphere_v2.m` |
| `control_sphere_v3_solvers_1` | The fmincon points (`control_sphere_v3.m`) as references for 3 simulated drones, with solver metrics. | `drone_main_simul.m` |
| `control_sphere_v4_APF_cartesean` | Unconstrained MPC to the Fibonacci points with a radius that shrinks from `sphere_radius` to `end_radius`, and a repulsion along the sphere tangent when a neighbour is closer than the minimum Cartesian distance. 3 drones. | `drone_main_simul.m` |
| `control_sphere_v4_APF_geodesic` | Same idea with geodesic distances between the positions projected on the sphere, a clamped repulsion vector and the reference projected back on the sphere. Minimum safe distance from `get_min_safe_distance.m`. Figure `APF.pdf`. | `drone_main_simul.m` |
| `control_sphere_v5_laypunov_1_drone` | First S^2 (Lyapunov) outer loop, one drone. `main_sim.m` is the stand-alone S^2 controller; `latex/` holds a short write-up. | `drone_main_simul.m` |
| `control_sphere_v5_laypunov_9_drone` | The S^2 outer loop for several drones, one Fibonacci goal each (`Param.nD` in `drone_init.m`, currently 4). Figures `lyapunov.pdf`, `result_fig.pdf`, write-up in `latex/`. | `drone_main_simul.m` |
| `control_sphere_v6_laypunov_example_1` | Radius dynamics (`update_radius.m`): the drones start far away and the radius converges to `sphere_radius`. 3 drones, target fixed at the origin. Animation (`animate_simulation.m`, `.mp4`), figures in `PDF/`, write-up in `latex/`. | `drone_main_simul.m` |
| `control_sphere_v6_laypunov_example_2` | Same as example 1 with other initial positions and radius gain (`k_geo`), target at the origin. Snapshots `final_result_t_*.pdf`, `.mp4`. | `drone_main_simul.m` |
| `control_sphere_v6_laypunov_example_3` | 9 drones starting on a grid, target moving on a circle until t = 8 s and then still. `.mp4`, `output.pdf`. | `drone_main_simul.m` |
| `control_sphere_v7_laypunov` | The target is a simulated drone (`target_motion.m`, Mellinger) following a circle. `update_radius.m` has four radius laws (`V = 1..4`: position, position + velocity, tangential velocity, ...), compared in `option_1.png` ... `option_4.png`. | `drone_main_simul.m` |
| `control_sphere_v8_laypunov+APF` | Geodesic APF collision avoidance on the reference (`APF_safe_guard.m`), minimum safe distance from the Delaunay graph of the Fibonacci points (`Delaunay_triangulation_network.m`). Plot scripts in `Animation_display/`. Results for 3 and 31 drones in `Results/FIGS` (`.fig`) and `Results/PDF`. | `drone_main_simul.m` |
| `control_sphere_v8_laypunov+APF_test` | Test copy of v8 with the APF call commented out: figures with the APF active / inactive and with a moving / stationary target (`APF_test_*.pdf`, `final_result_t_*.pdf`, `results_drone_1.*`). | `drone_main_simul.m` |
| `control_sphere_v9_laypunov+APF+consensus` | Each chaser measures the target with noise (`noise_std`), filters it with an exponential moving average (`ema_alpha`) and runs consensus (fully connected Laplacian) on the estimate. 3 drones. | `drone_main_simul.m` |
| `control_sphere_v9_laypunov+CBF` | The APF is replaced by a CLF + HOCBF on S^2 (`CLF_and_CBF.m`): one geodesic barrier per neighbour, applied as a closed-form half-space projection of the control. 9 drones. Snapshots `final_result_t_*.pdf`, `Lloyd_half_plane_*.pdf`. | `drone_main_simul.m` |

## `voronoin` versions

| Folder | Content | Entry point |
|---|---|---|
| `Learning/convex_hull/S2_voronoi` | Spherical Voronoi diagrams from the convex hull, centralized and decentralized (Delaunay or small-world neighbours), with agent drop. | `S2_voronoin.m` and the other `S2_voronoi*.m` |
| `Learning/halth_plane_method` | Voronoi cells by half-plane clipping: `R2/` (centralized vs decentralized, small-world network, consensus), `S2/` (`voronoi_half_plane.m`, timing in `.mat`), `generic_manifold/V1/`. | the `R2_voronoi*.m`, `S2/voronoi_half_plane.m`, `V1_voronoi_generic_half_plane.m` scripts |
| `Learning/disktra_method/V1` | Geodesic CVT on a triangle mesh with Dijkstra distances (`BinaryMinHeap.m`, `MeshVertex.m`), on generated surfaces and on `frog.stl` / `bust.stl`. | `main.m` (`main_time_measurement.m` for timing) |
| `v10_laypunov+consensus+decentralized_convexhull` | v9 consensus + S^2 outer loop, with the goals from a decentralized Lloyd step on the spherical Voronoi cell of each drone (convex hull, `voronoi_update.m`). 12 drones. | `drone_main_simul.m` |
| `v11_laypunov+consensus+decentralized_halfplane` | Same as v10 with half-plane Voronoi cells (`voronoi_update_halfplane.m`). Figures in `PDF/`. | `drone_main_simul.m` |
| `v12_laypunov+consensus+decentralized+dikstra` | Dijkstra geodesic CVT on a mesh (`VoronoiCVT/`): circle, sphere, ellipsoid, paraboloid or the STL objects in `Objects/` (`Param.objList`). 3 drones. | `drone_main_simul.m` |
| `v13_consensus+decentralized+dikstra` | Restructured: `Param_init.m`, target in `target/`, target estimation in `estimate_target_variables.m`, precomputed disc / cone / sphere meshes in `mesh/`, goal distribution `fibonacci` or `dijkstra` (`Param.distribuition`), drone drop (`nD_drop`, `nD_drop_time`). Results for 5 and 11 drones in `Results/FIGS` and `Results/PDF`. | `drone_main_simul.m` |
| `v14` | v13 with a choice of position controller, Mellinger or Lee (`Param.controller`, `actuate_control.m`, `Lee_controller/`). Results for 3 drones in `Results/`. | `drone_main_simul.m` |

v13 and v14 save the workspace to `log_vars.mat` in the run folder. `voronoin/` also holds `.fig` and `.pdf` figures next to the scripts.

## Shared helpers

Many folders have their own copy of the same helpers:

- `print2pdf.m` (`print2pdf(name, doprint)`): exports the current figure to PDF when `doprint` is true. There are two
  variants. The simple one calls `exportgraphics` on `name` in the current folder (some copies first enlarge the figure
  to 30 x 20 cm). The other saves `FIGS/<name>.fig` and `PDF/<name>.pdf` under the script folder (`Results/FIGS` and
  `Results/PDF` in `control_sphere_v8_laypunov+APF`). It builds the paths from `'.\'`, a Windows path.
- `sphere_uniformity.m`: uniformity metrics of a point set on the sphere (ratio of minimum to mean nearest-neighbour
  distance, coefficient of variation of the nearest-neighbour distances, Riesz energy). The copy in
  `thomson_minimum_energy_sphere` also compares against uniformly random points.
- Project CAPTURE drone simulation files (Bruno Guerreiro): `drone_init.m`, `drone_main_simul.m`, `drone_3dfull_dyn.m`,
  `drone_mellinger_ctrl.m`, `drone_show_data.m`, `drone_plot.m`, `skew.m`, `unskew.m`, `Euler2R.m`, `R2Euler.m`,
  `rot_integrate.m`.
- Sphere helpers in the `fibonacci_sphere` / `voronoin` versions: `get_fibonacci_points.m`, `projectToSphere.m`,
  `geo_distances.m`, `cartisian_distances.m`, `get_min_safe_distance.m`, `Delaunay_triangulation_network.m`.

Figures and results are stored next to the scripts that make them (`.pdf`, `.png`, `.fig`, `.mp4`), or in the
`PDF/`, `FIGS/` or `Results/` subfolders created by `print2pdf.m`. Timing data and saved runs are `.mat` files in the
same folders.
