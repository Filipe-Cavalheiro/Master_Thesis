2026-02-19: Checked the paper for ICUAS2026 one last time for any error, studied [[Control Barrier Function (CBF)]] @ames2019controlbarrierfunctionstheory. Also created a github of the paper for professor Bruno to read. Made 3 versions of CBF on in R2, R3 and S2 (two-sphere), this last one does not work.

2026-02-20:  Understand the CBF ([[Lie Derivatives]]) and how to get the correct CBF equation from the working R2 and R3 examples.


2026-02-23: Rewrite the paper for ICUAS 2026 as per indicated from the review of the professor Bruno, read @xiaoHighOrderControlBarrier2022

2026-02-24:  banking, read @xiaoHighOrderControlBarrier2022

2026-02-25: read @xiaoHighOrderControlBarrier2022 and try to solve $\mathbb{S}^2$ [[HOCBF in SO(2)]], was able to have a working solution!!! but is it really the right solution?!

2026-02-26: read @xiaoHighOrderControlBarrier2022 and verify the solution in $\mathbb{S}^2$ [[HOCBF in SO(2)]]. 

2026-02-27: Wrote and rewrote the pdf explaining the thought process of the proposed [[Drone S2 CLF+HOCBF QP geodesic]] without obtaining a desired conclusion, it never worked. :( gave up on it for now and will start consensus tomorrow.

2026-02-27: read @olfati-saberConsensusCooperationNetworked2007 and started implementing the first algorithm separated from any other code

2026-03-09: read @olfati-saberConsensusCooperationNetworked2007 and implemented the first iteration of consensus to the CLF+APF having now a CLF + APF + [[Consensus]]. The version of consensus for a stationary target has been implemented now I need to implement also for a moving target, this means it is now only the consensus of the initial value but a exponential moving target average [[Exponential Moving Average (EMA)]].

2026-03-10: started searching for methods of distributing points on a manifold that do not require the knowledge of position of all agents or number of agents, at the moment the [[Fibonacci Lattice]] is being used and it does require the number of agents to be known so other methods such as the Fortune's algorithm exist but it still need to know the position of all agents so maybe, Delaunay triangulation and Lloyd's algorithm might be useful. 

| Method                | Global knowledge | Decentralized |
| --------------------- | ---------------- | ------------- |
| Fortune's algorithm   | Yes              | ❌             |
| Local bisector method | No               | ✅             |
| Distributed Lloyd     | No               | ✅             |
2026-03-11: Read about decentralized system just like yesterday with @boldrerDistributedLloydBasedAlgorithm2026 , @lloydLeastSquaresQuantization1982, @cortesCoverageControlMobile2004 The idea will be to implement the decentralized version of Fortune algorithm which seams to be Distributed Lloyd algorithms that only depend on the neighbours. So it appears that Lloyd algorithms is also necessary to know the position of all agents which is not the desired system, Convex Weighted Voronoi Diagram is the one that is indeed decentralized, btw the difference between Lloyd and Fortune's is that Fortune's  is based on the progression of a line thought the space while Lloyd is more closely resembled by k clusters.

2026-03-16: Voronoi in Riemann manifolds study and comprehension. And implementation made if work in S2 just need to pass it to the main simulation. This version is decentralized!

2026-03-17: Work in the main simulation, the spacing between agents is not uniform from the standard Centroidal Voronoi tessellation so something need to be added like a consensus on intra distance, in this case an APF and consensus on the small world network already developed will be made. 

2026-03-18: Passed the centroid calculation of the voronoi in S2 from the average of the vertices which is a very bad approximation, to a much better calculation of the centroid of the spherical cap 
on the surface it self. At the moment it does not work for other sphere radii other than the unit sphere.

2026-03-19: Try to pass the sphere radii to other values other than the unit sphere correctly.

2026-03-20: Check the restricted air space of Lisbon.

2026-03-23: Write thesis, it is over due for 2 months already.
wrote the separation between centralized and decentralized systems

2026-03-24: write thesis, wrote about the new controller architecture both centralized missing APF and consensus and decentralized APF small world network, decentralized Voronoi, decentralized Voronoi. 

TODO: show simulations of decentralized version
TODO: check every thing that was written for consistency and correctness
TODO: stop calling it decentralized 

2026-03-26: Write thesis, do the TODO above :)
Ended up rewrite the code to not use K-nearest neighbour but instead Delaunay Triangulation.
And some TODO.

2026-03-27:  Write thesis, do the TODO above :)

2026-03-30:  Write thesis, have to write the testes of the decentralized system, welp ended up non only writing the testing but noticing that it is terrible for N <= 12 this is due to the convexhull method not working for small k because the geodesic curve chosen is ALWAYS the smallest and that might not be the correct geodesic to use. I have implemented the R2 version of half plane method and need to try the S2 version to see if it works with 2+ agents. not only 12+

2026-03-31: write thesis, see if I can finish this version by today. AKA no conclusions but the rest written with all the work done at the moment. Need to add the Voronoi half plane implementation I did yesterday to the method comparison, if I have time still need to simulate and implement the simulation with this method of point separation.

2026-04-01: Write thesis and the code for the half plane simulations.

*SPRING BREAK*

2026-04-07: started working on a general version of the system that works for any manifold, also made a transformation of a circle to a cone to a sphere to intersect the target when its at the edge of the forbidden region. 

2026-04-07: Trying to make the half plane version work using mesh grid instead of circle since I need to be able to cut sections of the mesh and discover new centres of region.

(Forgot to log) mostly writing matlab for voronoi on meshgrids

2026-04-15: EXPO FCT

2026-04-16: Read @mengEfficientAlgorithmApproximate2023 and trying to understand/implement it 

Lmao this was be to much of a hassle to write so there is a giant gap here my bad.

TIME  GAP:  things done in the mean time, finiches the voronoi implemnetation still with Lyapunov so that points move on the surface but had to vary the desired radius for each point since the shape might not be sphere, implemented a mesh update between a $\mathbb{R}^2$ circle to cone to sphere that its used to create a physical barrier against the drone and remove the Lyapunov geomtric control.

2026-05-12: Reading papers about the Mellinger and Snap controller to try to prove stability

TIME GAP: Implement the Lee controller and grab the rejected ICUAS paper and rewrite it for the APCA 2026

2026-05-22: worked on the papers matlab figures and continues to write the paper .

2026-05-24: (weekend I know) any way worked on the papers matlab figures and continues to write the paper, also kind of refactored the figure plots and funcs.

2026-05-25: Finished paper for APCA 2026.

2026-05-26:  Finished the revised version of the paper for APCA 2026 also add a line saying that I do not take into account communication delay based on time/distance for consensus also refactored the code for animation and what is being used in simulation (target path, fibonacci or dijstra,  what to plot).

2026-05-27: yupiiii, guess what gota revise the Paper again, need more pictures and merge proposition 2 and 3 into one! yay, also 

2026-05-28: continue to write paper? almost nothing, I got the new distance between agents figs that prove the APF works, also started to work on V14 that refactors some parts of the code to use structs instead of arrays to store drone state its easier to now chaser{i}.pos then chaser{i}\[1:3\]. Also started having Params to choose between using the disjktra and fibonacci, Mellinger or Lee, target path and number of measurable variables of the target.

2026-05-29: Submitted Paper and finished V14 and sent it to github as well as some other things I should have already sent but didn't.

2026-01-01: Going to add the drone collision avoidance in $\mathbb{R}^3$ so that drones have a safety sphere around them.

Work to do:
- Fix the dijkstra separation that is not working for 3 elements
- work on maximal communication delay calculation
- Kalman Filter to estimate the target variables?
- run experimental trials
- presentation in manim