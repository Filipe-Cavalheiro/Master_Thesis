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


