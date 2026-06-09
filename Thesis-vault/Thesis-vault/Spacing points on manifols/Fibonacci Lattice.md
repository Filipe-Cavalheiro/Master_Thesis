  The Fibonacci lattice, also heavily inspired and influenced by \cite{Latitude_Longitude_vs_fibonacci}, is named after the Fibonacci ratios, which was discovered in ancient India \cite{fibonacci_history}. The construction is based on the Fibonacci sequence, in which each term from the third onward is the sum of the two preceding terms:

\[

0, 1, 1, 2, 3, 5, 8, 13, 21, \ldots

\]

  

Given two consecutive Fibonacci numbers $F_i$ and $F_{i+1}$, the Fibonacci ratio is defined as $F_{i+1}/F_i$. As first proved by Robert Simson in 1753 \cite{fibonacci_to_golden_ratio_1,fibonacci_to_golden_ratio_2}, this ratio converges rapidly, as $i \to \infty$, to the golden ratio

\begin{equation}

  \varphi = 1 + \varphi^{-1} = \frac{1 + \sqrt{5}}{2} \simeq 1.618 .

  \label{eq:golden_ratio}

\end{equation}

  

The Fibonacci lattice differs in the use of the \emph{golden angle} along the spiral lattice on the sphere where the longitudinal rotation between consecutive points along the generating spiral is given by \ref{eq.3:golden_angle},

or, equivalently, its complementary angle \ref{eq.3:golden_angle_complementary}.

  

\begin{equation}

  360^\circ \left(1 - \varphi^{-1}\right)

  = 360^\circ \varphi^{-2}

  \simeq 137.5^\circ ,

  \label{eq.3:golden_angle}

\end{equation}

  

\begin{equation*}

  360^\circ \varphi^{-1} \simeq 222.5^\circ .

  \label{eq.3:golden_angle_complementary}

\end{equation*}

  

In some lattice constructions, the golden ratio $\varphi$ is replaced by a rational approximant given by a Fibonacci ratio; this is the case when the total number of points is constrained to be a Fibonacci number. But why use the golden angle as the spacing between points in the sphere lattice? Since the golden angle optimizes packing efficiency in spiral lattices, due to being the golden ratio the \emph{most irrational} number, the  periodic or near-periodic alignments in the spiral structure is avoided. As a result, lattice points do not cluster, and spatial uniformity is preserved.

  

When a Fibonacci ratio is employed, the total number of lattice points is $F + 1$, where $F > 1$ is a Fibonacci number. In contrast, the lattice used in this work is based directly on the golden ratio and admits any odd number of points.

  

To construct the lattice, let $N$ be a natural number and let the integer index $i$ range from $-N$ to $+N$.

  

The spherical coordinates (in radians) of the $i$th point are given by Equation~\ref{eq.3:fibonacci_lat_lon} being the total number of points defined by \ref{eq.3:fibonacci_number_of_points}.

  

\begin{align}

    \text{lat}_i &= \arcsin\!\left(\frac{2i}{2N + 1}\right), \\

    \text{lon}_i &= 2\pi i \varphi^{-1} .

    \label{eq.3:fibonacci_lat_lon}

\end{align}

  

\begin{equation}

    P = 2N + 1

    \label{eq.3:fibonacci_number_of_points}

\end{equation}

  

Each point is assigned a unique latitude, while the longitudinal separation between consecutive points corresponds to the golden angle (or its complement), approximately $222.5^\circ$. Also notice from equation \ref{eq.3:fibonacci_number_of_points} that the total number of points is always odd; therefore, the number of agents cannot be any even natural number greater than 1.

  

\begin{equation*}

    P \in 2N + 1 \;\Rightarrow\; P \notin 2\mathbb{N}

\end{equation*}

  

Finally the interactive visualization provided in \cite{fibonacci_lattice_internet} offers an intuitive and informative illustration of the Fibonacci lattice structure. It is also possible to see the resulting formation created by this method in Figure~\ref{fig.3:fibonacci_lattice_demonstration} where its clearly visible the lattice points do not cluster, and spatial uniformity is preserved.

  

\begin{figure}[htbp]

\centering

\subfloat[Normal]{\includegraphics[width=3cm]{5-Figures/Chapter4/fibonacci_sphere/fibonacci_sphere_normal.png}}\hfil  

\subfloat[Top]{\includegraphics[width=3cm]{5-Figures/Chapter4/fibonacci_sphere/fibonacci_sphere_top.png}}\hfil

\subfloat[Side]{\includegraphics[width=3cm]{5-Figures/Chapter4/fibonacci_sphere/fibonacci_sphere_side.png}}

\caption{Fibonacci Lattice demonstration (1001 points)}

\label{fig.3:fibonacci_lattice_demonstration}

\end{figure}