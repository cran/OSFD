% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/OSFD.R
\name{mMdist}
\alias{mMdist}
\title{Minimax distance}
\usage{
mMdist(X, X_space)
}
\arguments{
\item{X}{a matrix specifying the design.}

\item{X_space}{a large sample of uniform points in the space of interest.}
}
\value{
the minimax distance.
}
\description{
\code{mMdist} computes the minimax distance of a deisng in a specified region. A large uniform sample 
from the specified region is need to compute the minimax distance.
}
\details{
\code{mMdist} approximates the minimax distance of a set of points \code{X} by the large sample \code{X_space} in the space of interest.
}
\examples{
# the minimax distance of a random Latin hypercube design
D = randomLHS(5,2)
mMdist(D,replicate(2,runif(1e5)))


}
\references{
Johnson, Mark E., Leslie M. Moore, and Donald Ylvisaker. "Minimax and maximin distance designs." Journal of statistical planning and inference 26.2 (1990): 131-148. 

Wang, Shangkun, Adam P. Generale, Surya R. Kalidindi, and V. Roshan Joseph. "Sequential Designs for Filling Output Spaces." Technometrics, to appear (2023).
}
