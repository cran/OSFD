# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

scale_cpp <- function(M) {
    .Call(`_OSFD_scale_cpp`, M)
}

knnx_my <- function(data, points, k) {
    .Call(`_OSFD_knnx_my`, data, points, k)
}

knn_my <- function(data, k) {
    .Call(`_OSFD_knn_my`, data, k)
}

runif_in_sphere_cpp <- function(n, p) {
    .Call(`_OSFD_runif_in_sphere_cpp`, n, p)
}

ball_gen <- function(cen, rad, n, p = 0L, sub_ = NULL, rand = TRUE, twinsample_ = NULL) {
    .Call(`_OSFD_ball_gen`, cen, rad, n, p, sub_, rand, twinsample_)
}

filldist_cpp <- function(M, p, rand = TRUE, twinsample_ = NULL) {
    .Call(`_OSFD_filldist_cpp`, M, p, rand, twinsample_)
}

perturb_cpp <- function(D, filldist, CAND_ = NULL, EI = TRUE, rand = TRUE, twinsample_ = NULL) {
    .Call(`_OSFD_perturb_cpp`, D, filldist, CAND_, EI, rand, twinsample_)
}

