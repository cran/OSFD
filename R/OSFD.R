#' @name 
#' OSFD
#' 
#' @title
#' Output space-filling design
#'
#' @description
#' This function is for producing designs that fill the output space.
#' 
#' @details 
#' \code{OSFD} produces a design that fills the output space using the sequential algorithm by Wang et al. (2023).
#' 
#' @param D a matrix of the initial design. If not specified, a random Latin hypercube design of size n_ini and dimension p will be generated as initial design.
#' @param f black-box function.
#' @param p input dimension.
#' @param q output dimension.
#' @param n_ini the size of initial design. This initial size must be specified if D is not provided.
#' @param n the size of the final design.
#' @param scale whether to scale the output points to 0 to 1 for each dimension.
#' @param method two choices: 'EI' or 'Greedy'; the default is 'EI'.
#' @param CAND the candidate points in the input space. If Null, it will be automatically generated.
#' @param rand_out whether to use random uniform points or quasi random points by twinning algorithm for generating points in spheres for output space approximation. The default value is FALSE.
#' @param rand_in  whether to use random uniform points or quasi random points by twinning algorithm for generating points in spheres for input space candidate sets. The default value is FALSE.
#' 
#' 
#' @return 
#' \item{D}{the final design points in the input space}
#' \item{Y}{the output points}
#' 
#' @references 
#' Wang, Shangkun, Adam P. Generale, Surya R. Kalidindi, and V. Roshan Joseph. "Sequential Designs for Filling Output Spaces." Technometrics, to appear (2023).
#'
#' @export
#' 
#' @examples
#' # test function: inverse-radius function (Wang et.al 2023)
#' inverse_r = function(x){
#' epsilon = 0.1
#' y1=1/(x[1]^2+x[2]^2+epsilon^2)^(1/2)
#' if (x[2]==0){
#'  y2 = 0
#' }else if (x[1]==0) {
#'   y2 = pi/2}else{
#'     y2 = atan(x[2]/x[1])
#'   }
#' return (c(y1=y1,y2=y2))
#' }
#' 
#' set.seed(2022)
#' p = 2
#' q = 2
#' f = inverse_r
#' n_ini = 10
#' n = 50
#' osfd = OSFD(f=f,p=p,q=q,n_ini=n_ini,n=n)
#' D = osfd$D
#' Y = osfd$Y
#' 
OSFD  = function(D=NULL,f,p,q,n_ini=NA,n,scale=TRUE,method='EI',CAND=NULL,rand_out=FALSE,rand_in=FALSE){
  if(is.null(D)){
    if (is.na(n_ini)) stop('Please specifiy the size of initial design.')
    D = randomLHS(n_ini,p)
  }
  if (q==1) {
    Y = matrix(apply(D, 1, f),ncol=1)
  }else{
    Y = t(apply(D, 1, f))}
  if (method=='EI'){
    EI = TRUE;
  }else if (method=='Greedy'){
    EI = FALSE;
  }else{
    stop('Please specify method from EI or Greedy!')
  }
  n_ini = nrow(D)
  
  k1 = 2*min(p,q)
  if (q==1| p==1){rand_out = TRUE}
  if (rand_out == FALSE){
    # generate twinning sample used in output balls
    n_ball = k1+2*(min(p,q)+1)+1 # number of point in each ball
    twinsample_out = runif_in_sphere_cpp(100*n_ball,min(p,q))
    twinsample_out = twinsample_out[twin(twinsample_out,r=100),,drop=FALSE]
  }else{
    twinsample_out = NULL
  }
  
  if (p==1){rand_in = TRUE}
  if (rand_in == FALSE){
    # generate twinning sample used in intput balls
    n_ball = 10*p # number of point in each ball
    twinsample_in = runif_in_sphere_cpp(100*n_ball,p)
    twinsample_in = twinsample_in[twin(twinsample_in,r=100),,drop=FALSE]
  }else{
    twinsample_in = NULL
  }
  
  # Progress bar
  pb <- txtProgressBar(min = 0,max = (n-n_ini),style = 3,width = 50,char = "=")
  
  # two step alg
  if (!is.null(CAND)){
    CAND = as.matrix(dplyr::setdiff(as.data.frame(CAND),as.data.frame(D)))
  }
  
  for (i in 1:(n-n_ini)){
    if (scale){
      Y_scale = scale(Y)
      filldist = filldist_cpp(Y_scale,p,rand=rand_out,twinsample_=twinsample_out)
    }else{
      filldist = filldist_cpp(Y,p,rand=rand_out,twinsample_=twinsample_out)
    }
    unew = perturb_cpp (D,filldist,CAND_=CAND,EI=EI,rand=rand_in,twinsample_=twinsample_in)
    
    D = rbind(D,unew)
    Y = rbind(Y,f(unew))
    
    if (!is.null(CAND)){
      CAND = as.matrix(dplyr::setdiff(as.data.frame(CAND),as.data.frame(matrix(unew,ncol=p))))
    }
    
    setTxtProgressBar(pb, i)
  }
  return(list(D=D,Y=Y))
}

#' @name 
#' mMdist
#' @title
#' Minimax distance
#'
#' @description
#' \code{mMdist} computes the minimax distance of a deisng in a specified region. A large uniform sample 
#' from the specified region is need to compute the minimax distance.
#' 
#' @details 
#' \code{mMdist} approximates the minimax distance of a set of points \code{X} by the large sample \code{X_space} in the space of interest.
#' 
#' @param X a matrix specifying the design.
#' @param X_space a large sample of uniform points in the space of interest.
#' 
#' @return the minimax distance.
#' 
#' @references 
#' Johnson, Mark E., Leslie M. Moore, and Donald Ylvisaker. "Minimax and maximin distance designs." Journal of statistical planning and inference 26.2 (1990): 131-148. 
#' 
#' Wang, Shangkun, Adam P. Generale, Surya R. Kalidindi, and V. Roshan Joseph. "Sequential Designs for Filling Output Spaces." Technometrics, to appear (2023).
#'
#' @export
#' 
#' @examples
#' # the minimax distance of a random Latin hypercube design
#' D = randomLHS(5,2)
#' mMdist(D,replicate(2,runif(1e5)))
#' 
#' 
mMdist = function (X,X_space){
  return(max(knnx_my(X,X_space,k=1)$nn_dist))
}


#' @name 
#' ball_unif
#' @title
#' (Quasi) uniform points in a p-dimensional ball
#'
#' @description
#' \code{ball_unif} generate random or quasi-random uniform points in a p-dimensional ball.
#' 
#' @details 
#' \code{ball_unif} generate random uniform points or quasi uniform points by twinning algorithm in a p-dimensional ball.
#' 
#' @param cen a vector specifying the center of the ball.
#' @param rad radius of the ball.
#' @param n number of points.
#' @param rand whether to generate random or quasi random points. Default value is TRUE.
#' 
#' @return a matrix of the generated points.
#' 
#' @references 
#' Vakayil, Akhil, and V. Roshan Joseph. "Data twinning." Statistical Analysis and Data Mining: The ASA Data Science Journal 15.5 (2022): 598-610.
#' 
#' Wang, Shangkun, Adam P. Generale, Surya R. Kalidindi, and V. Roshan Joseph. "Sequential Designs for Filling Output Spaces." Technometrics, to appear (2023).
#'
#' @export
#' 
#' @examples
#' 
#' x = ball_unif(c(0,0),1,10,rand=FALSE)
#' plot(x,type='p')
#' 
ball_unif = function (cen,rad,n,rand=TRUE){
  p = length(cen)
  if (rand){
    point = ball_gen(cen, rad, n, p, rand)
  }else{
    twinsample = runif_in_sphere_cpp(500*n,p)
    twinsample = twinsample[twin(twinsample,r=500),,drop=FALSE]
    point = ball_gen(cen, rad, n, p, rand=FALSE,twinsample_=twinsample)
  }
  return(point)
}



