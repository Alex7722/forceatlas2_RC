#' ForceAtlas2 layout of a graph
#'
#' @description
#' `forceatlas2()` computes a ForceAtlas2 layout and stores it in the graph:
#' it returns the graph it was given with two node columns (by default `x` and
#' `y`) added. It is meant for [tidygraph::tbl_graph] objects and pipelines,
#' and works the same way on plain igraph graphs (the coordinates are then
#' stored as vertex attributes).
#'
#' `layout_forceatlas2()` runs the same algorithm and returns the coordinates
#' as a two-column matrix, like the `layout_*()` functions of igraph.
#'
#' Both can be used with ggraph: either plot the stored coordinates with
#' `ggraph(graph, layout = "manual", x = x, y = y)`, or let ggraph compute the
#' layout with `ggraph(graph, layout = forceatlas2)` (further arguments such
#' as `linlog = TRUE` are passed on).
#'
#' @details
#' ForceAtlas2 is a force-directed layout: nodes repulse each other like
#' charged particles while edges attract the nodes they connect like springs.
#' Its particularity is that the repulsion between two nodes is proportional
#' to the product of their degrees (plus one), which pulls poorly connected
#' nodes close to the hubs they are attached to and makes communities stand
#' out.
#'
#' This is an implementation in C of the algorithm described in Jacomy et al.
#' (2014). The forces, the adaptive speed and the default values are those of
#' the reference implementation distributed with Gephi, and the arguments map
#' to Gephi's settings as follows:
#'
#' | **Gephi**                     | **forceatlas2r**            |
#' |:------------------------------|:----------------------------|
#' | Scaling                       | `scaling_ratio`             |
#' | Gravity                       | `gravity`                   |
#' | Stronger Gravity              | `strong_gravity`            |
#' | LinLog mode                   | `linlog`                    |
#' | Dissuade Hubs                 | `dissuade_hubs`             |
#' | Prevent Overlap               | `prevent_overlap`           |
#' | Edge Weight Influence         | `edge_weight_influence`     |
#' | Normalize edge weights        | `normalize_weights`         |
#' | Inverted edge weights         | `invert_weights`            |
#' | Tolerance (speed)             | `jitter_tolerance`          |
#' | Approximate Repulsion         | `barnes_hut`                |
#' | Approximation                 | `theta`                     |
#' | Threads number                | `threads`                   |
#'
#' Two things differ from Gephi. Gephi runs until the user stops it, whereas
#' here the number of `iterations` is fixed in advance. And with the
#' Barnes-Hut approximation, Gephi counts the repulsion between nearby nodes
#' twice; this implementation counts it once, so that the approximation
#' converges to the exact computation when `theta` goes to zero.
#'
#' The mass of a node is its degree plus one, counting every edge (edge
#' weights and directions are ignored for the masses, multiple edges count
#' several times). Edge directions only matter when `dissuade_hubs = TRUE`.
#'
#' The initial positions are drawn at random with R's random number generator
#' unless `init` is given: call [set.seed()] beforehand to obtain reproducible
#' layouts. For a given starting point, the result is deterministic and does
#' not depend on the number of threads.
#'
#' @param graph A [tidygraph::tbl_graph] or an igraph graph.
#' @param iterations Number of iterations to run. Large graphs may need more
#'   iterations to settle, or fewer if you only want a rough layout quickly.
#' @param scaling_ratio Strength of the repulsion; larger values give a
#'   sparser, larger layout. `NULL` (the default) uses Gephi's default: 10 for
#'   graphs with fewer than 100 nodes and 2 otherwise.
#' @param gravity Strength of the force that attracts every node to the
#'   centre, which keeps disconnected components from drifting away.
#' @param strong_gravity If `TRUE`, gravity grows with the distance to the
#'   centre, which gives much more compact layouts.
#' @param linlog If `TRUE`, use a logarithmic attraction force (Noack's LinLog
#'   energy model), which makes clusters tighter. The layout is then much
#'   smaller: you will usually want to change `scaling_ratio` as well.
#' @param dissuade_hubs If `TRUE`, the attraction along an edge is divided by
#'   the mass of its source node, which pushes hubs to the periphery and
#'   favours authorities (nodes with a high indegree) in the centre.
#' @param prevent_overlap If `TRUE`, nodes are treated as discs of radius
#'   `node_size` that must not overlap. This is best used to finish a layout
#'   that has already converged: see the examples.
#' @param node_size Radius of the nodes, in the units of the layout, used when
#'   `prevent_overlap = TRUE`. Either a single number, a numeric vector with
#'   one value per node, or the name of a node column.
#' @param weights Edge weights; larger weights give a stronger attraction.
#'   Either `NULL` (the default: use the `weight` edge column if there is one,
#'   and no weights otherwise), `NA` (ignore the weights even if there is a
#'   `weight` column), the name of an edge column, or a numeric vector with
#'   one value per edge. Weights must be non-negative.
#' @param edge_weight_influence Exponent applied to the edge weights: 0 means
#'   that the weights are ignored, 1 that the attraction is proportional to
#'   the weights.
#' @param normalize_weights If `TRUE`, rescale the edge weights to the range
#'   0 to 1 (before applying `edge_weight_influence`).
#' @param invert_weights If `TRUE`, use the inverse of the edge weights, for
#'   weights that measure a distance rather than a strength.
#' @param init Initial positions of the nodes. Either `NULL` (random
#'   positions), a matrix or data frame with one row per node and two columns,
#'   or the names of two node columns, e.g. `c("x", "y")` to continue from a
#'   previous layout.
#' @param fixed Nodes that should stay where `init` puts them. Either `NULL`
#'   (all the nodes can move), a logical vector with one value per node, or
#'   the name of a logical node column.
#' @param jitter_tolerance How much swinging is tolerated. Lower values give
#'   a slower but more precise convergence.
#' @param barnes_hut Whether to approximate the repulsion with the Barnes-Hut
#'   algorithm, which lowers the cost of an iteration from O(n^2) to
#'   O(n log n). `NULL` (the default) enables it for graphs with at least 1000
#'   nodes, as Gephi does.
#' @param theta Precision of the Barnes-Hut approximation: smaller values are
#'   more precise and slower.
#' @param threads Number of threads to use for the repulsion. This has no
#'   effect if the package was compiled without OpenMP support (the default
#'   toolchain on macOS), in which case a single thread is used.
#'
#' @return
#' `forceatlas2()` returns `graph`, of the same class as its input, with the
#' two coordinate columns added to its nodes (replacing existing columns with
#' the same names).
#'
#' `layout_forceatlas2()` returns a numeric matrix with one row per node and
#' two columns.
#'
#' @references
#' Jacomy M, Venturini T, Heymann S, Bastian M (2014). ForceAtlas2, a
#' Continuous Graph Layout Algorithm for Handy Network Visualization Designed
#' for the Gephi Software. *PLoS ONE* 9(6): e98679.
#' \doi{10.1371/journal.pone.0098679}
#'
#' @examples
#' g <- igraph::sample_islands(4, 15, 0.5, 2)
#'
#' # A matrix of coordinates, as with igraph's own layouts
#' set.seed(1)
#' xy <- layout_forceatlas2(g)
#' head(xy)
#' plot(g, layout = xy, vertex.size = 6, vertex.label = NA)
#'
#' # Tighter clusters
#' xy <- layout_forceatlas2(g, linlog = TRUE, gravity = 0.1)
#'
#' if (requireNamespace("tidygraph", quietly = TRUE)) {
#'   # Coordinates added to the node table of a tidy graph
#'   tg <- tidygraph::as_tbl_graph(g)
#'   tg <- forceatlas2(tg)
#'   tg
#'
#'   # Refine an existing layout: start from the stored coordinates and
#'   # prevent the nodes from overlapping
#'   tg <- forceatlas2(tg, init = c("x", "y"), iterations = 200,
#'                     prevent_overlap = TRUE, node_size = 1)
#' }
#' @export
forceatlas2 <- function(graph, iterations = 1000, scaling_ratio = NULL,
                        gravity = 1, strong_gravity = FALSE, linlog = FALSE,
                        dissuade_hubs = FALSE, prevent_overlap = FALSE,
                        node_size = 1, weights = NULL,
                        edge_weight_influence = 1, normalize_weights = FALSE,
                        invert_weights = FALSE, init = NULL, fixed = NULL,
                        jitter_tolerance = 1, barnes_hut = NULL, theta = 1.2,
                        threads = 1, coords = c("x", "y")) {
  if (!is.character(coords) || length(coords) != 2L || anyNA(coords) ||
      any(!nzchar(coords)) || coords[1L] == coords[2L]) {
    stop("`coords` must be two different column names.", call. = FALSE)
  }
  xy <- layout_forceatlas2(
    graph, iterations = iterations, scaling_ratio = scaling_ratio,
    gravity = gravity, strong_gravity = strong_gravity, linlog = linlog,
    dissuade_hubs = dissuade_hubs, prevent_overlap = prevent_overlap,
    node_size = node_size, weights = weights,
    edge_weight_influence = edge_weight_influence,
    normalize_weights = normalize_weights, invert_weights = invert_weights,
    init = init, fixed = fixed, jitter_tolerance = jitter_tolerance,
    barnes_hut = barnes_hut, theta = theta, threads = threads
  )
  out <- graph
  igraph::vertex_attr(out, coords[1L]) <- xy[, 1L]
  igraph::vertex_attr(out, coords[2L]) <- xy[, 2L]
  # A tbl_graph must stay a tbl_graph, with the same active table and groups.
  # `vertex_attr<-` normally keeps the class and attributes of the object;
  # this makes sure of it whatever the version of igraph.
  for (a in setdiff(names(attributes(graph)), names(attributes(out)))) {
    attr(out, a) <- attr(graph, a)
  }
  if (!identical(class(out), class(graph))) class(out) <- class(graph)
  out
}

#' @param coords Names of the two node columns in which `forceatlas2()`
#'   stores the coordinates.
#' @rdname forceatlas2
#' @export
layout_forceatlas2 <- function(graph, iterations = 1000, scaling_ratio = NULL,
                               gravity = 1, strong_gravity = FALSE,
                               linlog = FALSE, dissuade_hubs = FALSE,
                               prevent_overlap = FALSE, node_size = 1,
                               weights = NULL, edge_weight_influence = 1,
                               normalize_weights = FALSE,
                               invert_weights = FALSE, init = NULL,
                               fixed = NULL, jitter_tolerance = 1,
                               barnes_hut = NULL, theta = 1.2, threads = 1) {
  if (!igraph::is_igraph(graph)) {
    stop("`graph` must be a tbl_graph or an igraph object, not an object of ",
         "class <", class(graph)[1L], ">.", call. = FALSE)
  }
  n <- igraph::vcount(graph)
  m <- igraph::ecount(graph)

  iterations <- check_number(iterations, "iterations", min = 0, whole = TRUE)
  gravity <- check_number(gravity, "gravity", min = 0)
  edge_weight_influence <- check_number(edge_weight_influence,
                                        "edge_weight_influence", min = 0)
  jitter_tolerance <- check_number(jitter_tolerance, "jitter_tolerance",
                                   min = 0, strict = TRUE)
  theta <- check_number(theta, "theta", min = 0)
  threads <- check_number(threads, "threads", min = 1, whole = TRUE)
  strong_gravity <- check_flag(strong_gravity, "strong_gravity")
  linlog <- check_flag(linlog, "linlog")
  dissuade_hubs <- check_flag(dissuade_hubs, "dissuade_hubs")
  prevent_overlap <- check_flag(prevent_overlap, "prevent_overlap")
  normalize_weights <- check_flag(normalize_weights, "normalize_weights")
  invert_weights <- check_flag(invert_weights, "invert_weights")

  if (is.null(scaling_ratio)) {
    scaling_ratio <- if (n >= 100) 2 else 10
  } else {
    scaling_ratio <- check_number(scaling_ratio, "scaling_ratio", min = 0,
                                  strict = TRUE)
  }
  if (is.null(barnes_hut)) {
    barnes_hut <- n >= 1000
  } else {
    barnes_hut <- check_flag(barnes_hut, "barnes_hut")
  }
  if (threads > 1 && !.Call(C_has_openmp)) threads <- 1

  w <- edge_weights(graph, weights, m)
  if (invert_weights) w <- ifelse(w == 0, 0, 1 / w)
  if (normalize_weights && m > 0) {
    rng <- range(w)
    w <- if (rng[1L] < rng[2L]) (w - rng[1L]) / (rng[2L] - rng[1L]) else rep(1, m)
  }
  w <- if (edge_weight_influence == 0) rep(1, m) else w^edge_weight_influence
  if (any(!is.finite(w))) {
    stop("The edge weights are too large or too small to be used.",
         call. = FALSE)
  }

  size <- node_values(graph, node_size, n, "node_size")
  if (any(size < 0)) stop("`node_size` must be non-negative.", call. = FALSE)

  fixed <- node_fixed(graph, fixed, n)
  pos <- initial_positions(graph, init, n)
  if (is.null(init) && any(fixed)) {
    stop("`fixed` nodes need a position: please provide `init`.",
         call. = FALSE)
  }

  el <- igraph::as_edgelist(graph, names = FALSE)

  xy <- .Call(
    C_forceatlas2,
    as.integer(n),
    as.integer(el[, 1L] - 1),
    as.integer(el[, 2L] - 1),
    as.double(w),
    pos,
    as.double(size),
    fixed,
    as.integer(iterations),
    as.double(c(scaling_ratio, gravity, jitter_tolerance, theta)),
    c(strong_gravity, linlog, dissuade_hubs, prevent_overlap, barnes_hut),
    as.integer(threads)
  )
  dimnames(xy) <- NULL
  xy
}

# Helpers ---------------------------------------------------------------------

check_number <- function(x, name, min = -Inf, strict = FALSE, whole = FALSE) {
  ok <- is.numeric(x) && length(x) == 1L && is.finite(x) &&
    (if (strict) x > min else x >= min) && (!whole || x == round(x))
  if (!ok) {
    stop("`", name, "` must be a single ", if (whole) "whole ", "number ",
         if (strict) "greater than " else "greater than or equal to ", min,
         ".", call. = FALSE)
  }
  x
}

check_flag <- function(x, name) {
  if (!is.logical(x) || length(x) != 1L || is.na(x)) {
    stop("`", name, "` must be TRUE or FALSE.", call. = FALSE)
  }
  x
}

edge_weights <- function(graph, weights, m) {
  if (is.null(weights)) {
    if (!"weight" %in% igraph::edge_attr_names(graph)) return(rep(1, m))
    w <- igraph::edge_attr(graph, "weight")
    what <- "The `weight` edge column"
  } else if (length(weights) == 1L && is.na(weights)) {
    return(rep(1, m))
  } else if (is.character(weights)) {
    if (length(weights) != 1L || !weights %in% igraph::edge_attr_names(graph)) {
      stop("`weights` must be the name of an existing edge column.",
           call. = FALSE)
    }
    w <- igraph::edge_attr(graph, weights)
    what <- paste0("The `", weights, "` edge column")
  } else {
    w <- weights
    what <- "`weights`"
    if (length(w) != m) {
      stop("`weights` must have one value per edge (", m, "), not ",
           length(w), ".", call. = FALSE)
    }
  }
  if (!is.numeric(w)) stop(what, " must be numeric.", call. = FALSE)
  if (any(!is.finite(w)) || any(w < 0)) {
    stop(what, " must contain only finite, non-negative values.",
         call. = FALSE)
  }
  as.double(w)
}

node_values <- function(graph, value, n, name) {
  if (is.character(value)) {
    if (length(value) != 1L || !value %in% igraph::vertex_attr_names(graph)) {
      stop("`", name, "` must be the name of an existing node column.",
           call. = FALSE)
    }
    value <- igraph::vertex_attr(graph, value)
  } else if (length(value) == 1L) {
    value <- rep(value, n)
  }
  if (!is.numeric(value) || length(value) != n || any(!is.finite(value))) {
    stop("`", name, "` must be a single number, one finite number per node (",
         n, "), or the name of a numeric node column.", call. = FALSE)
  }
  as.double(value)
}

node_fixed <- function(graph, fixed, n) {
  if (is.null(fixed)) return(logical(n))
  if (is.character(fixed)) {
    if (length(fixed) != 1L || !fixed %in% igraph::vertex_attr_names(graph)) {
      stop("`fixed` must be the name of an existing node column.",
           call. = FALSE)
    }
    fixed <- igraph::vertex_attr(graph, fixed)
  }
  if (!is.logical(fixed) || length(fixed) != n || anyNA(fixed)) {
    stop("`fixed` must be a logical vector with one value per node (", n,
         ") and no missing value, or the name of a logical node column.",
         call. = FALSE)
  }
  fixed
}

initial_positions <- function(graph, init, n) {
  if (is.null(init)) {
    # Spread the nodes over an area that grows with the size of the graph,
    # which is roughly what the layout converges to.
    r <- 10 * sqrt(n)
    return(matrix(stats::runif(2 * n, -r, r), ncol = 2L))
  }
  if (is.character(init)) {
    if (length(init) != 2L || !all(init %in% igraph::vertex_attr_names(graph))) {
      stop("`init` must name two existing node columns.", call. = FALSE)
    }
    pos <- cbind(igraph::vertex_attr(graph, init[1L]),
                 igraph::vertex_attr(graph, init[2L]))
  } else {
    pos <- as.matrix(init)
  }
  if (!is.numeric(pos) || length(dim(pos)) != 2L || nrow(pos) != n ||
      ncol(pos) != 2L) {
    stop("`init` must provide two numeric coordinates for each of the ", n,
         " nodes.", call. = FALSE)
  }
  if (any(!is.finite(pos))) {
    stop("`init` must not contain missing or infinite values.", call. = FALSE)
  }
  matrix(as.double(pos), ncol = 2L)
}
