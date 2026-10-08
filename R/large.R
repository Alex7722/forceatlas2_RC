#' ForceAtlas2 layout of very large graphs, entirely on a graphics card
#'
#' @description
#' `forceatlas2_large()` and `layout_forceatlas2_large()` compute a
#' ForceAtlas2 layout without leaving the graphics card (GPU): the graph is
#' sent to the card once, every step of every iteration runs there, and the
#' positions are fetched at the end. They are meant for graphs of hundreds of
#' thousands to millions of nodes, and need a GPU (see [gpu_devices()]).
#'
#' `forceatlas2_large()` returns the graph with the coordinates added to its
#' nodes, like [forceatlas2()], and `layout_forceatlas2_large()` returns the
#' matrix of coordinates, like [layout_forceatlas2()].
#'
#' @details
#' The forces, the adaptive speed and the settings are those of
#' [forceatlas2()]. What differs is how the repulsion is approximated.
#' [forceatlas2()] and [forceatlas2_gpu()] build, on the CPU, the tree of
#' regions that Gephi uses. Here the tree is built on the GPU at every
#' iteration with a method made for it: the nodes are sorted along a
#' space-filling curve (Morton order) and a binary tree is derived from the
#' sorted nodes, all in parallel. The repulsion is then the usual Barnes-Hut
#' one: a region acts as a single node when it is far enough, that is when
#' `distance * theta > size`. The size of a region is here the diagonal of
#' its bounding box times the square root of two, chosen so that a given
#' `theta` is about as precise as with the other functions.
#'
#' As a consequence:
#'
#' * the layouts have the same properties as those of the other functions but
#'   are not identical to them, since the approximation is different (with
#'   `theta = 0` nothing is approximated, and the result is the exact one);
#' * the repulsion is always approximated: there is no `barnes_hut` argument;
#' * for a given graph, starting point and device, the result is reproducible.
#'
#' With `precision = "single"` (the default), the positions are also stored in
#' single precision, i.e. with about seven significant digits. This is fine
#' for drawing a graph. Use `precision = "double"` if you need more, knowing
#' that consumer cards are much slower in double precision.
#'
#' The memory used on the card is about 180 bytes per node and 16 bytes per
#' edge in single precision (twice as much for most of it in double
#' precision): roughly 2 GB for ten million nodes and fifty million edges.
#'
#' Graphs with fewer than two nodes are handled on the CPU. For small graphs
#' (a few thousand nodes) these functions work but are slower than
#' [forceatlas2()], because of the fixed cost of driving the card.
#'
#' With `options(forceatlas2r.timings = TRUE)`, the matrix returned by
#' `layout_forceatlas2_large()` has a `"timings"` attribute giving, in
#' seconds, the total time and the time spent in each phase of the
#' iterations (measuring them slows the layout down a little).
#'
#' @inheritParams forceatlas2
#' @inheritParams forceatlas2_gpu
#' @param theta Precision of the approximation of the repulsion: smaller
#'   values are more precise and slower, and 0 computes the exact repulsion
#'   (which is only feasible for small graphs).
#'
#' @return
#' `forceatlas2_large()` returns `graph` with the two coordinate columns added
#' to its nodes, and `layout_forceatlas2_large()` a numeric matrix with one
#' row per node and two columns.
#'
#' @references
#' Karras T (2012). Maximizing Parallelism in the Construction of BVHs,
#' Octrees, and k-d Trees. *High Performance Graphics*, 33-37.
#'
#' @seealso [forceatlas2()] for the description of the algorithm and of its
#'   settings; [forceatlas2_gpu()], which uses the GPU for the repulsion only
#'   and gives the same layouts as the CPU.
#'
#' @examples
#' # Only run when a GPU is available
#' if (any(gpu_devices()$type == "GPU")) {
#'   g <- igraph::sample_pa(20000, m = 2, directed = FALSE)
#'   xy <- layout_forceatlas2_large(g, iterations = 100)
#'   head(xy)
#' }
#' @export
forceatlas2_large <- function(graph, iterations = 1000, scaling_ratio = NULL,
                              gravity = 1, strong_gravity = FALSE,
                              linlog = FALSE, dissuade_hubs = FALSE,
                              prevent_overlap = FALSE, node_size = 1,
                              weights = NULL, edge_weight_influence = 1,
                              normalize_weights = FALSE,
                              invert_weights = FALSE, init = NULL,
                              fixed = NULL, jitter_tolerance = 1, theta = 1.2,
                              device = NULL,
                              precision = c("single", "double"),
                              coords = c("x", "y")) {
  coords <- check_coords(coords)
  xy <- layout_forceatlas2_large(
    graph, iterations = iterations, scaling_ratio = scaling_ratio,
    gravity = gravity, strong_gravity = strong_gravity, linlog = linlog,
    dissuade_hubs = dissuade_hubs, prevent_overlap = prevent_overlap,
    node_size = node_size, weights = weights,
    edge_weight_influence = edge_weight_influence,
    normalize_weights = normalize_weights, invert_weights = invert_weights,
    init = init, fixed = fixed, jitter_tolerance = jitter_tolerance,
    theta = theta, device = device, precision = precision
  )
  set_coords(graph, xy, coords)
}

#' @rdname forceatlas2_large
#' @export
layout_forceatlas2_large <- function(graph, iterations = 1000,
                                     scaling_ratio = NULL, gravity = 1,
                                     strong_gravity = FALSE, linlog = FALSE,
                                     dissuade_hubs = FALSE,
                                     prevent_overlap = FALSE, node_size = 1,
                                     weights = NULL, edge_weight_influence = 1,
                                     normalize_weights = FALSE,
                                     invert_weights = FALSE, init = NULL,
                                     fixed = NULL, jitter_tolerance = 1,
                                     theta = 1.2, device = NULL,
                                     precision = c("single", "double")) {
  precision <- match.arg(precision)
  fa2_layout(
    graph, iterations = iterations, scaling_ratio = scaling_ratio,
    gravity = gravity, strong_gravity = strong_gravity, linlog = linlog,
    dissuade_hubs = dissuade_hubs, prevent_overlap = prevent_overlap,
    node_size = node_size, weights = weights,
    edge_weight_influence = edge_weight_influence,
    normalize_weights = normalize_weights, invert_weights = invert_weights,
    init = init, fixed = fixed, jitter_tolerance = jitter_tolerance,
    barnes_hut = TRUE, theta = theta,
    gpu = list(device = device, precision = precision, full = TRUE)
  )
}

# Source of the OpenCL kernels of the all-GPU layout.
pipeline_source <- function() {
  path <- system.file("opencl", "pipeline.cl", package = "forceatlas2r")
  if (!nzchar(path)) {
    stop("The OpenCL kernels of the package were not found: please reinstall ",
         "forceatlas2r.", call. = FALSE)
  }
  paste(readLines(path, warn = FALSE), collapse = "\n")
}
