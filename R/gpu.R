#' ForceAtlas2 layout computed on a graphics card
#'
#' @description
#' `forceatlas2_gpu()` and `layout_forceatlas2_gpu()` are the counterparts of
#' [forceatlas2()] and [layout_forceatlas2()] that compute the repulsion
#' between the nodes, which is where nearly all the time goes, on a graphics
#' card (GPU). They take the same arguments and return the same results.
#'
#' `gpu_devices()` lists the devices that can be used.
#'
#' @details
#' The GPU is driven through OpenCL, which is installed with the driver of the
#' graphics card: nothing else needs to be installed, and the package does not
#' need OpenCL (or CUDA) to be built. NVIDIA cards are the target of this
#' backend; cards of other vendors that provide OpenCL should work as well.
#' Use `gpu_devices()` to check that your card is found.
#'
#' The algorithm is the same as on the CPU:
#'
#' * with `barnes_hut = FALSE`, the repulsion between every pair of nodes is
#'   computed on the GPU;
#' * with `barnes_hut = TRUE`, the Barnes-Hut tree is built on the CPU at
#'   every iteration and the GPU traverses it for all the nodes at once.
#'
#' By default the first is used below 10,000 nodes and the second above. The
#' attraction along the edges, the gravity and the displacement of the nodes
#' are cheap and stay on the CPU.
#'
#' With `precision = "single"` (the default) the GPU computes in single
#' precision, which is what consumer graphics cards are fast at. This is
#' accurate enough for a layout, but the result is not identical to the one of
#' [layout_forceatlas2()]: starting from the same positions, the two layouts
#' are very close after the first iterations and then drift apart, as two runs
#' with different starting positions would. With `precision = "double"` the
#' GPU performs the same operations as the CPU and the two layouts are the
#' same, at the price of a slower computation on most cards.
#'
#' A GPU only pays off for large graphs. For a few thousand nodes the CPU
#' version is as fast or faster, because each iteration has a fixed cost of
#' exchanging data with the card. The first GPU layout of a session also
#' takes a moment longer, while the driver compiles the code for the card.
#'
#' @inheritParams forceatlas2
#' @param barnes_hut Whether to approximate the repulsion with the Barnes-Hut
#'   algorithm. `NULL` (the default) enables it for graphs with at least
#'   10,000 nodes.
#' @param device The device to use, as its row number in `gpu_devices()`.
#'   `NULL` (the default) uses the first GPU, giving priority to NVIDIA cards.
#' @param precision `"single"` (the default) or `"double"`: the precision of
#'   the computations made on the GPU.
#'
#' @return
#' `forceatlas2_gpu()` returns `graph` with the two coordinate columns added
#' to its nodes, and `layout_forceatlas2_gpu()` a numeric matrix with one row
#' per node and two columns, like their CPU counterparts.
#'
#' `gpu_devices()` returns a data frame with one row per OpenCL device: its
#' `name`, `vendor`, `platform` (the driver), OpenCL `version`, `type`
#' (`"GPU"` or `"other"`), memory in gigabytes and whether it supports
#' `double` precision. It has no rows when no device is available.
#'
#' @seealso [forceatlas2()] for the description of the algorithm and of its
#'   settings.
#'
#' @examples
#' gpu_devices()
#'
#' # Only run when a GPU is available
#' if (any(gpu_devices()$type == "GPU")) {
#'   g <- igraph::sample_islands(4, 15, 0.5, 2)
#'   xy <- layout_forceatlas2_gpu(g, iterations = 100)
#'   head(xy)
#' }
#' @export
forceatlas2_gpu <- function(graph, iterations = 1000, scaling_ratio = NULL,
                            gravity = 1, strong_gravity = FALSE,
                            linlog = FALSE, dissuade_hubs = FALSE,
                            prevent_overlap = FALSE, node_size = 1,
                            weights = NULL, edge_weight_influence = 1,
                            normalize_weights = FALSE, invert_weights = FALSE,
                            init = NULL, fixed = NULL, jitter_tolerance = 1,
                            barnes_hut = NULL, theta = 1.2, device = NULL,
                            precision = c("single", "double"),
                            coords = c("x", "y")) {
  coords <- check_coords(coords)
  xy <- layout_forceatlas2_gpu(
    graph, iterations = iterations, scaling_ratio = scaling_ratio,
    gravity = gravity, strong_gravity = strong_gravity, linlog = linlog,
    dissuade_hubs = dissuade_hubs, prevent_overlap = prevent_overlap,
    node_size = node_size, weights = weights,
    edge_weight_influence = edge_weight_influence,
    normalize_weights = normalize_weights, invert_weights = invert_weights,
    init = init, fixed = fixed, jitter_tolerance = jitter_tolerance,
    barnes_hut = barnes_hut, theta = theta, device = device,
    precision = precision
  )
  set_coords(graph, xy, coords)
}

#' @rdname forceatlas2_gpu
#' @export
layout_forceatlas2_gpu <- function(graph, iterations = 1000,
                                   scaling_ratio = NULL, gravity = 1,
                                   strong_gravity = FALSE, linlog = FALSE,
                                   dissuade_hubs = FALSE,
                                   prevent_overlap = FALSE, node_size = 1,
                                   weights = NULL, edge_weight_influence = 1,
                                   normalize_weights = FALSE,
                                   invert_weights = FALSE, init = NULL,
                                   fixed = NULL, jitter_tolerance = 1,
                                   barnes_hut = NULL, theta = 1.2,
                                   device = NULL,
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
    barnes_hut = barnes_hut, theta = theta,
    gpu = list(device = device, precision = precision)
  )
}

#' @rdname forceatlas2_gpu
#' @export
gpu_devices <- function() {
  d <- .Call(C_gpu_devices)
  d <- lapply(d, function(x) if (is.character(x)) trimws(x) else x)
  as.data.frame(d, stringsAsFactors = FALSE, check.names = FALSE)
}

# Turn the `device` and `precision` arguments into the settings expected by
# the C code: the 0-based index of the device and whether to use doubles.
gpu_resolve <- function(device, precision) {
  devices <- gpu_devices()
  if (nrow(devices) == 0L) {
    stop("No OpenCL device was found on this computer. The GPU layout needs ",
         "a graphics card with an up-to-date driver (for NVIDIA cards, the ",
         "driver from nvidia.com). Use `layout_forceatlas2()` to compute the ",
         "layout on the CPU.", call. = FALSE)
  }
  if (is.null(device)) {
    gpus <- which(devices$type == "GPU")
    if (length(gpus) == 0L) {
      stop("No GPU was found among the OpenCL devices of this computer (see ",
           "`gpu_devices()`). Choose one of them with `device`, or use ",
           "`layout_forceatlas2()` to compute the layout on the CPU.",
           call. = FALSE)
    }
    nvidia <- gpus[grepl("nvidia", devices$vendor[gpus], ignore.case = TRUE)]
    device <- if (length(nvidia) > 0L) nvidia[1L] else gpus[1L]
  } else {
    ok <- is.numeric(device) && length(device) == 1L && !is.na(device) &&
      device == round(device) && device >= 1 && device <= nrow(devices)
    if (!ok) {
      stop("`device` must be the row number of a device in `gpu_devices()` ",
           "(between 1 and ", nrow(devices), ").", call. = FALSE)
    }
  }
  if (precision == "double" && !devices$double[device]) {
    stop("The device \"", devices$name[device], "\" does not support double ",
         "precision. Use `precision = \"single\"`.", call. = FALSE)
  }
  c(as.integer(device) - 1L, as.integer(precision == "double"))
}
