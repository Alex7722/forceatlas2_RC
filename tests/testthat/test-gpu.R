# These tests need an OpenCL device. Any device will do, including a CPU
# driver such as PoCL, since they check the results and not the speed.
gpu_test_device <- function(double = TRUE) {
  devices <- gpu_devices()
  ok <- if (double) which(devices$double) else seq_len(nrow(devices))
  if (length(ok) == 0L) testthat::skip("no suitable OpenCL device")
  ok[1L]
}

test_that("gpu_devices() returns a data frame", {
  d <- gpu_devices()
  expect_s3_class(d, "data.frame")
  expect_named(d, c("name", "vendor", "platform", "version", "type",
                    "memory_gb", "double"))
  expect_true(all(d$type %in% c("GPU", "other")))
})

test_that("a missing or invalid device gives a clear error", {
  g <- test_graph(20)
  d <- gpu_devices()
  if (nrow(d) == 0L) {
    expect_error(layout_forceatlas2_gpu(g), "No OpenCL device")
  } else {
    expect_error(layout_forceatlas2_gpu(g, device = nrow(d) + 1), "device")
    expect_error(layout_forceatlas2_gpu(g, device = 0), "device")
    expect_error(layout_forceatlas2_gpu(g, device = "a"), "device")
  }
  if (nrow(d) > 0L && !any(d$type == "GPU")) {
    expect_error(layout_forceatlas2_gpu(g), "No GPU")
  }
  expect_error(layout_forceatlas2_gpu(g, precision = "half"), "arg")
})

test_that("in double precision the GPU gives the same layout as the CPU", {
  dev <- gpu_test_device()
  g <- test_graph(400, directed = TRUE)
  pos <- start_positions(400)
  set.seed(3)
  w <- stats::runif(igraph::ecount(g), 0.2, 3)
  size <- stats::runif(400, 0.5, 4)

  compare <- function(...) {
    for (bh in c(FALSE, TRUE)) {
      expect_equal(
        layout_forceatlas2_gpu(g, 10, init = pos, barnes_hut = bh, device = dev,
                               precision = "double", ...),
        layout_forceatlas2(g, 10, init = pos, barnes_hut = bh, ...),
        tolerance = 1e-9
      )
    }
  }
  compare()
  compare(theta = 0.5, scaling_ratio = 5, gravity = 3)
  compare(linlog = TRUE, dissuade_hubs = TRUE, strong_gravity = TRUE)
  compare(weights = w, edge_weight_influence = 0.5)
  compare(prevent_overlap = TRUE, node_size = size)
  compare(fixed = rep(c(TRUE, FALSE), 200))
})

test_that("in single precision the GPU stays close to the CPU", {
  dev <- gpu_test_device(double = FALSE)
  g <- test_graph(400)
  pos <- start_positions(400)
  for (bh in c(FALSE, TRUE)) {
    cpu <- layout_forceatlas2(g, 2, init = pos, barnes_hut = bh)
    gpu <- layout_forceatlas2_gpu(g, 2, init = pos, barnes_hut = bh,
                                  device = dev)
    expect_lt(max(abs(gpu - cpu)) / diff(range(cpu)), 1e-3)
    expect_false(identical(gpu, cpu))
  }
  # and gives a layout of the same kind after many iterations
  g <- igraph::disjoint_union(igraph::make_full_graph(10), igraph::make_full_graph(10))
  g <- igraph::add_edges(g, c(1, 11))
  set.seed(1)
  xy <- layout_forceatlas2_gpu(g, 500, device = dev)
  d <- as.matrix(stats::dist(xy))
  expect_true(all(is.finite(xy)))
  expect_lt(max(d[1:10, 1:10]), min(d[2:10, 12:20]))
})

test_that("the GPU layout handles degenerate graphs", {
  dev <- gpu_test_device(double = FALSE)
  expect_identical(layout_forceatlas2_gpu(igraph::make_empty_graph(0), device = dev),
                   matrix(numeric(), ncol = 2))
  set.seed(1)
  expect_true(all(is.finite(
    layout_forceatlas2_gpu(igraph::make_empty_graph(1), 10, device = dev)
  )))
  g <- test_graph(50)
  zero <- matrix(0, 50, 2)
  pos <- start_positions(50)
  pos[1:20, ] <- 1
  for (bh in c(FALSE, TRUE)) {
    expect_identical(
      layout_forceatlas2_gpu(g, 50, init = zero, barnes_hut = bh, device = dev),
      zero
    )
    expect_true(all(is.finite(
      layout_forceatlas2_gpu(g, 50, init = pos, barnes_hut = bh, device = dev)
    )))
  }
})

test_that("forceatlas2_gpu() adds the coordinates to the graph", {
  dev <- gpu_test_device(double = FALSE)
  g <- test_graph(30)
  set.seed(4)
  out <- forceatlas2_gpu(g, 20, device = dev, coords = c("gx", "gy"))
  set.seed(4)
  xy <- layout_forceatlas2_gpu(g, 20, device = dev)
  expect_true(igraph::is_igraph(out))
  expect_identical(igraph::V(out)$gx, xy[, 1])
  expect_identical(igraph::V(out)$gy, xy[, 2])

  skip_if_not_installed("tidygraph")
  tg <- forceatlas2_gpu(tidygraph::as_tbl_graph(g), 20, device = dev)
  expect_s3_class(tg, "tbl_graph")
  expect_named(as.data.frame(tidygraph::activate(tg, "nodes")), c("x", "y"))
})
