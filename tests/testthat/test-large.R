# Tests of the all-GPU layout. Like those of test-gpu.R, they need an OpenCL
# device but not a GPU: they check the results, not the speed.
large_test_device <- function(double = TRUE) {
  devices <- gpu_devices()
  ok <- if (double) which(devices$double) else seq_len(nrow(devices))
  if (length(ok) == 0L) testthat::skip("no suitable OpenCL device")
  ok[1L]
}

test_that("without approximation, the all-GPU layout is the exact one", {
  dev <- large_test_device()
  # More than 2^16 would be needed for equal Morton codes to be likely, so
  # some nodes are also put at the same place to exercise that case.
  g <- test_graph(700, directed = TRUE)
  pos <- start_positions(700)
  pos[1:30, ] <- pos[rep(1:3, 10), ]
  set.seed(3)
  w <- stats::runif(igraph::ecount(g), 0.2, 3)
  size <- stats::runif(700, 0.5, 4)

  compare <- function(...) {
    expect_equal(
      layout_forceatlas2_large(g, 5, init = pos, theta = 0, device = dev,
                               precision = "double", ...),
      layout_forceatlas2(g, 5, init = pos, barnes_hut = FALSE, ...),
      tolerance = 1e-8
    )
  }
  compare()
  compare(scaling_ratio = 5, gravity = 3)
  compare(linlog = TRUE, strong_gravity = TRUE)
  compare(dissuade_hubs = TRUE, gravity = 0)
  compare(weights = w, edge_weight_influence = 0.5, jitter_tolerance = 0.3)
  compare(prevent_overlap = TRUE, node_size = size)
  compare(prevent_overlap = TRUE, node_size = size, linlog = TRUE,
          dissuade_hubs = TRUE, weights = w)
  compare(fixed = rep(c(TRUE, FALSE), 350))
})

test_that("the approximation gets better as theta decreases", {
  dev <- large_test_device()
  g <- test_graph(3000)
  pos <- start_positions(3000)
  exact <- layout_forceatlas2(g, 1, init = pos, barnes_hut = FALSE)
  err <- function(theta, precision = "double") {
    xy <- layout_forceatlas2_large(g, 1, init = pos, theta = theta,
                                   device = dev, precision = precision)
    max(abs(xy - exact)) / diff(range(exact))
  }
  expect_lt(err(0.1), 1e-4)
  expect_lt(err(0.1), err(0.5))
  expect_lt(err(0.5), err(1.2))
  expect_lt(err(1.2), 0.02)
  # Single precision adds little to the error of the approximation
  expect_lt(err(1.2, "single"), 0.02)
  expect_lt(err(0, "single"), 1e-3)
})

test_that("the all-GPU layout is reproducible", {
  dev <- large_test_device(double = FALSE)
  g <- test_graph(5000)
  pos <- start_positions(5000)
  a <- layout_forceatlas2_large(g, 30, init = pos, device = dev)
  b <- layout_forceatlas2_large(g, 30, init = pos, device = dev)
  expect_identical(a, b)
  set.seed(8)
  a <- layout_forceatlas2_large(g, 5, device = dev)
  set.seed(8)
  b <- layout_forceatlas2_large(g, 5, device = dev)
  expect_identical(a, b)
})

test_that("the all-GPU layout makes sense", {
  dev <- large_test_device(double = FALSE)
  g <- igraph::disjoint_union(igraph::make_full_graph(10), igraph::make_full_graph(10))
  g <- igraph::add_edges(g, c(1, 11))
  set.seed(1)
  xy <- layout_forceatlas2_large(g, 500, device = dev)
  d <- as.matrix(stats::dist(xy))
  expect_true(all(is.finite(xy)))
  expect_lt(max(d[1:10, 1:10]), min(d[2:10, 12:20]))

  # Same kind of layout as on the CPU
  g <- igraph::sample_islands(4, 100, 0.1, 2)
  el <- igraph::as_edgelist(g)
  pos <- start_positions(400)
  edge_length <- function(xy) {
    mean(sqrt(rowSums((xy[el[, 1], ] - xy[el[, 2], ])^2))) / stats::sd(xy)
  }
  cpu <- layout_forceatlas2(g, 300, init = pos)
  gpu <- layout_forceatlas2_large(g, 300, init = pos, device = dev)
  expect_equal(edge_length(gpu), edge_length(cpu), tolerance = 0.05)
  expect_equal(stats::sd(gpu), stats::sd(cpu), tolerance = 0.05)
})

test_that("the all-GPU layout handles degenerate graphs", {
  dev <- large_test_device(double = FALSE)
  # Fewer than two nodes: computed on the CPU
  expect_identical(layout_forceatlas2_large(igraph::make_empty_graph(0), device = dev),
                   matrix(numeric(), ncol = 2))
  set.seed(1)
  expect_identical(dim(layout_forceatlas2_large(igraph::make_empty_graph(1), 10, device = dev)),
                   c(1L, 2L))

  for (n in 2:12) {
    set.seed(n)
    xy <- layout_forceatlas2_large(igraph::make_ring(n), 50, device = dev)
    expect_true(all(is.finite(xy)))
    expect_gt(min(stats::dist(xy)), 0)
  }
  # No edges; self-loops and multiple edges
  set.seed(1)
  expect_true(all(is.finite(
    layout_forceatlas2_large(igraph::make_empty_graph(300), 100, device = dev)
  )))
  g <- igraph::make_graph(c(1, 1, 1, 2, 1, 2, 2, 3), n = 5, directed = FALSE)
  set.seed(1)
  expect_true(all(is.finite(layout_forceatlas2_large(g, 100, device = dev))))

  # All the nodes at the same place: nothing can move, nothing breaks
  g <- test_graph(500)
  zero <- matrix(0, 500, 2)
  expect_identical(layout_forceatlas2_large(g, 20, init = zero, device = dev), zero)
  # Nodes on a line, and many nodes at the same place
  line <- cbind(seq_len(500), 0)
  expect_true(all(is.finite(layout_forceatlas2_large(g, 20, init = line, device = dev))))
  pos <- start_positions(500)
  pos[1:300, ] <- 1
  expect_true(all(is.finite(layout_forceatlas2_large(g, 20, init = pos, device = dev))))

  expect_equal(layout_forceatlas2_large(g, 0, init = pos, device = dev), pos,
               tolerance = 1e-6)
})

test_that("forceatlas2_large() adds the coordinates to the graph", {
  dev <- large_test_device(double = FALSE)
  g <- test_graph(60)
  set.seed(4)
  out <- forceatlas2_large(g, 20, device = dev, coords = c("gx", "gy"))
  set.seed(4)
  xy <- layout_forceatlas2_large(g, 20, device = dev)
  expect_identical(igraph::V(out)$gx, xy[, 1])
  expect_identical(igraph::V(out)$gy, xy[, 2])
  expect_error(layout_forceatlas2_large(g, device = nrow(gpu_devices()) + 1), "device")
  expect_error(layout_forceatlas2_large(g, theta = -1, device = dev), "theta")

  skip_if_not_installed("tidygraph")
  tg <- forceatlas2_large(tidygraph::as_tbl_graph(g), 20, device = dev)
  expect_s3_class(tg, "tbl_graph")
  expect_named(as.data.frame(tidygraph::activate(tg, "nodes")), c("x", "y"))
})
