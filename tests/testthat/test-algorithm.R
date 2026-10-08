test_that("the C code agrees with the R transcription of Gephi's algorithm", {
  g <- test_graph(40)
  pos <- start_positions(40)
  # The comparison is made after a few iterations only: started from random
  # positions, the system is chaotic at first and amplifies rounding errors.
  expect_equal(
    layout_forceatlas2(g, iterations = 10, init = pos, barnes_hut = FALSE),
    reference_fa2(g, pos, 10, scaling = 10),
    tolerance = 1e-9
  )
  # After many iterations both still give the same picture
  expect_equal(
    layout_forceatlas2(g, iterations = 100, init = pos, barnes_hut = FALSE),
    reference_fa2(g, pos, 100, scaling = 10),
    tolerance = 1e-3
  )
})

test_that("every option agrees with the reference implementation", {
  g <- test_graph(40, directed = TRUE)
  pos <- start_positions(40)
  set.seed(3)
  w <- stats::runif(igraph::ecount(g), 0.2, 3)
  size <- stats::runif(40, 0.5, 4)

  expect_equal(
    layout_forceatlas2(g, 10, init = pos, strong_gravity = TRUE, gravity = 0.3),
    reference_fa2(g, pos, 10, scaling = 10, strong = TRUE, gravity = 0.3),
    tolerance = 1e-8
  )
  expect_equal(
    layout_forceatlas2(g, 10, init = pos, linlog = TRUE, scaling_ratio = 3),
    reference_fa2(g, pos, 10, scaling = 3, linlog = TRUE),
    tolerance = 1e-8
  )
  expect_equal(
    layout_forceatlas2(g, 10, init = pos, dissuade_hubs = TRUE, gravity = 0),
    reference_fa2(g, pos, 10, scaling = 10, outbound = TRUE, gravity = 0),
    tolerance = 1e-8
  )
  expect_equal(
    layout_forceatlas2(g, 10, init = pos, weights = w,
                       edge_weight_influence = 0.5, jitter_tolerance = 0.2),
    reference_fa2(g, pos, 10, scaling = 10, w = sqrt(w), jt_user = 0.2),
    tolerance = 1e-8
  )
  expect_equal(
    layout_forceatlas2(g, 10, init = pos, prevent_overlap = TRUE,
                       node_size = size),
    reference_fa2(g, pos, 10, scaling = 10, adjust = TRUE, size = size),
    tolerance = 1e-8
  )
  expect_equal(
    layout_forceatlas2(g, 10, init = pos, prevent_overlap = TRUE,
                       node_size = size, linlog = TRUE, dissuade_hubs = TRUE,
                       weights = w),
    reference_fa2(g, pos, 10, scaling = 10, adjust = TRUE, size = size,
                  linlog = TRUE, outbound = TRUE, w = w),
    tolerance = 1e-8
  )
})

test_that("Barnes-Hut converges to the exact computation", {
  g <- test_graph(300)
  pos <- start_positions(300)
  run <- function(iterations, ...) {
    layout_forceatlas2(g, iterations, init = pos, ...)
  }
  # theta = 0 never approximates anything
  expect_equal(run(5, barnes_hut = TRUE, theta = 0), run(5, barnes_hut = FALSE),
               tolerance = 1e-9)
  expect_equal(
    run(3, barnes_hut = TRUE, theta = 0, prevent_overlap = TRUE, node_size = 2),
    run(3, barnes_hut = FALSE, prevent_overlap = TRUE, node_size = 2),
    tolerance = 1e-9
  )
  # The error of one iteration decreases with theta
  exact <- run(1, barnes_hut = FALSE)
  err <- function(theta) {
    max(abs(run(1, barnes_hut = TRUE, theta = theta) - exact)) / diff(range(exact))
  }
  expect_lt(err(0.1), 1e-4)
  expect_lt(err(0.1), err(0.5))
  expect_lt(err(0.5), err(1.2))
  expect_lt(err(1.2), 0.02)
})

test_that("the number of threads does not change the result", {
  g <- test_graph(500)
  pos <- start_positions(500)
  for (bh in c(FALSE, TRUE)) {
    expect_identical(
      layout_forceatlas2(g, 40, init = pos, barnes_hut = bh, threads = 1),
      layout_forceatlas2(g, 40, init = pos, barnes_hut = bh, threads = 2)
    )
  }
  expect_identical(
    layout_forceatlas2(g, 40, init = pos, barnes_hut = FALSE, threads = 1,
                       prevent_overlap = TRUE),
    layout_forceatlas2(g, 40, init = pos, barnes_hut = FALSE, threads = 2,
                       prevent_overlap = TRUE)
  )
})

test_that("layouts are reproducible with set.seed()", {
  g <- test_graph(50)
  set.seed(10)
  a <- layout_forceatlas2(g, 50)
  set.seed(10)
  b <- layout_forceatlas2(g, 50)
  expect_identical(a, b)
  expect_false(identical(a, layout_forceatlas2(g, 50)))
})

test_that("the layout makes sense", {
  # Two cliques joined by one edge end up as two separate groups
  g <- igraph::disjoint_union(igraph::make_full_graph(10), igraph::make_full_graph(10))
  g <- igraph::add_edges(g, c(1, 11))
  set.seed(1)
  xy <- layout_forceatlas2(g, 500)
  d <- as.matrix(stats::dist(xy))
  expect_true(all(is.finite(xy)))
  expect_lt(max(d[1:10, 1:10]), min(d[2:10, 12:20]))

  # A converged layout barely moves any more
  again <- layout_forceatlas2(g, 50, init = xy)
  expect_lt(max(abs(again - xy)) / diff(range(xy)), 0.01)

  # Stronger repulsion gives a larger layout, stronger gravity a smaller one
  spread <- function(...) {
    set.seed(1)
    mean(stats::dist(layout_forceatlas2(g, 500, ...)))
  }
  expect_gt(spread(scaling_ratio = 50), spread(scaling_ratio = 5))
  expect_lt(spread(gravity = 20), spread(gravity = 1))
  expect_lt(spread(strong_gravity = TRUE), spread())
})

test_that("prevent_overlap separates the nodes", {
  g <- igraph::make_full_graph(30)
  set.seed(2)
  xy <- layout_forceatlas2(g, 500, scaling_ratio = 1)
  expect_lt(min(stats::dist(xy)), 12)
  xy2 <- layout_forceatlas2(g, 500, scaling_ratio = 1, init = xy,
                            prevent_overlap = TRUE, node_size = 6)
  expect_gte(min(stats::dist(xy2)), 12)
})

test_that("fixed nodes do not move", {
  g <- test_graph(30)
  pos <- start_positions(30)
  fixed <- rep(c(TRUE, FALSE), c(5, 25))
  xy <- layout_forceatlas2(g, 100, init = pos, fixed = fixed)
  expect_identical(xy[fixed, ], pos[fixed, ])
  expect_true(all(xy[!fixed, ] != pos[!fixed, ]))
  expect_equal(layout_forceatlas2(g, 10, init = pos, fixed = fixed),
               reference_fa2(g, pos, 10, scaling = 10, fixed = fixed),
               tolerance = 1e-9)
  expect_error(layout_forceatlas2(g, fixed = fixed), "init")
})

test_that("degenerate graphs are handled", {
  expect_identical(layout_forceatlas2(igraph::make_empty_graph(0)),
                   matrix(numeric(), ncol = 2))
  set.seed(1)
  one <- layout_forceatlas2(igraph::make_empty_graph(1))
  expect_identical(dim(one), c(1L, 2L))
  expect_true(all(is.finite(one)))

  # No edges, isolated nodes, self-loops and multiple edges
  expect_true(all(is.finite(layout_forceatlas2(igraph::make_empty_graph(20), 2000))))
  g <- igraph::make_graph(c(1, 1, 1, 2, 1, 2, 2, 3), n = 5, directed = FALSE)
  for (bh in c(FALSE, TRUE)) {
    expect_true(all(is.finite(layout_forceatlas2(g, 200, barnes_hut = bh))))
  }

  # All the nodes start at the same place: nothing can move, nothing breaks
  g <- test_graph(50)
  zero <- matrix(0, 50, 2)
  for (bh in c(FALSE, TRUE)) {
    expect_identical(layout_forceatlas2(g, 2000, init = zero, barnes_hut = bh), zero)
  }
  # Coincident nodes in an otherwise normal layout
  pos <- start_positions(50)
  pos[1:20, ] <- 1
  expect_equal(
    layout_forceatlas2(g, 5, init = pos, barnes_hut = TRUE, theta = 0),
    layout_forceatlas2(g, 5, init = pos, barnes_hut = FALSE),
    tolerance = 1e-9
  )
  expect_true(all(is.finite(layout_forceatlas2(g, 30, init = pos, barnes_hut = TRUE))))

  expect_identical(layout_forceatlas2(g, 0, init = pos), pos)
})
