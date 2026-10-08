test_that("forceatlas2() adds the coordinates to a tbl_graph", {
  skip_if_not_installed("tidygraph")
  tg <- tidygraph::as_tbl_graph(test_graph(30))
  tg <- tidygraph::mutate(tg, name = letters[(seq_len(30) - 1) %% 26 + 1])

  set.seed(5)
  out <- forceatlas2(tg, iterations = 50)
  set.seed(5)
  xy <- layout_forceatlas2(tg, iterations = 50)

  expect_s3_class(out, "tbl_graph")
  nodes <- as.data.frame(tidygraph::activate(out, "nodes"))
  expect_named(nodes, c("name", "x", "y"))
  expect_identical(nodes$x, xy[, 1])
  expect_identical(nodes$y, xy[, 2])
  expect_identical(nodes$name, igraph::V(tg)$name)
  expect_identical(igraph::as_edgelist(out), igraph::as_edgelist(tg))
})

test_that("forceatlas2() keeps the active table and the grouping", {
  skip_if_not_installed("tidygraph")
  tg <- tidygraph::as_tbl_graph(test_graph(30))
  edges <- tidygraph::activate(tg, "edges")
  out <- forceatlas2(edges, iterations = 10)
  expect_identical(tidygraph::active(out), "edges")
  expect_true(all(c("x", "y") %in% igraph::vertex_attr_names(out)))

  grouped <- tidygraph::group_by(tidygraph::mutate(tg, grp = rep(1:3, 10)), grp)
  out <- forceatlas2(grouped, iterations = 10)
  expect_identical(class(out), class(grouped))
  expect_identical(tidygraph::group_vars(out), "grp")
})

test_that("forceatlas2() works in a tidygraph pipeline", {
  skip_if_not_installed("tidygraph")
  `%>%` <- tidygraph::`%>%`
  out <- tidygraph::as_tbl_graph(test_graph(30)) %>%
    tidygraph::mutate(deg = tidygraph::centrality_degree()) %>%
    forceatlas2(iterations = 20, node_size = "deg", prevent_overlap = TRUE) %>%
    tidygraph::filter(x > 0)
  expect_s3_class(out, "tbl_graph")
  expect_true(all(igraph::V(out)$x > 0))
})

test_that("forceatlas2() works on igraph objects and with other column names", {
  g <- test_graph(30)
  out <- forceatlas2(g, iterations = 10, coords = c("fa_x", "fa_y"))
  expect_true(igraph::is_igraph(out))
  expect_setequal(igraph::vertex_attr_names(out), c("fa_x", "fa_y"))
  expect_error(forceatlas2(g, coords = "x"), "coords")
  expect_error(forceatlas2(g, coords = c("x", "x")), "coords")
})

test_that("a layout can be continued from the stored coordinates", {
  g <- forceatlas2(test_graph(30), iterations = 20)
  pos <- cbind(igraph::V(g)$x, igraph::V(g)$y)
  expect_identical(layout_forceatlas2(g, 10, init = c("x", "y")),
                   layout_forceatlas2(g, 10, init = pos))
  expect_identical(layout_forceatlas2(g, 10, init = as.data.frame(pos)),
                   layout_forceatlas2(g, 10, init = pos))
  igraph::V(g)$pin <- rep(c(TRUE, FALSE), 15)
  out <- forceatlas2(g, 10, init = c("x", "y"), fixed = "pin")
  expect_identical(igraph::V(out)$x[igraph::V(g)$pin], pos[igraph::V(g)$pin, 1])
  expect_error(layout_forceatlas2(g, init = c("x", "nope")), "init")
  expect_error(layout_forceatlas2(g, init = pos[-1, ]), "init")
  pos[3, 1] <- NA
  expect_error(layout_forceatlas2(g, init = pos), "init")
})

test_that("edge weights are found and transformed as documented", {
  g <- test_graph(30)
  m <- igraph::ecount(g)
  pos <- start_positions(30)
  set.seed(9)
  w <- stats::runif(m, 1, 5)
  gw <- igraph::set_edge_attr(g, "weight", value = w)
  gw <- igraph::set_edge_attr(gw, "other", value = rev(w))
  run <- function(graph, ...) layout_forceatlas2(graph, 20, init = pos, ...)

  expect_identical(run(gw), run(g, weights = w))
  expect_identical(run(gw, weights = NA), run(g))
  expect_identical(run(gw, weights = "other"), run(g, weights = rev(w)))
  expect_identical(run(gw, edge_weight_influence = 0), run(g))
  expect_false(identical(run(gw), run(g)))
  expect_identical(run(gw, invert_weights = TRUE), run(g, weights = 1 / w))
  expect_identical(run(gw, normalize_weights = TRUE),
                   run(g, weights = (w - min(w)) / (max(w) - min(w))))
  expect_identical(run(g, weights = rep(3, m), normalize_weights = TRUE), run(g))
  expect_identical(run(g, weights = c(0, w[-1]), invert_weights = TRUE),
                   run(g, weights = c(0, 1 / w[-1])))

  expect_error(run(g, weights = "nope"), "weights")
  expect_error(run(g, weights = w[-1]), "one value per edge")
  expect_error(run(g, weights = -w), "non-negative")
  expect_error(run(g, weights = replace(w, 2, NA)), "finite")
  expect_error(run(g, weights = as.character(w)), "weights")
})

test_that("invalid arguments give clear errors", {
  g <- test_graph(10)
  expect_error(layout_forceatlas2(data.frame(from = 1, to = 2)), "tbl_graph or an igraph")
  expect_error(layout_forceatlas2(g, iterations = -1), "iterations")
  expect_error(layout_forceatlas2(g, iterations = 1.5), "iterations")
  expect_error(layout_forceatlas2(g, scaling_ratio = 0), "scaling_ratio")
  expect_error(layout_forceatlas2(g, gravity = -1), "gravity")
  expect_error(layout_forceatlas2(g, gravity = NA_real_), "gravity")
  expect_error(layout_forceatlas2(g, theta = -1), "theta")
  expect_error(layout_forceatlas2(g, threads = 0), "threads")
  expect_error(layout_forceatlas2(g, linlog = NA), "linlog")
  expect_error(layout_forceatlas2(g, barnes_hut = "yes"), "barnes_hut")
  expect_error(layout_forceatlas2(g, node_size = c(1, 2)), "node_size")
  expect_error(layout_forceatlas2(g, node_size = -1), "node_size")
  expect_error(layout_forceatlas2(g, node_size = "nope"), "node_size")
  expect_error(layout_forceatlas2(g, fixed = c(TRUE, FALSE)), "fixed")
})

test_that("defaults follow Gephi", {
  pos <- start_positions(100)
  g <- test_graph(100)
  expect_identical(layout_forceatlas2(g, 5, init = pos),
                   layout_forceatlas2(g, 5, init = pos, scaling_ratio = 2))
  expect_identical(layout_forceatlas2(test_graph(99), 5, init = pos[-1, ]),
                   layout_forceatlas2(test_graph(99), 5, init = pos[-1, ],
                                      scaling_ratio = 10))
  big <- test_graph(1000)
  pos <- start_positions(1000)
  expect_identical(layout_forceatlas2(big, 3, init = pos),
                   layout_forceatlas2(big, 3, init = pos, barnes_hut = TRUE))
  expect_false(identical(layout_forceatlas2(big, 3, init = pos),
                         layout_forceatlas2(big, 3, init = pos, barnes_hut = FALSE)))
})
