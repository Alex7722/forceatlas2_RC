# forceatlas2r

ForceAtlas2, the graph layout algorithm of [Gephi](https://gephi.org), for R.

* **Fast**: written in C, with the Barnes-Hut approximation for large graphs
  and optional multithreading.
* **Light**: the only dependency is igraph (which tidygraph already requires).
  No Java, no Rcpp, no system library.
* **Tidy**: `forceatlas2()` takes a `tbl_graph` and returns it with `x` and `y`
  columns added to the node table.

## Installation

From the source archive:

```r
install.packages("forceatlas2r_0.1.0.tar.gz", repos = NULL, type = "source")
```

or, once the package is in a Git repository:

```r
remotes::install_github("<user>/forceatlas2r")
```

The package contains C code, so installing it from source needs a C compiler:
[Rtools](https://cran.r-project.org/bin/windows/Rtools/) on Windows, the Xcode
command line tools on macOS (`xcode-select --install`), `r-base-dev` or the
equivalent on Linux.

## Usage

```r
library(tidygraph)
library(forceatlas2r)

graph <- play_islands(n_islands = 5, size_islands = 30, p_within = 0.3, m_between = 2)

set.seed(1)                      # the starting positions are random
graph <- graph %>% forceatlas2()
graph
#> # A tbl_graph: 150 nodes and 661 edges
#> #
#> # An undirected simple graph with 1 component
#> #
#> # Node Data: 150 x 2 (active)
#>        x     y
#>    <dbl> <dbl>
#>  1  137.  160.
#>  2  151.  187.
#>  3  136.  178.
#> ...
```

The coordinates are ordinary node columns, so they can be used by any plotting
tool. With ggraph:

```r
library(ggraph)

ggraph(graph, layout = "manual", x = x, y = y) +
  geom_edge_link(alpha = 0.2) +
  geom_node_point()

# or let ggraph call the layout itself
ggraph(graph, layout = forceatlas2, linlog = TRUE) +
  geom_edge_link(alpha = 0.2) +
  geom_node_point()
```

`layout_forceatlas2()` returns the coordinates as a matrix instead, like
igraph's own layout functions:

```r
g <- igraph::sample_pa(500, directed = FALSE)
plot(g, layout = layout_forceatlas2(g), vertex.size = 2, vertex.label = NA)
```

## Settings

The arguments are Gephi's settings:

| Gephi                  | forceatlas2r            | Default                           |
|:-----------------------|:------------------------|:----------------------------------|
| (number of steps)      | `iterations`            | 1000                              |
| Scaling                | `scaling_ratio`         | 10 below 100 nodes, 2 otherwise   |
| Gravity                | `gravity`               | 1                                 |
| Stronger Gravity       | `strong_gravity`        | `FALSE`                           |
| LinLog mode            | `linlog`                | `FALSE`                           |
| Dissuade Hubs          | `dissuade_hubs`         | `FALSE`                           |
| Prevent Overlap        | `prevent_overlap`, `node_size` | `FALSE`, 1                 |
| Edge Weight Influence  | `edge_weight_influence` | 1                                 |
| Normalize edge weights | `normalize_weights`     | `FALSE`                           |
| Inverted edge weights  | `invert_weights`        | `FALSE`                           |
| Tolerance (speed)      | `jitter_tolerance`      | 1                                 |
| Approximate Repulsion  | `barnes_hut`            | from 1000 nodes                   |
| Approximation          | `theta`                 | 1.2                               |
| Threads number         | `threads`               | 1                                 |

Edge weights are taken from the `weight` edge column when there is one; use
`weights = "other_column"` to pick another column or `weights = NA` to ignore
them.

A layout can be continued or refined by starting from the stored coordinates,
for instance to remove overlaps once the layout has settled, and some nodes can
be pinned in place:

```r
graph <- graph %>%
  forceatlas2() %>%
  forceatlas2(init = c("x", "y"), iterations = 200,
              prevent_overlap = TRUE, node_size = 2)

graph %>%
  mutate(pinned = centrality_degree() > 20) %>%
  forceatlas2(init = c("x", "y"), fixed = "pinned")
```

## Differences with Gephi

* Gephi runs the layout until you stop it; here the number of `iterations` is
  fixed in advance. Large graphs may need more than the default.
* With the Barnes-Hut approximation, Gephi counts the repulsion between nearby
  nodes twice. This package counts it once, so that the approximation converges
  to the exact computation as `theta` decreases.
* Computations are done in double precision (Gephi stores positions in single
  precision).

## References

Jacomy M, Venturini T, Heymann S, Bastian M (2014). ForceAtlas2, a Continuous
Graph Layout Algorithm for Handy Network Visualization Designed for the Gephi
Software. *PLoS ONE* 9(6): e98679. <https://doi.org/10.1371/journal.pone.0098679>

The implementation follows the reference Java implementation distributed with
Gephi (© Gephi Consortium, GPL-3 / CDDL). This package is released under the
GPL (>= 3).
