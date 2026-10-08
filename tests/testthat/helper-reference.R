# A plain R transcription of the ForceAtlas2 iteration as implemented in Gephi
# (exact repulsion, no Barnes-Hut), used to check the C code.
reference_fa2 <- function(graph, pos, iterations, scaling, gravity = 1,
                          strong = FALSE, linlog = FALSE, outbound = FALSE,
                          adjust = FALSE, size = 1, w = NULL, jt_user = 1,
                          fixed = NULL) {
  n <- igraph::vcount(graph)
  el <- igraph::as_edgelist(graph, names = FALSE)
  a <- el[, 1]
  b <- el[, 2]
  if (is.null(w)) w <- rep(1, nrow(el))
  if (is.null(fixed)) fixed <- rep(FALSE, n)
  size <- rep_len(size, n)
  mass <- 1 + tabulate(c(a, b), n)
  x <- pos[, 1]
  y <- pos[, 2]
  dx <- dy <- numeric(n)
  speed <- 1
  eff <- 1
  mm <- outer(mass, mass)
  ss <- outer(size, size, "+")

  for (it in seq_len(iterations)) {
    odx <- dx
    ody <- dy

    # Repulsion
    xd <- outer(x, x, "-")
    yd <- outer(y, y, "-")
    d <- sqrt(xd^2 + yd^2)
    if (adjust) {
      d <- d - ss
      f <- ifelse(d > 0, scaling * mm / d^2,
                  ifelse(d < 0, 100 * scaling * mm, 0))
    } else {
      f <- ifelse(d > 0, scaling * mm / d^2, 0)
    }
    diag(f) <- 0
    dx <- rowSums(xd * f)
    dy <- rowSums(yd * f)

    # Gravity
    d0 <- sqrt(x^2 + y^2)
    g <- if (strong) gravity * mass else ifelse(d0 > 0, gravity * mass / d0, 0)
    dx <- dx - x * g
    dy <- dy - y * g

    # Attraction
    coef <- if (outbound) mean(mass) else 1
    for (e in seq_along(a)) {
      i <- a[e]
      j <- b[e]
      ex <- x[i] - x[j]
      ey <- y[i] - y[j]
      fe <- -coef * w[e]
      if (adjust || linlog) {
        de <- sqrt(ex^2 + ey^2)
        if (adjust) de <- de - size[i] - size[j]
        if (de <= 0) next
        if (linlog) fe <- fe * log(1 + de) / de
      }
      if (outbound) fe <- fe / mass[i]
      dx[i] <- dx[i] + ex * fe
      dy[i] <- dy[i] + ey * fe
      dx[j] <- dx[j] - ex * fe
      dy[j] <- dy[j] - ey * fe
    }

    # Speed
    swing_i <- mass * sqrt((odx - dx)^2 + (ody - dy)^2)
    swing <- sum(swing_i[!fixed])
    tract <- sum((0.5 * mass * sqrt((odx + dx)^2 + (ody + dy)^2))[!fixed])
    est <- 0.05 * sqrt(n)
    jt <- jt_user * max(sqrt(est), min(10, est * tract / n^2))
    if (swing / tract > 2) {
      if (eff > 0.05) eff <- eff * 0.5
      jt <- max(jt, jt_user)
    }
    target <- jt * eff * tract / swing
    if (swing > jt * tract) {
      if (eff > 0.05) eff <- eff * 0.7
    } else if (speed < 1000) {
      eff <- eff * 1.3
    }
    speed <- speed + min(target - speed, 0.5 * speed)

    # Displacement
    if (adjust) {
      fac <- 0.1 * speed / (1 + sqrt(speed * swing_i))
      df <- sqrt(dx^2 + dy^2)
      fac <- ifelse(df == 0, 0, pmin(fac * df, 10) / df)
    } else {
      fac <- speed / (1 + sqrt(speed * swing_i))
    }
    fac[fixed] <- 0
    x <- x + dx * fac
    y <- y + dy * fac
  }
  cbind(x, y, deparse.level = 0)
}

test_graph <- function(n = 40, directed = FALSE, seed = 42) {
  set.seed(seed)
  igraph::sample_pa(n, m = 2, directed = directed)
}

start_positions <- function(n, seed = 7) {
  set.seed(seed)
  matrix(stats::runif(2 * n, -50, 50), ncol = 2)
}
