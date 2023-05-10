	calibrate.my <- function (x, xlab, ylab, xlim, ylim, legend = TRUE, subtitles = TRUE, 
	  cex.subtitles = 0.75, riskdist = TRUE, scat1d.opts = list(nhistSpike = 200), 
		 ...) {
	    at <- attributes(x)
	    if (missing(ylab)) 
		ylab <- if (at$model == "lr") 
		    "Actual Probability"
		else paste("Observed", at$yvar.name)
	    if (missing(xlab)) {
		if (at$model == "lr") {
		    xlab <- paste("Predicted Pr{", at$yvar.name, 
			sep = "")
		    if (at$non.slopes == 1) {
			xlab <- if (at$lev.name == "TRUE") 
			  paste(xlab, "}", sep = "")
			else paste(xlab, "=", at$lev.name, "}", 
			  sep = "")
		    }
		    else xlab <- paste(xlab, ">=", at$lev.name, 
			"}", sep = "")
		}
		else xlab <- paste("Predicted", at$yvar.name)
	    }
	    p <- x[, "predy"]
	    p.app <- x[, "calibrated.orig"]
	    p.cal <- x[, "calibrated.corrected"]
	    if (missing(xlim) & missing(ylim)) 
		xlim <- ylim <- range(c(p, p.app, p.cal), na.rm = TRUE)
	    else {
		if (missing(xlim)) 
		    xlim <- range(p)
		if (missing(ylim)) 
		    ylim <- range(c(p.app, p.cal, na.rm = TRUE))
	    }
	    plot(p, p.app, xlim = xlim, ylim = ylim, xlab = xlab, ylab = ylab, 
		type = "n", ...)
	    predicted <- at$predicted
	    err <- NULL
	    if (length(predicted)) {
		s <- !is.na(p + p.cal)
		err <- predicted - approx(p[s], p.cal[s], xout = predicted, 
		    ties = mean)$y
		cat("\nn=", n <- length(err), "   Mean absolute error=", 
		    round(mae <- mean(abs(err), na.rm = TRUE), 3), "   Mean squared error=", 
		    round(mean(err^2, na.rm = TRUE), 5), "\n0.9 Quantile of absolute error=", 
		    round(quantile(abs(err), 0.9, na.rm = TRUE), 3), 
		    "\n\n", sep = "")
		if (subtitles) 
		    title(sub = paste("Mean absolute error=", round(mae, 
			3), " n=", n, sep = ""), cex.sub = cex.subtitles, 
			adj = 1)
		if (riskdist) 
		    do.call("scat1d", c(list(x = predicted), scat1d.opts))
	    }
	    lines(p, p.app, lty = 2, lwd = 2, col = "red")
	    lines(p, p.cal, lty = 1, lwd = 2, col = "blue")
	    abline(a = 0, b = 1, lty = 3, lwd = 2, col = "black")
	    if (subtitles) 
		title(sub = paste("B=", at$B, "repetitions,", 
		    at$method), cex.sub = cex.subtitles, adj = 0)
	    if (!(is.logical(legend) && !legend)) {
		if (is.logical(legend)) 
		    legend <- list(x = xlim[1] + 0.55 * diff(xlim), y = ylim[1] + 
			0.32 * diff(ylim))
		legend(legend, c("Apparent", "Bias-corrected", 
		    "Ideal"), lty = c(2, 1, 3), lwd = c(2,2,2), cex = 1.5, col = c("red","blue","black"), 
		    text.col = c("red","blue","black"), bty = "n")
	    }
	    invisible(err)
	}
