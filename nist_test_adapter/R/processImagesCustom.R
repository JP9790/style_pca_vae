processImagesCustom <- function( imagefolder, resultfolder )
{
    # Get list of thinned image files
    thinned_files <- list.files(path = imagefolder, pattern = "-thinned\\.txt$", full.names = FALSE)
    nrimages <- length(thinned_files)
    
    print( paste( 'Found', nrimages, 'thinned images to process'))
    
    if (nrimages == 0) {
        print("No thinned images found!")
        return()
    }
    
    for ( i in 1:nrimages )
    {
        thinned_file <- thinned_files[i]
        # Extract base name (remove -thinned.txt suffix)
        base_name <- sub("-thinned\\.txt$", "", thinned_file)
        
        print( paste( 'Processing image:', base_name))
        flush.console()
        
        fn <- paste( imagefolder, thinned_file, sep = '' )
        thinned <- as.matrix( read.table( fn ))
        img <- thinned > 0
        img_denoised <- removeSmallComponents( img )
        
        # Extract strokes using TSP
        points <- extractStrokes_TSP( img_denoised )
        fn <- paste( resultfolder, base_name, '-points.txt', sep = '' )
        write.csv(points, file = fn, quote = F, row.names = F )
        
        # Create input data without class outputs (we don't have labels)
        addClassOutputs <- F
        inputdata <- constructInputData( points, 0, addClassOutputs )  # label = 0 (dummy)
        
        fn <- paste( resultfolder, base_name, '-inputdata.txt', sep = '' )
        write.table( inputdata, file = fn, quote = F, row.names = F, col.names = F )
        
        # Create a visualization plot
        tryCatch({
            png(paste( resultfolder, base_name, '-strokes.png', sep = '' ), width = 400, height = 400)
            plot(points[,1], points[,2], type = 'l', main = paste('Strokes:', base_name),
                 xlab = 'X', ylab = 'Y', asp = 1)
            # Draw points in order
            points_valid <- points[points[,1] >= 0 & points[,2] >= 0, , drop = FALSE]
            if (nrow(points_valid) > 0) {
                points(points_valid[,1], points_valid[,2], col = 'red', pch = 20, cex = 0.5)
                # Mark start point
                points(points_valid[1,1], points_valid[1,2], col = 'green', pch = 19, cex = 2)
            }
            dev.off()
        }, error = function(e) {
            print(paste("Could not create plot for", base_name, ":", e$message))
        })
    }
    
    print(paste("Processed", nrimages, "images successfully!"))
}
