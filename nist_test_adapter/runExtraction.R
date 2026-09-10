source( 'R/extractStrokes_TSP.R' )
source( 'R/findEndpoints_rowcol.R' )
source( 'R/turn.R' )
source( 'R/constructInputData.R' )
source( 'R/removeSmallComponents.R' )
source( 'R/processImagesCustom.R' )
source( 'R/plotstrokes.R' )
source( 'R/getPointList.R' )
library( TSP )

imagefolder = 'thinned_images/'
resultfolder = '../output/'

# Process all thinned images (no labels needed for visualization)
processImagesCustom( imagefolder, resultfolder )

