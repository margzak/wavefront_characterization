# Wavefront characterization with `psf_utils`

A set of useful functions for wavefront characterization for square aperture optics like multilayer Laue lenses (MLLs). 

The example of typical wavefront characterization workflow is shown in the `example_wavefront_characterization.ipynb`. It includes:

1. Loading the wavefront data in the form of a h5 file. This file is the output of the Speckle tracking wavefront characterization technique which uses `pyrost` package. Important data for the following characterizatoin workflow are: 
- phase_2d: two-dimensional numpy array containing measured phase of the pupil (here in radians). 
- whitefield: two-dimensional numpy array containing the intensity data measured in the pupil (here with photon counting detector). 
- wavelength, pixel size and detector distance from the wavefront measurements. 

2. Performing zernike decomposition of the wavefront. Here I use `zernike` python module, which can be used for square aperture optics. 

3. Building the PSFs for a range of defoci, finding the brightest PSF (with maximum intensity). Creating a plot of beam caustics for the scanned range of defoci. Calculating the focal spot size as a FWHM of PSF for the lens system. 

4. Creating an encircled energy plot for the stack of PSFs for different defoci. 


