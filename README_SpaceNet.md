# SpaceNet dataset

https://registry.opendata.aws/spacenet/
https://spacenetchallenge.github.io/

https://medium.com/the-downlinq/getting-started-with-spacenet-data-827fd2ec9f53

https://devblogs.nvidia.com/solving-spacenet-road-detection-challenge-deep-learning/
https://devblogs.nvidia.com/exploring-spacenet-dataset-using-digits/

## Preprocessing

Download the data:

aws s3api get-object --bucket spacenet-dataset \
    --key AOI_1_Rio/processedData/processedBuildingLabels.tar.gz \
    --request-payer requester processedBuildingLabels.tar.gz

Image cutouts for the pan-sharpened 3-band imagery are 438–439 pixels in width, and 406–407 pixels 
in height. 8-band images have not been pan-sharpened and so have 1/4 the resolution of the 3-band 
imagery at 110 x 102 pixels. For each unique image ID we find a corresponding entry in the 
vectordata/geoJson directory with image footprints.

After unpacking `processedBuildingLabels.tar.gz` the images and geoJson files are in the following 
directories:

  * processedBuildingLabels/3band
  * processedBuildingLabels/vectordata/geojson

The next step is to transform the latitude-longitude coordinates in the GeoJSON label files to 
pixel coordinates.  

Install Python deps:

    conda install gdal==1.11.2
    conda install pandas==0.17.1
    conda install Rtree
    conda install numpy
    conda install geopandas

    cd utilities/python/
    python createDataSpaceNet.py /home/ubuntu/small --srcImageryDirectory 3band --geoJsonDirectory geoJson --outputDirectory /tmp/out

