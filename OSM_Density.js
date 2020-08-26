Map.setCenter(39.2854, -6.8167, 12)


// import OSM data as vector layer
var OSM_clean=ee.FeatureCollection("users/ollielballinger/Tanzania_OSM_clean")


//calculate reciprocal of each building footprint's area 
var OSM=OSM_clean.map(function(feature) {
              return feature.set({area: ee.Number(1).divide(feature.geometry().area()).multiply(100)}).set({const: 1})})

//convert to raster using reciprocal area as the band  
var empty = ee.Image().byte();
var OSM_color = empty.paint({
  featureCollection: OSM,
  color: 'area',
}).unmask(0)  
  
  
// define kernel
var gaussian = ee.Kernel.gaussian({
  radius: 60, units: 'meters', normalize: true, sigma:15
});

// calculate density of small buildings
var sums= OSM_color.reduceNeighborhood({
  reducer: ee.Reducer.sum(),
  kernel: gaussian
})


var OSM_smooth = OSM_color.convolve(gaussian);

var palette =["0034f5","1e7d83","4da910","b3c120","fcc228","ff8410","fd3000"]

Map.addLayer(OSM_color,{palette:palette, max:1}, 'OSM_color');
Map.addLayer(sums,{palette:palette, min:0, max:1}, 'OSM_smooth');

