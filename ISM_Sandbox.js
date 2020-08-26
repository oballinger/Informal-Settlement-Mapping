Map.setCenter(39.2854, -6.8167, 14)

// load collections
var wards=ee.FeatureCollection("users/ollielballinger/Tanzania_Wards"),
    subwards=ee.FeatureCollection("users/ollielballinger/Tanzania_SubWards")
var OSM=ee.FeatureCollection("users/ollielballinger/Tanzania_OSM_clean")

// OSM raster of building height 
var empty = ee.Image().byte();
var OSM_color = empty.paint({
  featureCollection: OSM,
  color: 'building_3',
}).unmask(0).clip(wards)


//calculate reciprocal of each building footprint's area 
var OSM=OSM.map(function(feature) {
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
var density= OSM_color.reduceNeighborhood({
  reducer: ee.Reducer.sum(),
  kernel: gaussian
})

var palette =["0034f5","1e7d83","4da910","b3c120","fcc228","ff8410","fd3000"]

Map.addLayer(density,{palette:palette, min:0, max:1}, 'OSM density');

//i've saved the "density" layer as a raster, imported below. Using it speeds up the RF classification 
var Tanzania_Density = ee.Image("users/ollielballinger/Tanzania_Density")


//// S2 collection and processing

var sentinel_2A = ee.ImageCollection("COPERNICUS/S2_SR");
    
// filter out cloudy collctions
function maskS2clouds(image) {
  var qa = image.select('QA60')

  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;

  var mask = qa.bitwiseAnd(cloudBitMask).eq(0).and(
             qa.bitwiseAnd(cirrusBitMask).eq(0))

  return image.updateMask(mask).divide(10000)
      .select("B.*")
      .copyProperties(image, ["system:time_start"])
}

var collection = sentinel_2A
    .filterDate('2020-01-01', '2021-01-01')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .map(maskS2clouds)
    .median()

//Normalized Difference Vegetation Index     
var ndvi=(collection.select('B8')
          .subtract(collection.select('B4')))
          .divide(collection.select('B8')
          .add(collection.select('B4')))
          .select(['B8'],['NDVI'])


// Normalized Difference Built-Up Index          
var ndbi=(collection.select('B11')
          .subtract(collection.select('B8')))
          .divide(collection.select('B11')
          .add(collection.select('B8')))
          .select(['B11'],['NDBI'])


/// Sentinel 1 Synthetic Aperture Radar collection and processing 
var sentinel1 = ee.ImageCollection('COPERNICUS/S1_GRD');

// Filter by metadata properties.
var vh = sentinel1
  .filterDate('2019-01-01', '2019-12-31')
  // Filter to get images with VV and VH dual polarization.
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  // Filter to get images collected in interferometric wide swath mode.
  .filter(ee.Filter.eq('instrumentMode', 'IW'));

// Filter to get images from different look angles.
var vhAscending = vh.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'));
var vhDescending = vh.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'));

// Create a composite from means at different polarizations and look angles.
var sar = ee.Image.cat([
  vhAscending.select('VH').mean(),
  ee.ImageCollection(vhAscending.select('VV').merge(vhDescending.select('VV'))).mean(),
  vhDescending.select('VH').mean()
]).focal_median();

Map.addLayer(sar, {min: [-25, -20, -25], max: [0, 10, 0]}, 'SAR', false);

// add bands from all analysis layers 
var image = collection.addBands(OSM_color)//
                      .addBands(Tanzania_Density)//OSM density
                      .addBands(ndvi)
                      .addBands(ndbi)
                      .addBands(sar)//S1 SAR
                      .clip(wards)
                      .updateMask(ndvi.lt(0.3).and(ndbi.gt(0)))//filter out non-urban landcover


//input bands used for classification 
var bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12','NDVI','NDBI','VV','VH','VH_1','constant_sum']

var sample=ee.FeatureCollection([
                                  //blue_roof,
                                  metal_roof,
                                  //apartments,
                                  road,
                                  //suburban_red,
                                  //suburban_mixed,
                                  apartments_downtown
                                ])
                                  .flatten()
                                  .randomColumn();


// assign 60% of training points to validation 
var split=0.6
var training_sample = sample.filter(ee.Filter.lt('random', split));
var validation_sample = sample.filter(ee.Filter.gte('random', split));


// distance filter to mitigate autocorrelation 
// turned off for now because one landcover type is only found in a 1km area downtown
var distFilter = ee.Filter.withinDistance({
  distance: 1000,
  leftField: '.geo',
  rightField: '.geo',
  maxError: 10
});

var join = ee.Join.inverted();

// training_sample = join.apply(training_sample, validation_sample, distFilter);



// take samples from image for training and validation  
var training = image.select(bands).sampleRegions({
  collection: training_sample,
  properties: ['class'],
  scale: 10
});

var validation = image.select(bands).sampleRegions({
  collection: validation_sample,
  properties: ['class'],
  scale: 10
});



// Random Forest Classification

var rf1 = ee.Classifier.smileRandomForest(500)
    .train(training, 'class');
var rf2 = image.classify(rf1);

var validated = validation.classify(rf1);

Map.addLayer(rf2,{palette:palette, min:1 , max:3},'RF')


var trainAccuracy = rf1.confusionMatrix();
print('Resubstitution error matrix: ', trainAccuracy);
print('Training overall accuracy: ', trainAccuracy.accuracy());

var testAccuracy = validated.errorMatrix('class', 'classification');
print('Validation error matrix: ', testAccuracy);
print('Validation overall accuracy: ', testAccuracy.accuracy())
print('Validation consumer accuracy: ', testAccuracy.consumersAccuracy());



/// other supervised classification algorithms 

// Classification And Regression Tree
var cart1 = ee.Classifier.cart().train(training, "class", bands);
var cart2 = image.select(bands).classify(cart1);

// Support Vector Machine 
var svm1 = ee.Classifier.svm({
  kernelType: 'RBF',
  gamma: 0.5,
  cost: 10
});
var svm2 = svm1.train(training, 'class', bands);
var svm3 = image.classify(svm2);

// Naive Bayes

var bayes1 = ee.Classifier.naiveBayes()
    .train(training, 'class');
var bayes2 = image.classify(bayes1);






/// Unsupervised clustering testing

var cluster_image = collection
                  .select('B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12')
                  //.addBands(ndvi)
                  .addBands(ndbi)
                  //.addBands(sar)
                  .clip(wards)
                  .updateMask(ndvi.lt(0.3).and(ndbi.gt(0)))


//define 10k polygon around city center as AOR for sampling 
var aor=ee.Geometry.Point([39.2854, -6.8167]).buffer(10000)

//take a training sample of 10000 pixels
var cluster_training = cluster_image
                        .sample({
                          region: metal_roof_poly,
                          scale: 1,
                          numPixels: 10000
                        });

// set params and train; 
var clusterer = ee.Clusterer
                .wekaXMeans({
                      minClusters:5,
                      maxClusters:10
                      })
                .train(cluster_training);
                
var cluster = image.cluster(clusterer);

// figure out how many clusters were chosen 
var k=cluster.reduceRegion(ee.Reducer.max(), wards,100)//.get('cluster')

//create a smoother version of the output
var cluster_conv= cluster.reduceNeighborhood({
  reducer: ee.Reducer.median(),
  kernel: ee.Kernel.gaussian({
          radius: 30, 
          units: 'meters', 
          normalize: true, 
          sigma:50})
})

//adapt viz parameters to the number of clusters 
k.evaluate(function(dict) {
  var vizParams = {
    min: 1, 
    max: dict.cluster, 
    palette: palette
  }
  print("number of classes:", dict.cluster)
  Map.addLayer(cluster,vizParams,'cluster')
  //Map.addLayer(cluster_conv,vizParams,'cluster_conv')
})

var sd = cluster.reduceRegion(ee.Reducer.stdDev(), metal_roof_poly,10)
var mean = cluster.reduceRegion(ee.Reducer.mean(), metal_roof_poly,10)

Map.addLayer(image, {bands: ['B4', 'B3', 'B2'], min: 0, max: 0.3}, 'Sentinel-2', false)


