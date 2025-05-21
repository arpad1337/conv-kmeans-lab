# conv-kmeans-lab

Convolutional K-means Color Segmenter in CIELAB

## Usage

```cpp

  KMeansColorSegmenter segmenter = KMeansColorSegmenter::create(inputImage, colors, padding);
  segmenter.train(epochs);
  segmenter.convert();

  ...

  segmenter.setPreviousOutputAsInput();
  segmenter.setK(colors2);
  segmenter.train(epochs2);
  segmenter.convert();

  ...

  Mat processed = segmenter.getLABOutput();

```

## Author

@arpad1337

## License

MIT
