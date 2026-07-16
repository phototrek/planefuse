# RAW test fixtures

The engine tests generate tiny DNG files at runtime from deterministic NumPy
arrays. The generated sensor mosaic, tags, and camera identity are original
PlaneFuse test data and may be redistributed under the same terms as this
repository. No manufacturer file, profile, image, or proprietary user photo is
included.

The mosaic uses constant R/G/B sensor values so tests can detect accidental
white balance, gamma, automatic brightness, denoise, or tone processing.
