# Golden test assets

The golden tests are generated procedurally at test time so this repository
contains no proprietary camera files or photographed subject matter.

- The rendered fixture is a pair of constant, losslessly encoded 16-bit RGB
  TIFF frames with an analytically known weighted-stack result.
- The RAW fixture uses the tiny synthetic same-camera DNG generator documented
  in `tests/fixtures/raw/README.md`. It contains a deterministic Bayer ramp and
  CC0-style generated metadata, not samples copied from a commercial camera.

The tests use numeric tolerances for pixel math and exact checks for processing
domain, camera identity, decoder recipe, DNG conformance, and provenance.
