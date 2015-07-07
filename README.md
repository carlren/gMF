# gMF

CUDA based mean-field inference for dense CRF

## Build:
```
mkdir build
cd build
cmake ..
make -j12
```

Then test with 

```
./gMF ../data/2007_000129.jpg ../data/2007_000129.png ../data/2007_000129_test.png
./gMF <input image> <input unary> <output image>
```

Play with the parameters in demo.cpp, have fun!
