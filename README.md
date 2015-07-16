# gMF

CUDA based mean-field inference for dense CRF

## Build:
```
mkdir build
cd build
cmake ..
make -j12
```

Multi-class segmentation demo:

```
./gMF ../data/2007_000129.jpg ../data/2007_000129.png ../data/2007_000129_test.png
./gMF <input image> <input unary> <output image>
```

Play with the parameters in demo.cpp, have fun!


Binary segmentation demo:

```
./bSeg ../data/me.mp4 ../data/msk.ppm
./bSeg <input video> <mask image>
```

'x' to next frame
'z' to previous frame
'q' to exit

Binary segmentation demo, with grascale per-pixel likelihood input:

```
cd ../data/
sh batch_binary.sh
```

then go to the data/binary folder to see results