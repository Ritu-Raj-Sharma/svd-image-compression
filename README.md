# SVD Image Compression

A from-scratch exploration of using **Singular Value Decomposition (SVD)** to compress images, with visualizations showing how image quality changes as you keep more or fewer singular values.

## What is this?

Any grayscale image can be represented as a matrix. SVD decomposes that matrix into three components:

```
A = U Σ Vᵗ
```

By keeping only the top `r` singular values (and their corresponding vectors in `U` and `V`), you get the **best possible rank-r approximation** of the original image — using far less data than the full image. This project visualizes that tradeoff: how few singular values do you actually need before an image still looks "good enough"?

## How it works

The script (`svd.py`) does the following:

1. **Load the image and convert to grayscale**
   ```python
   A = imread('demo.jpeg')
   X = np.mean(A, -1)  # average R, G, B channels
   ```

2. **Display the original image**
   ```python
   img = plt.imshow(256 - X)
   img.set_cmap('gray')
   ```

3. **Compute the SVD**
   ```python
   U, S, VT = np.linalg.svd(X, full_matrices=False)
   S = np.diag(S)
   ```
   - `U`: left singular vectors
   - `S`: singular values (sorted largest to smallest)
   - `VT`: right singular vectors (transposed)

4. **Reconstruct low-rank approximations** for different values of `r`
   ```python
   for r in (5, 20, 100):
       X_approx = U[:, :r] @ S[0:r, :r] @ VT[:r, :]
   ```
   Each approximation keeps only the top `r` singular values/vectors — discarding the rest. The result is displayed and labeled with its rank.

The key insight: singular values are sorted from largest to smallest, and for most natural images, they decay quickly — meaning a small number of singular values capture most of the image's visual structure. That's *why* this works as compression.

## Project structure

```
svd-image-compression/
├── README.md
├── .gitignore
└── svd.py          # core SVD compression script
```

Note: the test image used (`demo.jpeg`) is excluded via `.gitignore` and is not included in this repo. To run the script, place your own image in the project folder and update the filename in `svd.py` accordingly.


This will display the original grayscale image, followed by reconstructions at rank 5, 20, and 100.

## Key concepts demonstrated

- Singular Value Decomposition (SVD) and low-rank matrix approximation
- Eckart–Young theorem (why truncated SVD gives the *optimal* rank-r approximation for a given rank)
- Tradeoff between compression ratio and reconstruction error
- Grayscale conversion and matrix representation of images

## Results & analysis

- **Singular value decay**: plotting singular value magnitude vs. index shows how quickly information is concentrated in the first few components
- **Compression ratio**: a rank-r approximation requires storing `r(m + n + 1)` numbers instead of the full `m × n` matrix — compression improves as `r` shrinks relative to image dimensions
- **Visual quality vs. r**: at low `r` (e.g. 5), the image is blurry/blocky; by `r = 100`, it's typically close to indistinguishable from the original, depending on image complexity

*(Add your own plots/numbers here once generated)*

## Roadmap

- [ ] Add singular value decay plot and reconstruction error metrics
- [ ] Extend to color images (per-channel SVD or YCbCr-based compression)
- [ ] Build an interactive web demo (slider for `r`, live before/after)
- [ ] Compare against JPEG (DCT-based) compression
- [ ] Implement SVD from scratch (e.g. via power iteration) as an extension

## License

MIT
