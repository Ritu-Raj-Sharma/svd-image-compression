# SVD Image Compression

A from-scratch exploration of using **Singular Value Decomposition (SVD)** to compress images, with visualizations showing how image quality changes as you keep more or fewer singular values. Includes both grayscale and color implementations.

## What is this?

Any image can be represented as a matrix (or, for color, three matrices — one per channel). SVD decomposes a matrix into three components:

```
A = U Σ Vᵗ
```

By keeping only the top `r` singular values (and their corresponding vectors in `U` and `V`), you get the **best possible rank-r approximation** of the original matrix — using far less data than the full image. This project visualizes that tradeoff: how few singular values do you actually need before an image still looks "good enough"?

## How it works

### Grayscale (`svd.py`)

1. **Load the image and convert to grayscale**
   ```python
   A = imread('demo.jpeg')
   X = np.mean(A, -1)  # average R, G, B channels
   ```

2. **Display the original image**
   ```python
   img = plt.imshow(X)
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

5. **Singular value plots** — a log-scale plot of singular value magnitude, and a cumulative-sum plot showing how much "energy"/information is captured as more singular values are included.

### Color (`svdcolour.py`)

Color images can't be SVD'd directly as a single 2D matrix — they're 3D (height × width × 3 channels). The approach: **split the image into its R, G, B channels and run SVD on each one independently**, then recombine.

```python
A = imread('demo.jpeg').astype(np.float64)

for r in (5, 20, 100):
    channels_approx = []
    for c in range(3):  # R, G, B
        X = A[:, :, c]
        U, S, VT = np.linalg.svd(X, full_matrices=False)
        S = np.diag(S)
        X_approx = U[:, :r] @ S[:r, :r] @ VT[:r, :]
        channels_approx.append(X_approx)

    Xapprox = np.stack(channels_approx, axis=-1)
    Xapprox = np.clip(Xapprox, 0, 255).astype(np.uint8)  # keep pixels valid
```

Key differences from the grayscale version:
- **3 separate SVDs** are computed per rank `r` (one per channel) instead of 1
- Reconstructed channel values must be **clipped to [0, 255]** before display, since low-rank approximation can push pixel values slightly outside the valid range
- This is noticeably slower than grayscale, since you're tripling the SVD computation per image

The key insight behind both versions: singular values are sorted from largest to smallest, and for most natural images, they decay quickly — meaning a small number of singular values capture most of the image's visual structure. That's *why* this works as compression.

## Project structure

```
svd-image-compression/
├── README.md
├── .gitignore
├── svd.py            # grayscale SVD compression script
└── svdcolour.py      # color (per-channel) SVD compression script
```

Note: the test image used (`demo.jpeg`) is excluded via `.gitignore` and is not included in this repo. To run either script, place your own image in the project folder and update the filename accordingly.

## Getting started

Clone the repo:

```bash
git clone https://github.com/Ritu-Raj-Sharma/svd-image-compression.git
cd svd-image-compression
```

Install dependencies:

```bash
pip install numpy matplotlib pillow
```

Add an image (e.g. `demo.jpeg`) to the project folder, then run either script:

```bash
python svd.py         # grayscale version
python svdcolour.py   # color version
```

This will display the original image, followed by reconstructions at rank 5, 20, and 100 (plus singular value plots, for the grayscale script).

## Key concepts demonstrated

- Singular Value Decomposition (SVD) and low-rank matrix approximation
- Eckart–Young theorem (why truncated SVD gives the *optimal* rank-r approximation for a given rank)
- Tradeoff between compression ratio and reconstruction error
- Per-channel decomposition for color images
- Grayscale conversion and matrix representation of images

## Results & analysis

- **Singular value decay**: plotting singular value magnitude vs. index shows how quickly information is concentrated in the first few components
- **Compression ratio**: a rank-r approximation requires storing `r(m + n + 1)` numbers instead of the full `m × n` matrix (per channel, for color) — compression improves as `r` shrinks relative to image dimensions
- **Visual quality vs. r**: at low `r` (e.g. 5), the image is blurry/blocky; by `r = 100`, it's typically close to indistinguishable from the original, depending on image complexity
- **Grayscale vs. color cost**: the color version requires 3x the SVD computations of the grayscale version, since each channel is decomposed independently

*(Add your own plots/numbers here once generated)*

## Roadmap

- [ ] Add singular value decay plot and reconstruction error metrics for the color version
- [ ] Explore YCbCr-based compression (allocate higher rank to luminance, lower rank to chrominance, since human eyes are more sensitive to brightness than color)
- [ ] Build an interactive web demo (slider for `r`, live before/after)
- [ ] Compare against JPEG (DCT-based) compression
- [ ] Implement SVD from scratch (e.g. via power iteration) as an extension
