# SMGP-2021 - Group 1
## Shape modeling and geometry processing project

Get the all_data folder from https://polybox.ethz.ch/index.php/s/lRKp9AjerxiQSzC
Put all folders in the all_data.zip into the data folder (if you are confused, look at the gitignore)

# Assignment 6 - Report

In this group assignment, we apply geometry processing to a real dataset of scanned 3D faces in order to perform morphing between them. The pipeline is made of the following 5 steps that we will cover in detail in this report:

+ Preprocessing of scanned faces
+ Landmark selection on template and scans
+ Rigid face alignment
+ Non-rigid face alignment
+ PCA on the registered faces

Additionally, a section is dedicated to the GUI and bonus tasks are discussed at the end of the report.

## 1. Preprocessing
**Worked on by:** Pascal Chang

**Relevant files:** `Preprocessor.h|.cpp`

Even though the scan faces from the dataset are supposed to be cleaned already, we observed that some meshes still contain multiple connected components and/or have rough boundaries. That is why we decided to do our own preprocessing on the dataset in five steps:

+ **Clean connected components**: find the largest connected components using `igl::facet_components`, remove the others. We also delete loose vertices and redirect face indices using `igl::remove_unreferenced`.
+ **Compute distance field to boundary**: compute the closest distance to boundary for each vertex in the mesh. Ideally this should be geodesic distance, but since we will only interested in points that are near the boundary, Euclidean distance is fine. This is done with nearest neighbor search using a KD-tree structure.
+ **Smooth distance field**: using the energy optimization formulation seen in the lectures, we smooth the scalar field to get smoother isolines. This seemed to be more stable than smoothing with the cotan Laplacian.
+ **Remesh along isoline**: create additional edges and vertices along a given isoline of the scalar distance field using `igl::remesh_along_isoline`.
+ **Cut mesh and remove the boundaries**: cut the mesh along the isoline and keep only the main component (same as step 1). We use `igl::cut_mesh`.

| 0. Initial scanned face | 1. Keep largest component | 2. Compute distance field | 
|:--:|:--:|:--:|
|<img src="results/1-initial.png" width="100%"> | <img src="results/1-removeCC.png" width="100%"> | <img src="results/1-distance.png" width="100%"> |

| 3. Smooth distance field | 4. Remesh along isoline | 5. Cut & keep largest |
|:--:|:--:|:--:|
 <img src="results/1-smooth.png" width="100%"> | <img src="results/1-remesh.png" width="100%"> | <img src="results/1-cut.png" width="100%"> |

The remeshing function in libigl creates duplicate vertices that we remove using `igl::remove_duplicate_vertices`. However, the resulting mesh can (in rare cases) become not edge-manifold. Because the remeshing requires edge-manifold meshes as input, the preprocessing is skipped for these few meshes (~ 3 out of 112).

## 2. Landmark selection
**Worked on by:** Isaak Hanimann

**Relevant files:** `LandmarkSelector.h|.cpp`

We used 23 landmarks. Those are the same landmarks as the example landmarks provided by the TA's.
<img src="results/landmark-selection.png" alt="drawing" width="500"/>

To specify a landmark the user enables selection and clicks on the face mesh. A ray is cast in the view direction starting from the mouse position and the intersection with the mesh is calculated.

The intersection is then stored as a landmark which consists of:

+  index of the face (triangle) of the intersection
+  barycenter coordinates of the point within the triangle

We chose this format for the landmarks because it allows us to specify landmarks with arbitrary precision.
The landmarks are identified by their index in the list of landmarks. So they must be added in the order of the image above.

Once the 23 landmarks have been specified one can save the landmarks to a text file next to the obj file such that they can be loaded from there again for later use.

## 3. Rigid face alignment
**Worked on by:** Franz Knobel & Pascal Chang

**Relevant files:** `FaceRegistor.h|.cpp`

Rigid alignment is a necessary step before warping the template mesh to the scanned face. This is done by first centering both:

+ **Center the scanned face and the template** such that the barycenter of their landmarks is at the origin.
+ **Re-scale the template** such that the average distance to the mean landmark is the same as the scanned face.
+ **Find optimal rotation matrix** between the two landmark sets using Kabsch algorithm. This basically extracts the rotation from an SVD decomposition.
+ **Apply rotation** to the scanned face.

It is the scanned face that is rotated so that the registered faces (template) all have the same orientation, which is necessary for performing meaningful PCA later. The different steps are illustrated below.

| Non aligned scan & template | Center & scale | Rotated & aligned | 
|:--:|:--:|:--:|
|<img src="results/3-initial.png" width="100%"> | <img src="results/3-centernscale.png" width="100%"> | <img src="results/3-aligned.png" width="100%"> |

As a general observation, the noses are often not well aligned after applying rigid alignment. This is where non-rigid warping comes in.

## 4. Non-rigid face alignment
**Worked on by:** Franz Knobel & Pascal Chang

**Relevant files:** `FaceRegistor.h|.cpp`

In order to perform PCA on the scanned faces we first need to register them with a common triangulation (template). Once the faces are rigidly aligned, we need to warp the template to match closely the surface of the scanned faces.

This is done in an iterative manner by solving repeatedly a linear system involving several terms. 

<img width="25%"><img src="results/4-system.png" width="50%">

+ **Laplacian smoothing term**: encourages the warped mesh to be as smooth as itself from the previous iteration.
+ **Boundary constraints**: enforces the vertices at the boundary of the template mesh to stay still.
+ **Landmark constraints**: enforces the position of the landmarks on the template to match the one of those on the scanned face.
+ **Dynamic constraints**: when a vertex of the template becomes close enough to a vertex on the scanned mesh, encourage it to match it exactly. These constraints change at every iteration and the threshold for "how close is close" is determined by the parameter `epsilon`. 

While most of the constraints leads to rows in the system matrix to be zero except at one position where it is 1, the landmark constraints are a bit different. In fact, since our landmarks are precise points on a mesh defined by barycenter coordinates on a given triangular face, the landmark constraints actually give rise to rows with 3 non-zero values that sum up to 1. Thus, if we want to keep this precision (instead of assigning to the nearest vertex), we cannot use the substitution method as in Assignment 5.

Here are the result for each iteration:

| Iteration 0 (rigid aligned) | Iteration 1 | Iteration 2 | Iteration 3 |
|:---:|:---:|:---:|:---:|
|<img src="results/4-step0.png" width="100%"> |<img src="results/4-step1.png" width="100%"> |<img src="results/4-step2.png" width="100%"> |<img src="results/4-step3.png" width="100%"> |

| Iteration 4 | Iteration 5 | Iteration 6 | Reference scanned face|
|:---:|:---:|:---:|:---:|
|<img src="results/4-step4.png" width="100%"> |<img src="results/4-step5.png" width="100%"> |<img src="results/4-step5.png" width="100%"> | <img src="results/4-reference.png" width="100%"> |

In general we perform the first iteration with a very small `epsilon` (e.g. = 0.01) so that there are no dynamic constraints. This makes sure that we first match the landmarks. Then,  the subsequent iterations are performed with a larger `epsilon` (~ 3.0). 
 
Solving the system above turned out to be slow because it is a rectangular matrix and corresponding solvers in `Eigen` are not as fast as the Simplicial Cholesky solver we used in Assignment 5. However, we can transform the system to obtain one with a square matrix which is SPD (as shown below). This allows us to use fast solvers for SPD matrices and get interactive computation rates even on the larger templates.

<img width="20%"><img src="results/4-solve_trick.png" width="60%">

We observed that reducing `lambda` gives smoother results especially on the boundaries of the scan mesh, but doing less iterations seems to give similar results. We set `lambda=1`.

## 5. PCA of faces
**Worked on by:** Clemens Bachmann & Nicolas Wicki

**Relevant files:** `PCA.h|.cpp`

**Compute mean face:**
To solve this task, we gather a dataset of faces in a data structure. We proceed by computing the mean face and use it to compute the offset between this mean face and each face in the data set.

**Prepare covariance matrix:**
Then, we construct a matrix A where each column represents one of these offsets. We construct the covariance matrix using Matrix A as described in [1] to significantly reduce computation time.

**Compute Eigen decomposition:**
Compute the PCA from which we can reconstruct the most dominant Eigen vectors of our dataset.

**Compute weights for approximation:**
To most accurately represent the original faces using those Eigen faces, we compute the dot product between the offset, and the Eigen faces to compute the weight for each Eigen face. Using these weights we can reconstruct the original faces through a linear combination of all Eigen faces scaled by those weights added to the mean face.

| Mean face | 1 Eigen face | 5 Eigen faces | 10 Eigen faces |
|:---:|:---:|:---:|:---:|
|<img src="results/5-mean-face.png" width="120" height="200"> |<img src="results/5-one-eigen-face.png" width="120" height="200"> |<img src="results/5-five-eigen-faces.png" width="120" height="200"> |<img src="results/5-ten-eigen-faces.png" width="120" height="200"> |

| 15 Eigen faces | 20 Eigen faces | 25 Eigen faces | Original face|
|:---:|:---:|:---:|:---:|
|<img src="results/5-15-eigen-faces.png" width="120" height="200"> |<img src="results/5-20-eigen-faces.png" width="120" height="200"> |<img src="results/5-25-eigen-faces.png" width="120" height="200"> | <img src="results/5-original-face.png" width="120" height="200"> |

**Compute morphing:**
We continued with the implementation of a morphing mechanism which is computed by linearly interpolating offsets (linear combinations of Eigen faces) of two faces and adding them to the mean face. This enables morphing from one face to another.
<img src="results/5-morph-face.gif" width="1000" height="500" />


[1]: Matthew Turk and Alex Pentland. 1991. Eigenfaces for recognition. J. Cognitive Neuroscience 3, 1 (Winter 1991), 71–86. DOI:https://doi.org/10.1162/jocn.1991.3.1.71

## 6. UI
**Worked on by**: Whole group (most credit goes to Pascal Chang)

To ensure modularity, the GUI contains a menu bar for selecting the desired menu. Each menu is associated with a specific task of the pipeline and can be viewed as an application on its own. There are five menus:

+ **Viewer**: shows the default viewer menu. It is possible to select this menu without loosing the ongoing task.
+ **Preprocessing**
+ **Landmark Selection**
+ **Face Registration** (rigid & non-rigid alignment)
+ **PCA Computation**

### Preprocessing UI

<img src="results/UI-preprocessing_general.png" width="100%">

**To preprocess a mesh manually:**

+ Start by selecting a mesh in the file browser
+ Successively click on `Clean connected components` → `Show signed distance` → `Smooth scalar field`
+ The terminal should show the range of distance e.g. `Scalar field distance range = [ 0 ; 92.3024 ]`
+ In `iso value` input box, select the isoline value to cut (if outside of distance range nothing will happen). The default `5.0` works for the dataset.
+ Successively select `Remesh & cut along isoline` → `Clean connected components`.

**To preprocess a mesh automatically:**

+ Start by selecting a mesh in the file browser
+ Click on `Preprocess face`

**To save a mesh:**

Simply select `Save mesh`. The current mesh (preprocessed or not) will be saved to the `data/preprocessed_faces` folder (overwriting existing ones if any).

**To preprocess all the meshes automatically and save them:**

Simply select `Preprocess all` at the bottom of the menu. All meshes will be preprocessed with isolue of 3 and saved to the `data/preprocessed_faces` folder (overwriting existing ones if any).



### Landmark Selection UI

<img src="results/UI-landmark_general.png" width="100%">

### Face Registration UI

<img src="results/UI-registration_general.png" width="100%">

**To register a face manually:**

+ Start by selecting a template face in the Combox Box `template face`
+ Also select the scanned face to register in the file browser. All the meshes are preprocessed already.
+ Successively select `Center & Scale face` → `Center & Scale template` → `Align Rigid`
+ With `epsilon` small (e.g. 0.01), click once on `Align Non-Rigid`. The landmarks should be roughly aligned.
+ Set `epsilon` to larger value (e.g. 3.0) and select `Align Non-Rigid` a few times until satisfactory results. 

**To register a face automatically:**

+ Start by selecting a template face and the scanned face to register (same as above).
+ Select `Register`. This does 5 iterations of non-rigid alignment after rigidly aligning the meshes.

**To save the registered face**

Once a face is manually or automatically registered, select `Save registered face`. This will save the mesh in the folder `data/aligned_faces` (overwriting existing one with same name if any!)

**To register all faces and save them**

Simply select `Register all` at the bottom of the menu. All meshes will be registered and saved to the `data/preprocessed_faces` folder (overwriting existing ones if any).

### PCA UI

<img src="results/UI-PCA_general.png" width="100%">
The UI supports the adjustment of the weights for each Eigen face and allows morphing between two faces.

As additional features, we implemented:
+ Changing the amount of Eigen faces used to approximate the original faces
+ Sliders for Eigen faces allowing to slide from the minimal weight of all original faces for each Eigen face to the maximal weight for reasonable adjustment of each weight, but still providing a big enough space to explore features
+ A function to show the error between the shown, and the original face
+ The PCA user interface starts with the general viewer settings (we assume these settings are well known).

**To prepare the dataset:**
+ `Choose data`: It provides a dropdown to choose the dataset from for easy dataset selection.
+ `Show average face`: It shows the mean face of the dataset. Since it is no face available in the dataset the face index below will be set to -1.
+ `Face index`: The face index interface allows decreasing/increasing the face index and scroll through each face in the dataset.
+ `Show face`: This shows the currently selected face from the dataset should any other mesh have been displayed in the meantime.
<img src="results/6-UI-PCA-choose-dataset.gif" width="1000" height="500" />

**To compute linear combinations of Eigen faces:**
+ `#Eigen faces`: The next integer input allows the user to adjust the amount of Eigen faces used to adjust the face offset computed through a linear combination of all Eigen faces weighted with the chosen weight in the range [-1,1]. The weights of each Eigen face can be adjusted through the listed sliders starting at Eigen face 0.
+ `Approximate face with Eigen faces`: This shows the original face approximated using the Eigen faces with weights chosen to minimize the distance. 
+ `Set weight approximated face`: This set the weights chosen by the above sliders according to the current face index and tries to approximate it as close as possible.
+ `Show face with current weights`: This button lets you display the mean face with a linear combination of Eigen faces weighted according to the weight specified using the sliders for each Eigen face. 
+ `Show error to face index`: This displays a coloured visualization of the distance between the computed offset using a linear combination of the Eigen faces added to the mean face and the face chosen by the face index.
<img src="results/6-UI-PCA-eigen-faces.gif" width="1000" height="500" />

**To morph between two faces:**
+ `Morph face index`: Specifies the face index of a face from the dataset which we will morph with the linear combination of Eigen faces currently chosen.
+ `Morph rate`: This scalar lets the user linearly interpolate between the two faces chosen for the morph process.
+ `Show morphed face`: This button lets the user display the result of the morph process should any other mesh have been displayed in the meantime.
+ `Save mesh`: This button lets the user save the displayed mesh to the folder `data/pca-results`.
<img src="results/6-UI-PCA-morph-face.gif" width="1000" height="500" />


