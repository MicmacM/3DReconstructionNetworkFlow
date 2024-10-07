# 3DReconstructionNetworkFlow
Image segmentation and 3D reconstruction from pictures using flow networks

## About
This is my work for the TIPE, a French exam for preparatory classes to the grandes écoles. After several rounds of research, I was able to develop a nearly identical method to solve both the problem of segmentation and of 3D reconstruction.

## How it works
### The max-flow min-cut theorem

First demonstrated by Ford and Fulkerson, this theorem allows one to compute a minimal cut in a network flow in polynomial time, by computing it's maximum flow. The overall approach is explained in the file [`Pdf_final`](PDF_FINAL.pdf), it is simply a matter of well defining the capacity of the edges in the flow network to match the problem.

## Results
The segmentation part works but not that well, but ther techniques solve this problem in a much better way. However it's still not bad considering the simplicity of the algorithm. 


Then nearly the same algorithm is applied to solve the 3D reconstruction problem, where it worked with nearly no adjustment.

## Workspace description

- The [`image_drawing.py`](Segmentation/image_drawing.py) file is a (very VERY) basic tkinter interface that allows one to select the object and background seeds used in the segmentation process
- The [`interactive_image_segmentation.py`](Segmentation/interactive_image_segmentation.py) is the core of the segmentation. An exemple of such a segmentation is provided at the bottom of the file (I wasn't aware of the separation between function file and test files at the time I wrote the program, so I left it that way for the authenticity)

- The [`solid_angle.py`](3DReconstruction/solid_angle.py) file computes the voxels that are seen by a camera. The calculation process isn't based on any ulterior research, I came up with it using some linear algebra but it is really inneficient altough it works surprisingly well

- The [`voxelOccupancy.py`](3DReconstruction/voxelOccupancy.py) is the core of the 3D reconstruction. An exemple of the segmentation is provided at the bottom of the file, it is quite complicated to set-up but feel free to try, I didn't have much time at the time to simplify it.

- The [`array_to_3D.py`](3DReconstruction/array_to_3D.py) file allows you to visualise the result of the 3D reconstruction in blender, by placing a cube at each occupied voxel.

## Bibliography

Finally the bibliography as well as other informations are availible at [MCOTIPE.pdf](MCOTIPE.pdf)
