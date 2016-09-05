FaceTrek Project for the Soft Computing Course and VR/AR Systems Course

Dependencies:

  - Dlib 19.0
  - OpenCV 3.1.0
  - VTK 7.1.0

Initial Commit was working properly, could be optimised using optical flow algorithm.
Additional commits added optical flow optimization, based on Lucas-Kanade approach.

Also, usage of viz module from OpenCV is based on VTK library, so tread carefully there :D

My way of building this is:
  - Get projects from their respective git repositories
  - Set their solutions via CMake
  - Statically build all of them (ensure OpenCV sees VTK build dir, and Dlib sees OpenCV build dir)
  - Set linker properties in your project to link all three of these libraries
  - If necessary, build VTK dinamically, and copy *.dll files to your bin folder (where your projects' *.exe file is)
