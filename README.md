# Indoor Climbing Route Segmentation and Classification

## Objective:

This project is an honors thesis project at The University of Massachusetts Lowell. The goal of this project is to create a dataset consisting of images of indoor rock climbing (bouldering) walls then train a machine learning model to segment and classify routes by V grade.

## Dataset:

This dataset was created specifically for this project. Every image is annotated with polygons for holds specifically in the foreground of the image. Every polygon has a label with the route id and route grade. Further, there is a label for incomplete routes where routes wrap around the sides of the wall or are obstructed in some way. There is also a label for if a polygon is a volume. Volumes are essentially an extension of the wall and can be used by any route, thus requiring special handling in the model.

## Model:

A Mask R-CNN (Region-based Convolutional Neural Network)