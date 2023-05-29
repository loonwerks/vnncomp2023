# VNNComp2023 - Collins Aerospace benchmark problem

## Content

[Collins Aerospace Applied Research & Technology](https://www.collinsaerospace.com/what-we-do/technology-and-innovation/applied-research-and-technology) provides a benchmark problem for the [2023 International Verifification of Neural Networks Competition (VNNComp)](https://sites.google.com/view/vnn2023). Proposed use case is an object detection system for unmanned aerial vehicles (UAVs) that perform maritime search and rescue missions. The benchmark is related to robustness against pixel modifications in the neighborhood of the objects to be detected by the system (e.g., persons, boats). The system contains a YOLOv5 nano neural network. Robustness properties are formulated on raw inputs and raw output tensors of YOLO, which are numerous, which makes the problem challenging.

More details are available in the benchmark description file (PDF).

The submission includes:

- **data** folder: test images (.jpeg) used to generate robustness properties
- **onnx** folder: trained YOLO neural network
- **vnnlib** folder: generated vnnlib specifications
- **benchmark_description.pdf**: information about the use case, model, properties, etc.
- **instances.csv**: description of verification instances with respective timeouts
- **generate_properties.py**: python script to generate properties, parameterized with random seed

## License

- All artifacts except the data are distributed under AGPL-3.0 license.
- Data (images) are used and redistributed under license from the dataset owner (https://seadronessee.cs.uni-tuebingen.de/home).
