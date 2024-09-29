# Control Allocation via Autoencoders

Control allocation refers to the method of distributing the desired forces and moments required by the vehicle to the control efforts in a system with redundant actuators.  

This project reimplements the work proposed by R. Skulstad et. al 
called [“Constrained control allocation for dynamic ship positioning using deep neural network.”](https://doi.org/10.1016/j.oceaneng.2023.114434) The paper introduces a novel approach for control allocation using deep neural 
networks, specifically deep auto-encoders, and a custom loss function to ensure motion objectives and thruster constraints are satisified. 

The reimplemented work includes data generation, the custom loss functions, and the development of an autoencoder model using MLPs.

## Dependencies

The dependencies for this project are included in the `requirements.txt` file.

The project utilizes Python 3.12.4 with the following packages and verisons:

- numpy==1.26.4
- matplotlib==3.8.4
- torch==2.4.0+cu124
- scikit-learn==1.4.2

To install the required dependencies, run the following command:

```
pip install -r requirements.txt
```

## Running the Model

The `src` folder contains the script that contains all the code developed for this project. It can be run via an IDE or through the following command:

```
python RBE_577_Homework_1_Script.py
```

## Documentation
The `docs` folder contains a report of the project where the methodologies and results are discussed. 
