# NeonatalSeizureDetection

## Description

This repository contains code for the implementation of Neonatal Seizure Detection. A typical neonatal seizure and non-seizure event is illustrated below. Continuous EEG signals are filtered and segmented with varying window lengths of 1, 2, 4, 8, and 16 seconds. The data used here for experimentation can be downloaded from [here](https://zenodo.org/record/1280684).

<p>
    <img src="assets/seizure_activity.png" width="500" alt="Seizure Event">
    <img src="assets/non_seizure_activity.png" width="500" alt="Non-seizure Event">
</p>

## Files description

- [dataprocessing.ipynb](https://github.com/vishaln15/NeonatalSeizureDetection/blob/main/datapreprocessing.ipynb) -> Notebook for converting edf files to csv files.
- [filtering.ipynb](https://github.com/vishaln15/NeonatalSeizureDetection/blob/main/filtering.ipynb)      -> Notebook for filtering the input EEG signals in order to observe the specific frequencies.
- [segmentation.ipynb](https://github.com/vishaln15/NeonatalSeizureDetection/blob/main/segmentation.ipynb)   -> Notebook for segmenting the input into appropriate windows lengths and overlaps.
- [features_final.ipynb](https://github.com/vishaln15/NeonatalSeizureDetection/blob/main/features_final.ipynb) -> Notebook for extracting relevant features from segmented data.
- [protoNN_example.py](https://github.com/vishaln15/NeonatalSeizureDetection/blob/main/protoNN_example.py)   -> Script used for running protoNN model using *.npy* files.
- [inference_time.py](https://github.com/vishaln15/NeonatalSeizureDetection/blob/main/inference_time.py)    -> Script used to record and report inference times.
- [knn.ipynb](https://github.com/vishaln15/NeonatalSeizureDetection/blob/main/knn.ipynb)            -> Notebook used to compare results of ProtoNN and kNN models.

## Dependencies

If you are using conda, it is recommended to switch to a new environment.

```
    $ conda create -n myenv
    $ conda activate myenv
    $ conda install pip
    $ pip install -r requirements.txt
```

If you wish to use virtual environment,

```
    $ pip install virtualenv
    $ virtualenv myenv
    $ source myenv/bin/activate
    $ pip install -r requirements.txt
```

# Usage

1. Clone the **ProtoNN** package from [here](https://github.com/microsoft/edgeml/), **antropy** package from [here](https://github.com/raphaelvallat/antropy/), and **envelope_derivative_operator** package from [here](https://github.com/otoolej/envelope_derivative_operator/).

2. Replace the [protoNN_example.py](https://github.com/microsoft/EdgeML/blob/master/examples/pytorch/ProtoNN/protoNN_example.py) with [protoNN_example.py](https://github.com/vishaln15/NeonatalSeizureDetection/blob/main/protoNN_example.py).

3. Prepare the train and test data *.npy* files and save it in a *DATA_DIR* directory.

4. Execute the following command in terminal after preparing the data files. Create an output directory should you need to save the weights of the ProtoNN object as *OUT_DIR*.
    ```
        $ python protoNN_example.py -d DATA_DIR -e 500 -o OUT_DIR
    ```

## Authors

[**Vishal Nagarajan**](https://www.linkedin.com/in/vishalnagarajan/)

[**Ashwini Muralidharan**](https://github.com/Ashwiinii)

[**Deekshitha Sriraman**](https://github.com/dtg311)

## Acknowledgements

ProtoNN built using [EdgeML](https://github.com/microsoft/edgeml/) provided by [Microsoft](https://github.com/microsoft/). Features extracted using [antropy](https://github.com/raphaelvallat/antropy/) and [otoolej](https://github.com/otoolej/envelope_derivative_operator/) repositories. 

## References

[1] Nathan Stevenson, Karoliina Tapani, Leena Lauronen, & Sampsa Vanhatalo. (2018). A dataset of neonatal EEG recordings with seizures annotations [Data set]. Zenodo. https://doi.org/10.5281/zenodo.1280684. 

[2] Gupta, Ankit et al. "ProtoNN: Compressed and Accurate kNN for Resource-scarce Devices." Proceedings of the 34th International Conference on Machine Learning, Sydney, Australia, PMLR 70.