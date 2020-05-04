***Recommendation***

- Our U-net based work for Visceral Fat segmentation.


- Prerequisites

    - Tensorflow 1.3 
    - Python > 3.5

- Dataset

    - Creat your data in seperate folders so that during training:
                -train 
                            1-images (images with prefix "image")
                            2-labels (same name as images with prefix "mask")
      and 
                -validation
                            1-images (images with prefix "image")
                            2-labels (same name as images with prefix "mask")
    - during the test:
                -test
                        -Test_results_A2B (images obtained from unpaired cycleGAN)
                        
- Example of training

    ```console
    CUDA_VISIBLE_DEVICES=0 python main.py --Input_dir --> .../data
    ```

- Example of testing

    Results will be saved at  -Predictions/Results_V

    ```console
    CUDA_VISIBLE_DEVICES=0 python main.py --Input_dir ./data --mode test
    ```
