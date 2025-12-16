# alpha-dent-slicing-aided-training
Instance segmentation of alpha-dent dataset with Slicing aided training and inference.
## File Descriptions

- **/data**: Contains the alpha-dent dataset used for training and testing the model.
- **/src**: Source code for the instance segmentation model and training scripts.
- **/models**: Pre-trained models and checkpoints for inference.
- **/notebooks**: Jupyter notebooks for exploratory data analysis and visualization.
- **/scripts**: Utility scripts for data preprocessing and evaluation.

## Running Instructions

1. Clone the repository:
    ```bash
    git clone https://github.com/brijeshnandaby/alpha-dent-slicing-aided-training.git
    cd alpha-dent-slicing-aided-training
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Prepare the dataset:
    - Place the alpha-dent dataset in the `/data` directory.

4. Train the model + inference:
    ```bash
    python3 main.py --data_dir data --output_dir models
    ```

5. Run inference:
    ```bash
    python src/inference.py --model_path models/your_model.pth --input_dir data/test_images
    ```

6. View results:
    - Check the output in the specified output directory.