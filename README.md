# DeflareMamba
Official Implementation of MM'25 paper "DeflareMamba: Hierarchical Vision Mamba for Contextually Consistent Lens Flare Removal"

## Model Weights
Download the pretrained model: [deflaremamba.pth](https://pan.baidu.com/s/1quwSo7uyBqWqQgz_g1O4gA?pwd=mauz) (提取码: mauz)

## Usage

### Testing
```bash
python test_large.py --input <path_to_input_images> --output <path_to_output_folder> --model_path <path_to_model.pth> --flare7kpp
```

### Evaluation
```bash
python evaluate.py --input <path_to_predicted_images> --gt <path_to_ground_truth> --mask <path_to_mask_images>
```