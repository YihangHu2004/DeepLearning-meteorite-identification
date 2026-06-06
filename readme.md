### How to Generate submission.csv from test_loader

#### 1. Define the Prediction Function
Use the following function to run inference on the test set and save every prediction by image filename.

What you need to pass in:
- `model`: your trained PyTorch model, for example a ResNet model after training is finished
- `loader`: the `test_loader` created from the test dataset
- `device`: the running device, usually `cuda` or `cpu`

What this function returns:
- `id_to_pred`: a dictionary
- the key is the image filename, such as `000001.jpg`
- the value is the predicted label, such as `0` or `1`

```python
def predict(model, loader, device):
   model.eval()  # Switch the model to evaluation mode
   id_to_pred = {}  # Save prediction results in the form: {image_id: predicted_label}

   with torch.no_grad():  # No gradient is needed during testing
      for images, img_paths in tqdm(loader, desc="Predicting", leave=False):
         # Move one batch of images to GPU or CPU
         images = images.to(device, non_blocking=True)

         # Forward pass: the model outputs classification scores for each class
         outputs = model(images)

         # Take the class with the highest score as the final prediction
         preds = torch.argmax(outputs, dim=1).cpu().numpy().tolist()

         for pred, path in zip(preds, img_paths):
            # Keep only the filename, for example:
            # /path/to/test_images/000001.jpg -> 000001.jpg
            image_id = os.path.basename(path)

            # Save prediction result for this image
            id_to_pred[image_id] = int(pred)

   return id_to_pred
```

#### 2. Define the Submission Function
Use `sample_submission.csv` as the standard template file, then fill each image id with its predicted label.

What you need to pass in:
- `id_to_pred`: the dictionary returned by `predict`
- `template_csv_path`: the path to `sample_submission.csv`
- `output_path`: the path where you want to save the final `submission.csv`

What each parameter should be:
- `id_to_pred`
  this should come directly from `predict(model, test_loader, device)`
- `template_csv_path`
   this is usually `os.path.join(TEST_DATA_DIR, "sample_submission.csv")`
- `output_path`
  this is usually something like `os.path.join(OUTPUT_DIR, "submission.csv")`

Why do we read `sample_submission.csv` first?
- because the final submission file must follow the official test image order
- because this file is exactly the one students will receive in the competition package
- because this also helps us check whether some test images were not predicted

```python
def make_submission(id_to_pred, template_csv_path, output_path):
    # Read the official submission template file
    # This file usually contains at least the id column,
    # and may already contain a placeholder label column
    ids_df = pd.read_csv(template_csv_path)

   # Make sure the csv file really contains the id column
   if "id" not in ids_df.columns:
         raise ValueError(f"{template_csv_path} must contain 'id' column")

   # Copy the id list first, then create a new label column
   submission_df = ids_df.copy()

   # Fill each id with its predicted label using the dictionary
   submission_df["label"] = submission_df["id"].map(id_to_pred)

   # If any label is missing, it means some test images were not predicted correctly
   if submission_df["label"].isna().any():
      missing_ids = submission_df.loc[
         submission_df["label"].isna(), "id"
      ].head(5).tolist()
      raise RuntimeError(f"Missing predictions for some ids, examples: {missing_ids}")

   # Convert labels to integer format such as 0 or 1
   submission_df["label"] = submission_df["label"].astype(int)

   # Save the final submission file
   submission_df.to_csv(output_path, index=False)
```

#### 3. Run Prediction and Save submission.csv

This step shows how to call the two functions above.

What you need before running this code:
- `model` has already been trained
- `test_loader` has already been created
- `device` has already been defined
- `TEST_DATA_DIR` points to the dataset root folder containing `sample_submission.csv` and `test_images`

```python
# Step 1: get prediction results from the test set
id_to_pred = predict(model, test_loader, device)

# Step 2: choose where to save the generated csv file
OUTPUT_DIR = "logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
submission_path = os.path.join(OUTPUT_DIR, "submission.csv")

# Step 3: generate the final submission.csv
make_submission(
   id_to_pred=id_to_pred,  # prediction dictionary returned by predict()
   template_csv_path=os.path.join(TEST_DATA_DIR, "sample_submission.csv"),  # official submission template
   output_path=submission_path,  # final csv save path
)

print(f"Kaggle submission file saved to {submission_path}")
```

### Output Description
- `submission.csv` contains two columns: `id` and `label`
- `id` comes from `sample_submission.csv`
- `label` is the predicted class index for each test image
- for a binary classification task, `label` is usually `0` or `1`

With the current dataset layout, `sample_submission.csv` and `test_images` are placed under the same dataset root folder.

Typical file meaning:
- `0`: class 0, for example `non-meteorite`
- `1`: class 1, for example `meteorite`

In short, students only need to remember this workflow:
- use `predict()` to get all predictions on the test set
- use `make_submission()` to turn those predictions into a standard csv file
- submit the generated `submission.csv`

Example:

```csv
id,label
000001.jpg,0
000002.jpg,1
000003.jpg,0
```
