# crnn-image-recognition
## Methodology
The task involves recognizing and reconstructing text sequences from highly distorted grayscale images. The images contain various real-world challenges such as background noise, overlapping characters, blur, shape deformation, occlusion, and irregular spacing.

To solve this, we adopted a CRNN (Convolutional Recurrent Neural Network) architecture combined with CTC (Connectionist Temporal Classification) loss. This approach is well-suited for sequence recognition tasks as it does not require explicit character segmentation.

The CNN extracts visual features from the input image, while the BiLSTM captures sequential dependencies across the feature map. CTC loss enables the model to learn alignment between the visual features and the target character sequence without needing pre aligned labels. During inference, CTC greedy decoding is used to convert the model's output into the final text string.

## Model Architecture
We used a custom CRNN model consisting of two main components:

CNN Feature Extractor: A 5-layer convolutional network that progressively downsamples the input image (256×64) while increasing feature depth. It uses BatchNorm and ReLU activations after each convolution, followed by MaxPooling layers (with asymmetric strides) to reduce spatial dimensions while preserving width for sequence modeling.

Sequence Modeling: A 2-layer Bidirectional LSTM with 256 hidden units per direction. The LSTM processes the feature sequence of length 128 and hidden size 512 (after concatenation of forward and backward passes).

Output Layer: A fully connected layer that maps the LSTM output to the character vocabulary size (including the CTC blank token).

The model takes a grayscale image as input and outputs a probability distribution over characters at each time step, which is then decoded using CTC.

## Training Strategy
Loss Function: nn.CTCLoss with blank=0 and zero_infinity=True.

Optimizer: Adam optimizer with an initial learning rate of 0.0005.

Learning Rate Scheduler: ReduceLROnPlateau (factor=0.5, patience=3) that reduces learning rate when validation loss plateaus.

Gradient Clipping: Applied with max_norm=5.0 to stabilize training.

Data Preprocessing: All images were resized to 256×64 pixels and normalized using mean and standard deviation of 0.5.

Data Split: 90% of the data was used for training and 10% was held out as a validation set.

Training Duration: The model was trained for 50 epochs with a batch size of 64.

Device: Training was performed on CPU due to MPS (Apple Silicon) incompatibility with CTCLoss.

Model Checkpointing: The model with the lowest validation Character Error Rate (CER) was saved as crnn_model_best.pth.

## Results
Character Error Rate (CER) of 0.02% is obtained

