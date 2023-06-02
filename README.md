# PyTorch Docker Assignment

Welcome to the PyTorch Docker (session 2) Assignment! This assignment provides an opportunity to work with Docker and PyTorch. It involves creating a Docker image, training a convolutional NN model on the MNIST dataset using docker, and saving checkpoints. Follow the instructions below to run the code successfully.

## Prerequisites

- Docker installed on your system
- Git installed on your system (to clone the repository)

## Getting Started

1. **Clone the repository:**
   ```shell
   git clone https://github.com/The-School-of-AI/emlov3-session-02-Rakesh-Raushan.git

2. **Change to the project directory:**
   ```shell
   cd emlov3-session-02-Rakesh-Raushan
## Running the Code

3. **Running the training script in docker:** Open a terminal and navigate to the project directory (`emlov3-session-02-Rakesh-Raushan`). Run the following command to build the Docker image:

   ```shell
   docker run --name pytorch-docker-cont --rm -v$(pwd):/workspace pytorch-docker python /workspace/train.py

4. **Wait for the training to complete:** The code will begin training a model on the MNIST dataset inside the Docker container. Please wait patiently until the training is finished. You will see the training progress and any logged information in the terminal.
5. **Check the trained model checkpoints:** After the training completes, you can find the trained model checkpoint by doing 
    ```shell
    ls

## Additional Features

- **Resume training from last checkpoint:** If you want to resume training from last checkpoint, use argument --resume.
- **Changing other default arguments:** Additionally you can pass these arguments to override default values:
    ```shell
    --batch-size <value>
    --epochs <value>
    --lr <value>
    --momentum <value>
    --seed <value>
    --log-interval <value>
    --dry-run
- **Customization:** You have the flexibility to customize the Dockerfile or the provided `train.py` script according to your needs. To make modifications:

  1. Open the Dockerfile and make the necessary changes to the environment setup, dependencies, or any other configurations.
  2. Save the Dockerfile.
  3. Open a terminal and navigate to the project directory (`emlov3-session-02-Rakesh-Raushan`).
  4. Rebuild the Docker image by running the following command:

     ```shell
     docker build -t pytorch-docker .
     ```

     This command rebuilds the Docker image based on the updated Dockerfile.

  5. Run the Docker container using the steps mentioned earlier to execute your customized code.
## Troubleshooting

If you encounter any issues or have questions, please follow these steps:

1. **Check the documentation:** Refer to the assignment documentation and README file for any specific instructions or troubleshooting steps provided.

2. **Search for existing issues:** Search the repository's issue tracker to see if someone has encountered a similar problem and if a solution has been provided.

3. **Create a new issue:** If you cannot find a solution to your problem, feel free to create a new issue in the repository's issue tracker. Be sure to provide detailed information about the problem you are facing, including any error messages, steps to reproduce the issue, and your environment setup (operating system, Docker version, etc.).

4. **Wait for a response:** Once you have created a new issue, the maintainers or other community members will review and respond to it as soon as possible. They may request additional information or provide troubleshooting steps to help resolve the issue.

5. **Be proactive:** While waiting for a response, you can also try searching online forums, community websites, or Docker-related resources for similar issues and possible solutions. Sometimes, the community can provide helpful insights and workarounds.

Remember, the more details you provide about the issue, the better chances of receiving an accurate and timely solution.