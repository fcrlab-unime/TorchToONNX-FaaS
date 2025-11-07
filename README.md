# 💻 Torch-to-ONNX-FaaS Implementation Guide

This document provides detailed instructions for setting up and deploying the **Torch-to-ONNX-FaaS** function using OpenFaaS on a Kubernetes environment.

-----

## OpenFaaS Setup

Follow these steps to install and configure OpenFaaS:

1.  **Install OpenFaaS with Basic Authentication**:
    Secure your deployment using basic authentication:

    ```bash
    sudo KUBECONFIG=/etc/rancher/k3s/k3s.yaml arkade install openfaas --basic-auth
    ```

2.  **Install the OpenFaaS CLI (faas-cli)**:
    Download and install the necessary command-line tool:

    ```bash
    sudo arkade get faas-cli
    ```

3.  **Retrieve the Basic Authentication Password**:
    Decode the gateway password stored in the Kubernetes secret:

    ```bash
    sudo kubectl -n openfaas get secret basic-auth -o jsonpath="{.data.basic-auth-password}" | base64 --decode; echo
    ```

4.  **Log in to OpenFaaS**:
    Authenticate the CLI using the retrieved password:

    ```bash
    sudo faas-cli login --password <password>
    ```

-----

## Function Creation and Deployment

1.  **Build the Function Image**:
    Build the function image based on your stack file (`.yml`):

    ```bash
    sudo faas-cli build -f ./torch-to-onnx-faas.yml --build-arg PYTHON_VERSION=3.11
    ```

2.  **Push the Function Image**:
    Upload the built image to your designated container registry:

    ```bash
    sudo faas-cli push -f ./torch-to-onnx-faas.yml
    ```

3.  **Deploy the Function**:
    Deploy the function onto the OpenFaaS gateway:

    ```bash
    sudo faas-cli deploy -f ./torch-to-onnx-faas.yml
    ```

4.  **Verify the Deployed Functions**:
    Confirm the function status and endpoint:

    ```bash
    sudo faas-cli list
    ```

-----

## Gateway Configuration (Timeout Adjustment)

Model conversion processes can be time-consuming. Adjust the gateway timeouts to prevent premature terminations.

1.  **Update Gateway Timeouts**:
    Edit the gateway deployment configuration:

    ```bash
    kubectl edit deployment gateway -n openfaas
    ```

    Add or modify the following environment variables under `env:` to set the timeout to **15 minutes**:

    ```yaml
    env:
    - name: read_timeout
      value: 15m
    - name: write_timeout
      value: 15m
    - name: upstream_timeout
      value: 15m
    ```

2.  **Restart the Gateway Deployment**:
    Apply the changes:

    ```bash
    kubectl rollout restart deployment gateway -n openfaas
    ```

3.  **Expose the Gateway Locally (for testing)**:
    Use port-forwarding for quick local access:

    ```bash
    sudo kubectl port-forward -n openfaas svc/gateway 8080:8080 &
    ```

-----

## Access Gateway from the Local Network

1.  **Expose the Gateway via NodePort**:
    Change the service type to allow external network access:

    ```bash
    sudo kubectl patch svc gateway -n openfaas -p '{"spec": {"type": "NodePort"}}'
    ```

2.  **Retrieve the Gateway Port**:
    Get the external port mapping for network access:

    ```bash
    sudo kubectl get svc -n openfaas gateway
    ```

    Note the **`NodePort`** value from the output.

-----

## Testing the Function

1.  **Invoke the Function with a Test JSON File**:
    Replace `test.json` with your input configuration file path and invoke the function endpoint:
    ```bash
    curl -X POST -H "Content-Type: application/json" -d @test.json http://127.0.0.1:8080/function/torch-to-onnx-faas
    ```

-----

## Notes and Best Practices

  * Ensure the **Kubernetes cluster** is operational before starting the OpenFaaS installation.
  * The function handler relies on access to **MinIO** (or the configured S3 compatible storage) for model source and weight files.
  * For persistent external access in production, a dedicated **Ingress Controller** is generally preferred over `NodePort`.
  * Thoroughly test MinIO connectivity and the conversion logic locally before live deployment.

-----