import os
import json
import torch
import logging
import inspect
import tempfile
import onnx
from minio import Minio
from minio.error import S3Error
from urllib.parse import urljoin
from .utils import load_class_from_file, parse_dynamic_args

# ============================================================
# CONFIGURATION
# ============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "172.16.6.62:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = os.getenv("MINIO_SECURE", "0") == "1"

client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=MINIO_SECURE
)

# Fixed MinIO buckets
BUCKET_MODELS = "models"
BUCKET_WEIGHTS = "weights"
BUCKET_OUTPUT = "onnx"


# ============================================================
# ENTRY POINT
# ============================================================
def handle(event, context):
    """OpenFaaS handler: converts a PyTorch model (.pth) to ONNX and uploads it to MinIO."""
    try:
        body = event.get("body", "") if isinstance(event, dict) else getattr(event, "body", "")
        if not body:
            raise ValueError("Empty request body")

        req = json.loads(body)
        result = execute(req)
        return {"statusCode": 200, "body": json.dumps(result)}

    except json.JSONDecodeError as e:
        logging.error("Invalid JSON input", exc_info=True)
        return {"statusCode": 400, "body": json.dumps({"error": "Invalid JSON", "details": str(e)})}
    except Exception as e:
        logging.error("Unhandled exception", exc_info=True)
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}


# ============================================================
# CORE EXECUTION LOGIC
# ============================================================
def execute(req: dict) -> dict:
    python_path = req["python_path"]
    weights_path = req["weights_path"]
    model_class = req["model_class"]
    dynamic_args = parse_dynamic_args(req.get("args", ""))
    input_shape = req.get("input_shape", [3, 224, 224])

    with tempfile.TemporaryDirectory() as tmpdir:
        py_local = os.path.join(tmpdir, os.path.basename(python_path))
        weights_local = os.path.join(tmpdir, os.path.basename(weights_path))
        onnx_local = os.path.join(tmpdir, f"{model_class}.onnx")

        # ------------------------------------------------------------
        # 1. Verify MinIO connectivity
        # ------------------------------------------------------------
        try:
            client.list_buckets()
            logging.info(f"Connected to MinIO at {MINIO_ENDPOINT}")
        except Exception as e:
            raise ConnectionError(f"Unable to connect to MinIO ({MINIO_ENDPOINT}): {e}")

        # ------------------------------------------------------------
        # 2. Download model source + weights
        # ------------------------------------------------------------
        try:
            client.fget_object(BUCKET_MODELS, python_path, py_local)
            logging.info(f"Downloaded model code: {BUCKET_MODELS}/{python_path}")

            client.fget_object(BUCKET_WEIGHTS, weights_path, weights_local)
            logging.info(f"Downloaded weights: {BUCKET_WEIGHTS}/{weights_path}")
        except S3Error as e:
            raise FileNotFoundError(f"Missing model or weights file: {e}")

        # ------------------------------------------------------------
        # 3. Load model
        # ------------------------------------------------------------
        model = load_class_from_file(model_class, os.path.dirname(py_local), py_local, **dynamic_args)
        model.load_state_dict(torch.load(weights_local, map_location=DEVICE))
        model = model.to(DEVICE).eval()

        dummy_input = torch.randn(1, *input_shape, device=DEVICE)

        # ------------------------------------------------------------
        # 4. Export to ONNX (modern + legacy support)
        # ------------------------------------------------------------
        export_onnx(model, dummy_input, onnx_local)

        # ------------------------------------------------------------
        # 5. Validate ONNX correctness
        # ------------------------------------------------------------
        onnx_model = onnx.load(onnx_local)
        onnx.checker.check_model(onnx_model)
        logging.info("✅ ONNX model validated successfully")

        # ------------------------------------------------------------
        # 6. Upload back to MinIO
        # ------------------------------------------------------------
        output_key = f"{model_class}.onnx"
        if not client.bucket_exists(BUCKET_OUTPUT):
            client.make_bucket(BUCKET_OUTPUT)
        client.fput_object(BUCKET_OUTPUT, output_key, onnx_local)
        logging.info(f"Uploaded ONNX model to {BUCKET_OUTPUT}/{output_key}")

        # ------------------------------------------------------------
        # 7. Construct download URL
        # ------------------------------------------------------------
        scheme = "https" if MINIO_SECURE else "http"
        url = urljoin(f"{scheme}://{MINIO_ENDPOINT}/", f"{BUCKET_OUTPUT}/{output_key}")

        return {
            "status": "success",
            "message": "Model exported and validated successfully.",
            "onnx_path": output_key,
            "download_url": url
        }


# ============================================================
# ONNX EXPORTER — FIXED IMPLEMENTATION
# ============================================================
def export_onnx(model, dummy_input, output_path):
    """
    Export model to ONNX as a single file (no external data).
    Handles both modern and legacy PyTorch ONNX exporters.
    """
    logging.info("Starting ONNX export...")

    # Try modern exporter first (PyTorch >= 2.1)
    if hasattr(torch.onnx, "dynamo_export"):
        try:
            logging.info("Attempting torch.onnx.dynamo_export()")
            
            # Export with dynamo
            export_options = torch.onnx.ExportOptions(
                dynamic_shapes=True,
                op_level_debug=False
            )
            
            exported = torch.onnx.dynamo_export(
                model, 
                dummy_input,
                export_options=export_options
            )
            
            # Save with external data disabled
            exported.save(output_path)
            
            # Force inline all data if it was externalized
            ensure_single_file_onnx(output_path)
            
            logging.info(f"✅ ONNX model saved via dynamo_export to {output_path}")
            return
            
        except Exception as e:
            logging.warning(f"dynamo_export() failed: {e}. Falling back to legacy exporter.")

    # Fallback to legacy exporter (PyTorch <= 2.0 or if dynamo fails)
    logging.info("Using torch.onnx.export() (legacy)")
    
    export_args = {
        "model": model,
        "args": (dummy_input,),
        "f": output_path,
        "export_params": True,
        "opset_version": 17,
        "do_constant_folding": True,
        "input_names": ["input"],
        "output_names": ["output"],
        "dynamic_axes": {
            "input": {0: "batch_size"}, 
            "output": {0: "batch_size"}
        }
    }

    # Disable external data format if parameter exists
    sig = inspect.signature(torch.onnx.export)
    if "use_external_data_format" in sig.parameters:
        export_args["use_external_data_format"] = False
        logging.info("Set use_external_data_format=False")

    torch.onnx.export(**export_args)
    
    # Ensure the model is a single file
    ensure_single_file_onnx(output_path)
    
    logging.info(f"✅ ONNX model exported via torch.onnx.export() to {output_path}")


def ensure_single_file_onnx(onnx_path):
    """
    Ensures ONNX model is stored as a single file by converting external data to inline.
    This fixes the issue where models reference missing .onnx.data files.
    """
    try:
        # Check if external data file exists
        external_data_path = onnx_path + ".data"
        
        if os.path.exists(external_data_path):
            logging.info(f"Found external data file: {external_data_path}")
            logging.info("Converting to single-file format...")
            
            # Load model with external data
            model = onnx.load(onnx_path, load_external_data=True)
            
            # Save with all data inline (no external data)
            onnx.save(
                model, 
                onnx_path,
                save_as_external_data=False
            )
            
            # Clean up external data file
            if os.path.exists(external_data_path):
                os.remove(external_data_path)
                logging.info(f"Removed external data file: {external_data_path}")
            
            logging.info("✅ Converted to single-file ONNX format")
        else:
            # Verify model loads correctly
            model = onnx.load(onnx_path)
            
            # Check if model references external data
            for tensor in model.graph.initializer:
                if tensor.HasField('data_location') and tensor.data_location == onnx.TensorProto.EXTERNAL:
                    logging.warning("Model references external data but file is missing!")
                    logging.info("Attempting to re-save as single file...")
                    
                    # Re-save as single file
                    onnx.save(model, onnx_path, save_as_external_data=False)
                    logging.info("✅ Re-saved as single-file ONNX")
                    break
                    
    except Exception as e:
        logging.warning(f"Could not verify/convert ONNX format: {e}")
        # Continue anyway - the export might still be valid