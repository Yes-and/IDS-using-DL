from .dnn import build_dnn


def build_model(config):
    """Build and return a compiled model from a config dict.

    Reads config["arch"] to select the architecture.
    config["input_dim"] and config["num_classes"] must be set before calling.

    To add a new architecture:
        1. Implement build_<name>(input_dim, num_classes) in its own module.
        2. Import it here and add an elif branch below.
    """
    arch = config["arch"]
    input_dim = config["input_dim"]
    num_classes = config["num_classes"]

    if arch == "dnn":
        return build_dnn(input_dim, num_classes)

    else:
        raise ValueError(f"Unsupported architecture: {arch!r}")
