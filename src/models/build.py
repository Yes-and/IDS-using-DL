from .dnn import build_dnn

def build_model(config, input_dim, num_classes):

    if config["model"]["type"] == "dnn":

        return build_dnn(
            input_dim,
            num_classes,
            config["model"]["hidden_layers"],
            config["model"]["dropout"]
        )

    else:
        raise ValueError("Unsupported model type")