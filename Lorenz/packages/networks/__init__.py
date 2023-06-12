from . import base_model
import torch



def model_initializer(config, mask):
    model_dict = {
        "base": base_model.Transformer(config["block_size"],
                                       config["lorenz_dim"],
                                       config["num_layers"],
                                       config["d_model"],
                                       config["num_heads"],
                                       config["ffc"],
                                       mask=mask
                                       )
    }
    model = model_dict[config["model"]]
    return model
    
def optim_initializer(config, model):
    optim_dict = {
        "ADAM": torch.optim.Adam(model.parameters(), 
                                 lr=config["lr"], 
                                 weight_decay=config["weight_decay"])
    }
    optimizer = optim_dict["ADAM"]
    return optimizer