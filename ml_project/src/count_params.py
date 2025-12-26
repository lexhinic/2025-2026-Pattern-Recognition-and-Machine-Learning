from prettytable import PrettyTable
from deep_models import get_vit_tiny,get_model

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

if __name__ == "__main__":
    model = get_model('resnet', num_classes=200)
    total_params = count_parameters(model)
    print(f"ResNet模型总参数量: {total_params}")

